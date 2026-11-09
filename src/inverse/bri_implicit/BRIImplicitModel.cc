#include "BRIImplicitModel.h"
#include "operations/Operation.h"
#include "semantic/SourceViews.h"
#include "FormulaLogger.h"
#include <stdexcept>

namespace {
class BRIImplicitOperation : public Operation {
  uint32_t M,U,R,B,K;json task_,policy_;
 public:
  BRIImplicitOperation(SimulationConfig c,Model* m,uint32_t nr,uint32_t nt,uint32_t rhs,uint32_t batches,uint32_t k,json task,json policy,std::map<std::string,std::string>& attrs)
    :Operation(c,m,"BRIImplicitV1",attrs,0),M(2*nr),U(2*nt),R(rhs),B(batches),K(k),task_(task),policy_(policy){_optype="BRIImplicitV1";}
  void initialize_tiles(MappingTable&) override {
    if((_config.precision!=2) || _config.dram_req_size!=64 || _config.num_cores!=24 || !B || B%24 || B>384 || (!K || K>76) || U!=32 || R!=8 || M!=128)
      throw std::runtime_error("BRI implicit requires FP16/request64/core24, Nr64/Nt16/RHS8, batch multiple24<=384 and K1..76");
    for(uint32_t b=0;b<B;++b){
      auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=b,.core_id=static_cast<int>(b%24)});
      initialize_instructions(tile.get(),Mapping{});_tiles.push_back(std::move(tile));
    }
  }
 private:
  void initialize_instructions(Tile* tile,Mapping) override {
    const uint32_t P=_config.precision;addr_type next=SPAD_BASE;json regions=json::array();
    auto alloc=[&](const std::string& n,uint32_t rows,uint32_t cols){addr_type a=next;next+=ceil_div(uint64_t(rows)*cols*P,64)*64;regions.push_back(region(a,rows,cols,n));return a;};
    auto h=alloc("H",M,U),y=alloc("Y",M,2*R),reg=alloc("RegI",U,U),eye=alloc("Identity",U,U);
    auto yy=alloc("packedY",M,R),x=alloc("X",U,R),previous=alloc("previous",U,R),carry=alloc("carry",U,R),value=alloc("new",U,R);
    auto hx=alloc("HX",M,R),res=alloc("data_residual",M,R),negativeM=alloc("negativeM",M,R);
    auto grad=alloc("gradient",U,R),direction=alloc("direction",U,R),tmp=alloc("tmp",U,R),ntmp=alloc("negative",U,R);
    auto delta=alloc("delta",U,R),update=alloc("update",U,R),p=alloc("expandedP",U,R),ones=alloc("ones",U,R);
    auto zero=alloc("zero",1,1),one=alloc("one",1,1),neg=alloc("negative_one",1,1),two=alloc("two",1,1);
    auto hi=alloc("upper",1,1),lo=alloc("lower",1,1),d=alloc("d",1,1),c=alloc("c",1,1),rho=alloc("rho",1,1),rr=alloc("newrho",1,1);
    auto alpha=alloc("alpha",1,1),beta=alloc("beta",1,1),den=alloc("den",1,1),scratch=alloc("scalar",1,1),momentum=alloc("momentum",1,1),energy=alloc("energy",1,1);
    const uint32_t sync_bytes=64*(256+8*U+64*K);
    addr_type sync=next;next+=sync_bytes;regions.push_back(region(sync,1,sync_bytes/P,"sync"));
    addr_type out=ACCUM_SPAD_BASE;regions.push_back(region(out,U,2*R,"xhat"));
    record_source_regions(regions,M,U);_numeric_header["profile"]=P==8?"structure_f64":"modeled_fp16_fp32_v1";if(!policy_.is_null())_numeric_header["precision_policy"]=policy_;_numeric_header["schema"]="source-views-program-v1";
    _numeric_header["task_contract"]=task_;_numeric_header["algorithm"]={{"id","bri_implicit_fp16_v1"},{"iterations",K},{"block_size_complex",1},{"momentum",0.9},{"bounds_lambda_factor",0.25},{"bounds_upper",8.0},{"recurrence","implicit_compensated_damped"},{"representation","left_complex_columns"},{"initialization","x0=previous=carry=0"}};
    FormulaLogger::instance().set_algorithm("bri_implicit_fp16_v1",1,K,U);
    uint32_t serial=0;std::string prefix="batch"+std::to_string(tile->batch)+".";
    auto barrier=[&](addr_type written){
      auto i=source_instruction(Instruction{.opcode=Opcode::PIPE_BARRIER,.id=prefix+"sync"+std::to_string(serial++),.dest_addr=sync,.size=1,.src_addrs={written},.src_from_accum=true,.my_tile=tile,.is_barrier=true},json::array(),nullptr,"bri_implicit:sync");
      sync+=64;tile->instructions.push_back(std::move(i));
    };
    auto arithmetic=[&](Opcode code,const std::string& id,json src,json dst,bool broadcast=false,uint32_t inner=0){
      uint32_t rows=dst["rows"],cols=dst["cols"];bool scalar=int(code)>=17 && int(code)<=21;
      bool gemm=code==Opcode::GEMM_PRELOAD;
      Instruction raw{.opcode=code,.id=prefix+id,.dest_addr=dst["base"].get<addr_type>(),.size=uint32_t(ceil_div(uint64_t(rows)*cols*P,64)),
         .compute_size=scalar?1u:gemm?cols:rows*cols*(P==2?4:P),.tile_m=rows,.tile_k=gemm?inner:cols,.tile_n=cols,.src_from_accum=true,.my_tile=tile};
      std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
      for(const auto& v:src){raw.src_addrs.push_back(v["base"]);names.push_back(std::to_string(v["base"].get<addr_type>()));uint32_t m=v["rows"],n=v["cols"];if(v["transpose"].get<bool>())std::swap(m,n);shapes.push_back({m,n});}
      auto inst=source_instruction(std::move(raw),src,dst,"bri_implicit:"+id);
      if(broadcast)inst->numeric_semantics["broadcast"]="scalar_rhs";
      FormulaLogger::instance().emit_step(prefix+id,gemm?"GEMM":"ELEMENTWISE",names,std::to_string(dst["base"].get<addr_type>()),shapes,{rows,cols},tile->batch,prefix+id);
      tile->instructions.push_back(std::move(inst));barrier(dst["base"]);
    };
    auto load=[&](uint32_t slot,addr_type local,uint32_t rows,uint32_t cols,bool batched){
      addr_type dram=get_operand_addr(_INPUT_OPERAND+slot)+(batched?uint64_t(tile->batch)*rows*cols*P:0);
      Instruction raw{.opcode=Opcode::MOVIN,.id=prefix+"load"+std::to_string(slot),.dest_addr=local,.size=uint32_t(ceil_div(uint64_t(rows)*cols*P,64)),.operand_id=_INPUT_OPERAND+slot,.base_addr=0,.tile_m=rows,.tile_k=cols,.my_tile=tile};
      for(uint32_t j=0;j<raw.size;++j)raw.src_addrs.push_back(dram+uint64_t(j)*64);
      tile->instructions.push_back(source_instruction(std::move(raw),json::array({view(dram,0,0,rows,cols)}),view(local,0,0,rows,cols),"bri_implicit:load"));barrier(local);
    };
    load(0,h,M,U,true);load(1,y,M,2*R,true);load(2,reg,U,U,false);load(3,eye,U,U,false);
    auto scalar=[&](Opcode op,const std::string& id,json left,json right,addr_type dst){arithmetic(op,id,json::array({left,right}),view(dst,0,0));};
    auto mul=[&](const std::string& id,json left,json right,json dst,bool broadcast=false){arithmetic(Opcode::MUL,id,json::array({left,right}),dst,broadcast);};
    auto add=[&](const std::string& id,json left,json right,json dst){arithmetic(Opcode::ADD,id,json::array({left,right}),dst);};
    auto copy=[&](const std::string& id,json left,json dst){mul(id,left,view(one,0,0),dst,true);};
    auto sub=[&](const std::string& id,json left,json right,json dst,addr_type work){
      mul(id+".neg",right,view(neg,0,0),view(work,0,0,right["rows"],right["cols"]),true);
      add(id+".add",left,view(work,0,0,right["rows"],right["cols"]),dst);
    };
    scalar(Opcode::SCALAR_SUB,"zero",view(eye,0,0),view(eye,0,0),zero);
    scalar(Opcode::SCALAR_DIV,"one",view(eye,0,0),view(eye,0,0),one);
    scalar(Opcode::SCALAR_SUB,"negative_one",view(zero,0,0),view(one,0,0),neg);
    scalar(Opcode::SCALAR_ADD,"two",view(one,0,0),view(one,0,0),two);
    scalar(Opcode::SCALAR_MUL,"four",view(two,0,0),view(two,0,0),scratch);
    scalar(Opcode::SCALAR_DIV,"lower",view(reg,0,0),view(scratch,0,0),lo);
    scalar(Opcode::SCALAR_MUL,"eight",view(scratch,0,0),view(two,0,0),hi);
    scalar(Opcode::SCALAR_ADD,"nine",view(hi,0,0),view(one,0,0),momentum);
    scalar(Opcode::SCALAR_ADD,"ten",view(momentum,0,0),view(one,0,0),scratch);
    scalar(Opcode::SCALAR_DIV,"momentum",view(momentum,0,0),view(scratch,0,0),momentum);
    scalar(Opcode::SCALAR_ADD,"sum_bounds",view(hi,0,0),view(lo,0,0),scratch);
    scalar(Opcode::SCALAR_DIV,"d",view(scratch,0,0),view(two,0,0),d);
    scalar(Opcode::SCALAR_SUB,"diff_bounds",view(hi,0,0),view(lo,0,0),scratch);
    scalar(Opcode::SCALAR_DIV,"c",view(scratch,0,0),view(two,0,0),c);
    scalar(Opcode::SCALAR_DIV,"rho_init",view(c,0,0),view(d,0,0),rho);
    copy("packY",view(y,0,0,M,R),view(yy,0,0,M,R));
    mul("initX",view(yy,0,0,U,R),view(zero,0,0),view(x,0,0,U,R),true);
    // Addition with +0 gives the constant +0 signed-zero semantics of the frozen model.
    arithmetic(Opcode::ADD,"positive_zero",json::array({view(x,0,0,U,R),view(zero,0,0)}),view(x,0,0,U,R),true);
    copy("initPrevious",view(x,0,0,U,R),view(previous,0,0,U,R));copy("initCarry",view(x,0,0,U,R),view(carry,0,0,U,R));
    arithmetic(Opcode::ADD,"ones",json::array({view(x,0,0,U,R),view(one,0,0)}),view(ones,0,0,U,R),true);
    copy("allocateP",view(x,0,0,U,R),view(p,0,0,U,R));
    for(uint32_t j=0;j<U/2;++j){
      std::string id="point"+std::to_string(j);
      arithmetic(Opcode::GEMM_PRELOAD,id+".energy",json::array({view(h,0,j,M,1,true),view(h,0,j,M,1)}),view(energy,0,0),false,M);
      scalar(Opcode::SCALAR_ADD,id+".regularize",view(energy,0,0),view(reg,0,0),energy);
      scalar(Opcode::SCALAR_DIV,id+".reciprocal",view(one,0,0),view(energy,0,0),scratch);
      mul(id+".real",view(ones,j,0,1,R),view(scratch,0,0),view(p,j,0,1,R),true);
      mul(id+".imag",view(ones,U/2+j,0,1,R),view(scratch,0,0),view(p,U/2+j,0,1,R),true);
    }
    for(uint32_t k=0;k<K;++k){
      std::string id="iteration"+std::to_string(k);
      if(k==0){
        scalar(Opcode::SCALAR_DIV,id+".alpha",view(one,0,0),view(d,0,0),alpha);
        scalar(Opcode::SCALAR_MUL,id+".beta0",view(zero,0,0),view(one,0,0),beta);
      }else{
        scalar(Opcode::SCALAR_MUL,id+".twod",view(two,0,0),view(d,0,0),den);
        scalar(Opcode::SCALAR_MUL,id+".crho",view(c,0,0),view(rho,0,0),scratch);
        scalar(Opcode::SCALAR_SUB,id+".den",view(den,0,0),view(scratch,0,0),den);
        scalar(Opcode::SCALAR_DIV,id+".newrho",view(c,0,0),view(den,0,0),rr);
        scalar(Opcode::SCALAR_DIV,id+".alpha",view(two,0,0),view(den,0,0),alpha);
        scalar(Opcode::SCALAR_MUL,id+".beta",view(rr,0,0),view(rho,0,0),beta);
        scalar(Opcode::SCALAR_MUL,id+".rho",view(rr,0,0),view(one,0,0),rho);
      }
      scalar(Opcode::SCALAR_MUL,id+".damping",view(beta,0,0),view(momentum,0,0),beta);
      arithmetic(Opcode::GEMM_PRELOAD,id+".HX",json::array({view(h,0,0,M,U),view(x,0,0,U,R)}),view(hx,0,0,M,R),false,U);
      sub(id+".data_residual",view(yy,0,0,M,R),view(hx,0,0,M,R),view(res,0,0,M,R),negativeM);
      arithmetic(Opcode::GEMM_PRELOAD,id+".HtR",json::array({view(h,0,0,M,U,true),view(res,0,0,M,R)}),view(grad,0,0,U,R),false,M);
      mul(id+".lambdaX",view(x,0,0,U,R),view(reg,0,0),view(tmp,0,0,U,R),true);
      sub(id+".gradient",view(grad,0,0,U,R),view(tmp,0,0,U,R),view(grad,0,0,U,R),ntmp);
      arithmetic(Opcode::ADD,id+".gradient_positive_zero",json::array({view(grad,0,0,U,R),view(zero,0,0)}),view(grad,0,0,U,R),true);
      mul(id+".Pg",view(p,0,0,U,R),view(grad,0,0,U,R),view(direction,0,0,U,R));
      sub(id+".difference",view(x,0,0,U,R),view(previous,0,0,U,R),view(tmp,0,0,U,R),ntmp);
      mul(id+".moment",view(tmp,0,0,U,R),view(beta,0,0),view(tmp,0,0,U,R),true);
      mul(id+".alphaPg",view(direction,0,0,U,R),view(alpha,0,0),view(direction,0,0,U,R),true);
      add(id+".delta",view(direction,0,0,U,R),view(tmp,0,0,U,R),view(delta,0,0,U,R));
      sub(id+".update",view(delta,0,0,U,R),view(carry,0,0,U,R),view(update,0,0,U,R),ntmp);
      add(id+".new",view(x,0,0,U,R),view(update,0,0,U,R),view(value,0,0,U,R));
      sub(id+".difference_new",view(value,0,0,U,R),view(x,0,0,U,R),view(tmp,0,0,U,R),ntmp);
      sub(id+".carry",view(tmp,0,0,U,R),view(update,0,0,U,R),view(carry,0,0,U,R),ntmp);
      copy(id+".previous",view(x,0,0,U,R),view(previous,0,0,U,R));
      copy(id+".state",view(value,0,0,U,R),view(x,0,0,U,R));
    }
    mul("allocateOutput",view(y,0,0,U,2*R),view(zero,0,0),view(out,0,0,U,2*R),true);
    copy("output.re",view(x,0,0,U/2,R),view(out,0,0,U/2,R));
    copy("output.im",view(x,U/2,0,U/2,R),view(out,U/2,0,U/2,R));
    copy("output.re_copy",view(x,0,0,U/2,R),view(out,U/2,R,U/2,R));
    mul("output.negative_im",view(x,U/2,0,U/2,R),view(neg,0,0),view(ntmp,0,0,U/2,R),true);
    arithmetic(Opcode::ADD,"output.negative_zero",json::array({view(ntmp,0,0,U/2,R),view(zero,0,0)}),view(out,0,R,U/2,R),true);
    addr_type dram=get_operand_addr(_OUTPUT_OPERAND)+uint64_t(tile->batch)*U*2*R*P;
    Instruction store{.opcode=Opcode::MOVOUT,.id=prefix+"store",.dest_addr=out,.size=uint32_t(ceil_div(uint64_t(U)*2*R*P,64)),.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=U,.tile_k=2*R,.src_from_accum=true,.last_inst=true,.my_tile=tile};
    for(uint32_t j=0;j<store.size;++j)store.src_addrs.push_back(dram+uint64_t(j)*64);
    tile->instructions.push_back(source_instruction(std::move(store),json::array({view(out,0,0,U,2*R)}),view(dram,0,0,U,2*R),"bri_implicit:store"));
  }
};
}
void BRIImplicitModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
  uint32_t M=_model_config.value("matrix_m",64u),U=_model_config.value("matrix_k",16u),R=_model_config.value("nrhs",8u),B=_model_config.value("batch_size",96u),K=_model_config.value("iterations",33u);
  std::map<std::string,std::string> attrs;
  auto op=std::make_unique<BRIImplicitOperation>(_config,this,M,U,R,B,K,_model_config.at("task_contract"),_model_config.value("precision_policy",json(nullptr)),attrs);
  for(auto entry:std::vector<std::pair<std::string,std::vector<uint32_t>>>{{"H",{B,2*M,2*U}},{"Y",{B,2*M,2*R}},{"RegI",{2*U,2*U}},{"Identity",{2*U,2*U}},{"xhat",{B,2*U,2*R}}}){
    bool input=entry.first!="xhat";auto tensor=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),entry.first),entry.second,_config.precision,input);
    uint32_t id=tensor->get_id();if(input)tensor->set_produced();add_tensor(std::move(tensor));if(input)op->add_input(id);else op->add_output(id);
  }
  op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
