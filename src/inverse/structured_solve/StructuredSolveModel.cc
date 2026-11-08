#include "StructuredSolveModel.h"
#include "operations/Operation.h"
#include "semantic/SourceViews.h"
#include "FormulaLogger.h"
#include <stdexcept>

namespace {
class StructuredSolveOperation : public Operation {
  uint32_t M,U,R,B,K;json task_,policy_;std::string variant_;
 public:
  StructuredSolveOperation(SimulationConfig c,Model* m,uint32_t nr,uint32_t nt,uint32_t rhs,uint32_t batches,uint32_t k,json task,json policy,std::map<std::string,std::string>& attrs,std::string variant)
    :Operation(c,m,variant,attrs,0),M(2*nr),U(2*nt),R(2*rhs),B(batches),K(k),task_(task),policy_(policy),variant_(variant){_optype=variant;}
  void initialize_tiles(MappingTable&) override {
    if((_config.precision!=8 && _config.precision!=2) || _config.dram_req_size!=64 || _config.num_cores!=24 || !B || B%24 || B>384 || K!=1 || !U || U>32 || U%4 || !R || M<U)
      throw std::runtime_error("Cholesky explicit requires structure precision8/request64/core24 and valid dimensions");
    for(uint32_t b=0;b<B;++b){
      auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=b,.core_id=static_cast<int>(b%24)});
      initialize_instructions(tile.get(),Mapping{});_tiles.push_back(std::move(tile));
    }
  }
 private:
  void initialize_instructions(Tile* tile,Mapping) override {
    const uint32_t P=_config.precision;const uint32_t BS=4;bool ldl=variant_=="ldl_block_explicit_v1",merge=variant_=="cholesky_merge_explicit_v1";addr_type next=SPAD_BASE;json regions=json::array();
    auto alloc=[&](const std::string& n,uint32_t rows,uint32_t cols){addr_type a=next;next+=ceil_div(uint64_t(rows)*cols*P,64)*64;regions.push_back(region(a,rows,cols,n));return a;};
    auto h=alloc("H",M,U),y=alloc("Y",M,R),reg=alloc("RegI",U,U),eye=alloc("Identity",U,U);
    auto a=alloc("A",U,U),rhs=alloc("rhs",U,R),l=alloc("L",U,U);
    auto forward=alloc("forward",U,R),back=alloc("backwork",U,R);
    auto d=alloc("D",1,U),weighted=alloc("weighted",U,BS);
    auto t=alloc("T",U,std::max(U,R)),negT=alloc("negativeT",U,std::max(U,R));
    auto zero=alloc("zero",1,1),one=alloc("one",1,1),neg=alloc("negative_one",1,1);
    const uint32_t sync_bytes=64*(U*U+16*U+64);
    addr_type sync=next;next+=sync_bytes;regions.push_back(region(sync,1,sync_bytes/P,"sync"));
    addr_type out=ACCUM_SPAD_BASE;regions.push_back(region(out,U,R,"xhat"));
    record_source_regions(regions,M,U);_numeric_header["profile"]=P==8?"structure_f64":"modeled_fp16_fp32_v1";if(!policy_.is_null())_numeric_header["precision_policy"]=policy_;_numeric_header["schema"]="source-views-program-v1";
    _numeric_header["task_contract"]=task_;
    _numeric_header["algorithm"]={{"id",variant_},{"factorization",ldl?"blocked_right_looking_ldl":merge?"left_looking_gemm_reduction":"blocked_right_looking_cholesky"},{"block_size_real",merge?1:4},{"initialization",ldl?"L=I;D;row_weighted_solve":merge?"L=0;gemm_triangular_solve":"L=0;row_triangular_solve"}};
    FormulaLogger::instance().set_algorithm(variant_,merge?1:4,0,U);
    uint32_t serial=0;std::string prefix="batch"+std::to_string(tile->batch)+".";
    auto barrier=[&](addr_type written){
      auto i=source_instruction(Instruction{.opcode=Opcode::PIPE_BARRIER,.id=prefix+"sync"+std::to_string(serial++),.dest_addr=sync,.size=1,.src_addrs={written},.src_from_accum=true,.my_tile=tile,.is_barrier=true},json::array(),nullptr,"structured_solve:sync");
      sync+=64;tile->instructions.push_back(std::move(i));
    };
    auto arithmetic=[&](Opcode code,const std::string& id,json src,json dst,bool broadcast=false,uint32_t inner=0){
      uint32_t rows=dst["rows"],cols=dst["cols"];bool scalar=int(code)>=17 && int(code)<=21;
      bool gemm=code==Opcode::GEMM_PRELOAD;
      Instruction raw{.opcode=code,.id=prefix+id,.dest_addr=dst["base"].get<addr_type>(),.size=uint32_t(ceil_div(uint64_t(rows)*cols*P,64)),
         .compute_size=scalar?1u:gemm?cols:rows*cols*(P==2?4:P),.tile_m=rows,.tile_k=gemm?inner:cols,.tile_n=cols,.src_from_accum=true,.my_tile=tile};
      std::vector<std::string> names;std::vector<std::vector<uint32_t>> shapes;
      for(const auto& v:src){raw.src_addrs.push_back(v["base"]);names.push_back(std::to_string(v["base"].get<addr_type>()));uint32_t m=v["rows"],n=v["cols"];if(v["transpose"].get<bool>())std::swap(m,n);shapes.push_back({m,n});}
      auto inst=source_instruction(std::move(raw),src,dst,"structured_solve:"+id);
      if(broadcast)inst->numeric_semantics["broadcast"]="scalar_rhs";
      FormulaLogger::instance().emit_step(prefix+id,gemm?"GEMM":"ELEMENTWISE",names,std::to_string(dst["base"].get<addr_type>()),shapes,{rows,cols},tile->batch,prefix+id);
      tile->instructions.push_back(std::move(inst));barrier(dst["base"]);
    };
    auto load=[&](uint32_t slot,addr_type local,uint32_t rows,uint32_t cols,bool batched){
      addr_type dram=get_operand_addr(_INPUT_OPERAND+slot)+(batched?uint64_t(tile->batch)*rows*cols*P:0);
      Instruction raw{.opcode=Opcode::MOVIN,.id=prefix+"load"+std::to_string(slot),.dest_addr=local,.size=uint32_t(ceil_div(uint64_t(rows)*cols*P,64)),.operand_id=_INPUT_OPERAND+slot,.base_addr=0,.tile_m=rows,.tile_k=cols,.my_tile=tile};
      for(uint32_t j=0;j<raw.size;++j)raw.src_addrs.push_back(dram+uint64_t(j)*64);
      tile->instructions.push_back(source_instruction(std::move(raw),json::array({view(dram,0,0,rows,cols)}),view(local,0,0,rows,cols),"structured_solve:load"));barrier(local);
    };
    load(0,h,M,U,true);load(1,y,M,R,true);load(2,reg,U,U,false);load(3,eye,U,U,false);
    arithmetic(Opcode::GEMM_PRELOAD,"gram",json::array({view(h,0,0,M,U,true),view(h,0,0,M,U)}),view(a,0,0,U,U),false,M);
    arithmetic(Opcode::ADD,"regularize",json::array({view(a,0,0,U,U),view(reg,0,0,U,U)}),view(a,0,0,U,U));
    arithmetic(Opcode::GEMM_PRELOAD,"rhs",json::array({view(h,0,0,M,U,true),view(y,0,0,M,R)}),view(rhs,0,0,U,R),false,M);
    arithmetic(Opcode::SCALAR_SUB,"zero",json::array({view(eye,0,0),view(eye,0,0)}),view(zero,0,0));
    arithmetic(Opcode::SCALAR_DIV,"one",json::array({view(reg,0,0),view(reg,0,0)}),view(one,0,0));
    arithmetic(Opcode::SCALAR_SUB,"neg1",json::array({view(zero,0,0),view(one,0,0)}),view(neg,0,0));
    arithmetic(Opcode::MUL,"initL",json::array({view(eye,0,0,U,U),view(ldl?one:zero,0,0)}),view(l,0,0,U,U),true);
    auto subtract=[&](const std::string& id,json left,json product,json dst){
      uint32_t rows=dst["rows"],cols=dst["cols"];
      arithmetic(Opcode::MUL,id+".neg",json::array({product,view(neg,0,0)}),view(negT,0,0,rows,cols),true);
      arithmetic(Opcode::ADD,id+".sub",json::array({left,view(negT,0,0,rows,cols)}),dst);
    };
    if(merge){
      for(uint32_t j=0;j<U;++j){auto id="factor"+std::to_string(j);
        if(j){
          arithmetic(Opcode::GEMM_PRELOAD,id+".diagonalDot",json::array({view(l,j,0,1,j),view(l,j,0,1,j,true)}),view(t,0,0),false,j);
          arithmetic(Opcode::SCALAR_SUB,id+".diagonalSub",json::array({view(a,j,j),view(t,0,0)}),view(a,j,j));
        }
        arithmetic(Opcode::SCALAR_SQRT,id+".sqrt",json::array({view(a,j,j)}),view(l,j,j));
        if(j+1<U){uint32_t n=U-j-1;
          if(j){
            arithmetic(Opcode::GEMM_PRELOAD,id+".columnDot",json::array({view(l,j+1,0,n,j),view(l,j,0,1,j,true)}),view(t,0,0,n,1),false,j);
            subtract(id,view(a,j+1,j,n,1),view(t,0,0,n,1),view(a,j+1,j,n,1));
          }
          arithmetic(Opcode::DIV,id+".column",json::array({view(a,j+1,j,n,1),view(l,j,j)}),view(l,j+1,j,n,1),true);
        }
      }
    }else{
      for(uint32_t start=0;start<U;start+=BS){uint32_t end=start+BS;auto panel="panel"+std::to_string(start);
        // Exact block POTRF/LDL with every internal update spelled out.
        for(uint32_t j=start;j<end;++j){auto id=panel+".factor"+std::to_string(j);
          if(ldl)arithmetic(Opcode::SCALAR_MUL,id+".D",json::array({view(a,j,j),view(one,0,0)}),view(d,0,j));
          else arithmetic(Opcode::SCALAR_SQRT,id+".sqrt",json::array({view(a,j,j)}),view(l,j,j));
          if(j+1<end){uint32_t n=end-j-1;
            arithmetic(Opcode::DIV,id+".column",json::array({view(a,j+1,j,n,1),ldl?view(d,0,j):view(l,j,j)}),view(l,j+1,j,n,1),true);
            if(ldl)arithmetic(Opcode::MUL,id+".weighted",json::array({view(l,j+1,j,n,1),view(d,0,j)}),view(weighted,0,0,n,1),true);
            arithmetic(Opcode::GEMM_PRELOAD,id+".outer",json::array({ldl?view(weighted,0,0,n,1):view(l,j+1,j,n,1),view(l,j+1,j,n,1,true)}),view(t,0,0,n,n),false,1);
            subtract(id,view(a,j+1,j+1,n,n),view(t,0,0,n,n),view(a,j+1,j+1,n,n));
          }
        }
        if(end<U){uint32_t tail=U-end;
          // TRSM: solve off-diagonal panel column by column, not one fake op.
          for(uint32_t j=start;j<end;++j){auto id=panel+".trsm"+std::to_string(j);
            arithmetic(Opcode::DIV,id+".divide",json::array({view(a,end,j,tail,1),ldl?view(d,0,j):view(l,j,j)}),view(l,end,j,tail,1),true);
            if(ldl)arithmetic(Opcode::MUL,id+".weighted",json::array({view(l,end,j,tail,1),view(d,0,j)}),view(weighted,0,j-start,tail,1),true);
            if(j+1<end){uint32_t n=end-j-1;
              arithmetic(Opcode::GEMM_PRELOAD,id+".outer",json::array({ldl?view(weighted,0,j-start,tail,1):view(l,end,j,tail,1),view(l,j+1,j,n,1,true)}),view(t,0,0,tail,n),false,1);
              subtract(id,view(a,end,j+1,tail,n),view(t,0,0,tail,n),view(a,end,j+1,tail,n));
            }
          }
          arithmetic(Opcode::GEMM_PRELOAD,panel+".schurGemm",json::array({ldl?view(weighted,0,0,tail,BS):view(l,end,start,tail,BS),view(l,end,start,tail,BS,true)}),view(t,0,0,tail,tail),false,BS);
          subtract(panel+".schur",view(a,end,end,tail,tail),view(t,0,0,tail,tail),view(a,end,end,tail,tail));
        }
      }
    }
    for(uint32_t j=0;j<U;++j){auto id="forward"+std::to_string(j);
      if(merge && j){
        arithmetic(Opcode::GEMM_PRELOAD,id+".dot",json::array({view(l,j,0,1,j),view(forward,0,0,j,R)}),view(t,0,0,1,R),false,j);
        subtract(id,view(rhs,j,0,1,R),view(t,0,0,1,R),view(rhs,j,0,1,R));
      }
      arithmetic(ldl?Opcode::MUL:Opcode::DIV,id+".solve",json::array({view(rhs,j,0,1,R),ldl?view(one,0,0):view(l,j,j)}),view(forward,j,0,1,R),true);
      if(!merge && j+1<U){uint32_t n=U-j-1;
        arithmetic(Opcode::GEMM_PRELOAD,id+".outer",json::array({view(l,j+1,j,n,1),view(forward,j,0,1,R)}),view(t,0,0,n,R),false,1);
        subtract(id,view(rhs,j+1,0,n,R),view(t,0,0,n,R),view(rhs,j+1,0,n,R));
      }
    }
    if(ldl){for(uint32_t j=0;j<U;++j)arithmetic(Opcode::DIV,"rowWeight"+std::to_string(j),json::array({view(forward,j,0,1,R),view(d,0,j)}),view(back,j,0,1,R),true);}
    else arithmetic(Opcode::MUL,"copyForward",json::array({view(forward,0,0,U,R),view(one,0,0)}),view(back,0,0,U,R),true);
    for(int j=int(U)-1;j>=0;--j){auto id="backward"+std::to_string(j);
      if(merge && j+1<int(U)){uint32_t n=U-j-1;
        arithmetic(Opcode::GEMM_PRELOAD,id+".dot",json::array({view(l,j+1,j,n,1,true),view(out,j+1,0,n,R)}),view(t,0,0,1,R),false,n);
        subtract(id,view(back,j,0,1,R),view(t,0,0,1,R),view(back,j,0,1,R));
      }
      arithmetic(ldl?Opcode::MUL:Opcode::DIV,id+".solve",json::array({view(back,j,0,1,R),ldl?view(one,0,0):view(l,j,j)}),view(out,j,0,1,R),true);
      if(!merge && j>0){uint32_t n=j;
        arithmetic(Opcode::GEMM_PRELOAD,id+".outer",json::array({view(l,j,0,1,n,true),view(out,j,0,1,R)}),view(t,0,0,n,R),false,1);
        subtract(id,view(back,0,0,n,R),view(t,0,0,n,R),view(back,0,0,n,R));
      }
    }
    addr_type dram=get_operand_addr(_OUTPUT_OPERAND)+uint64_t(tile->batch)*U*R*P;
    Instruction store{.opcode=Opcode::MOVOUT,.id=prefix+"store",.dest_addr=out,.size=uint32_t(ceil_div(uint64_t(U)*R*P,64)),.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=U,.tile_k=R,.src_from_accum=true,.last_inst=true,.my_tile=tile};
    for(uint32_t j=0;j<store.size;++j)store.src_addrs.push_back(dram+uint64_t(j)*64);
    tile->instructions.push_back(source_instruction(std::move(store),json::array({view(out,0,0,U,R)}),view(dram,0,0,U,R),"structured_solve:store"));
  }
};
}
void StructuredSolveModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
  uint32_t M=_model_config.value("matrix_m",64u),U=_model_config.value("matrix_k",16u),R=_model_config.value("nrhs",8u),B=_model_config.value("batch_size",96u),K=_model_config.value("iterations",1u);
  std::map<std::string,std::string> attrs;
  auto op=std::make_unique<StructuredSolveOperation>(_config,this,M,U,R,B,K,_model_config.at("task_contract"),_model_config.value("precision_policy",json(nullptr)),attrs,variant_);
  for(auto entry:std::vector<std::pair<std::string,std::vector<uint32_t>>>{{"H",{B,2*M,2*U}},{"Y",{B,2*M,2*R}},{"RegI",{2*U,2*U}},{"Identity",{2*U,2*U}},{"xhat",{B,2*U,2*R}}}){
    bool input=entry.first!="xhat";auto tensor=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),entry.first),entry.second,_config.precision,input);
    uint32_t id=tensor->get_id();if(input)tensor->set_produced();add_tensor(std::move(tensor));if(input)op->add_input(id);else op->add_output(id);
  }
  op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
