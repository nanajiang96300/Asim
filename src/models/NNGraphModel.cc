#include "NNGraphModel.h"
#include "../operations/Operation.h"
#include "../operations/nn/VectorUnits.h"
#include <stdexcept>
namespace {
// Validate the whole small graph before emitting tensors or instructions.
void validate_graph(const json& cfg) {
 const auto& g=cfg.at("graph");
 if(g.at("schema")!="nn-graph-v1" || cfg.at("batch_size")!=96 || g.at("inputs").empty() || g.at("inputs").size()>32 || g.at("nodes").empty() || g.at("nodes").size()>64)
  throw std::runtime_error("invalid NN graph schema/budget");
 using Shape=std::pair<uint32_t,uint32_t>;std::map<std::string,Shape> shapes;
 auto shape=[](const json& a)->Shape {
  if(!a.is_array() || a.size()!=2 || !a[0].is_number_integer() || !a[1].is_number_integer())throw std::runtime_error("NN shape must be two integers");
  int m=a[0],n=a[1];if(m<1||n<1||m>128||n>128||m*n*2%64)throw std::runtime_error("NN invalid shape/tail");return {m,n};
 };
 auto add=[&](const std::string& name,Shape s){if(name.empty() || shapes.count(name))throw std::runtime_error("NN duplicate/empty tensor");shapes[name]=s;};
 for(auto in:g.at("inputs")) {if(in.contains("batched")&&!in.at("batched").is_boolean())throw std::runtime_error("NN batch attribute");add(in.at("name"),shape(in.at("shape")));}
 for(auto node:g.at("nodes")) {
  std::string op=node.at("op"),id=node.at("id");Shape dst=shape(node.at("shape"));std::vector<Shape> src;
  for(auto name:node.at("src")){auto key=name.get<std::string>();if(!shapes.count(key))throw std::runtime_error("NN unknown/forward input");src.push_back(shapes.at(key));}
  if(op=="SOFTMAX"||op=="GELU") {if(src.size()!=1 || src[0]!=dst)throw std::runtime_error("NN unary shape");}
  else if(op=="GEMM") {
   if(src.size()!=2)throw std::runtime_error("NN GEMM arity");auto right=src[1];if(node.value("transpose_rhs",false))std::swap(right.first,right.second);
   if(src[0].second!=right.first || dst!=Shape(src[0].first,right.second))throw std::runtime_error("NN GEMM shape");
  }else if(op=="ADD"||op=="MUL") {
   auto broadcast=node.value("broadcast",std::string("none"));
   if(src.size()!=2 || src[0]!=dst || (broadcast=="none"?src[1]!=dst:broadcast!="row_rhs" || src[1]!=Shape(1,dst.second)))throw std::runtime_error("NN binary shape/broadcast");
  }else throw std::runtime_error("unsupported NN graph operator");
  add(id,dst);
 }
 std::string output=g.at("output");if(!shapes.count(output) || output!=g.at("nodes").back().at("id").get<std::string>())throw std::runtime_error("NN output must be final node");
 auto dims=cfg.at("output_shape").get<std::vector<uint32_t>>();auto s=shapes.at(output);
 if(dims!=std::vector<uint32_t>{96,s.first,s.second})throw std::runtime_error("NN output declaration");
}
class NNGraphOperation:public Operation {
 json cfg_;
 public:
 NNGraphOperation(SimulationConfig c,Model* m,json cfg,std::map<std::string,std::string>& a):Operation(c,m,"NNGraphV1",a,0),cfg_(cfg){_optype="NNGraphV1";}
 void initialize_tiles(MappingTable&) override {
  auto graph=cfg_.at("graph");uint32_t B=cfg_.at("batch_size");
  if(_config.precision!=2 || _config.num_cores!=24 || _config.dram_req_size!=64 || B!=96 || graph["nodes"].empty() || graph["nodes"].size()>64 || graph["inputs"].size()>32)throw std::runtime_error("NN graph budget/profile");
  for(uint32_t b=0;b<B;++b){
   auto tile=std::make_unique<Tile>(Tile{.status=Tile::Status::INITIALIZED,.optype=_optype,.layer_id=_id,.batch=b,.core_id=static_cast<int>(b%24)});
   std::map<std::string,json> views;json regions=json::array();addr_type next=SPAD_BASE,next_acc=ACCUM_SPAD_BASE;
   auto alloc=[&](std::string name,json shape,bool acc){uint32_t m=shape[0],n=shape[1];if(!m||!n||m>128||n>128||m*n*2%64)throw std::runtime_error("NN region shape/tail");auto& cursor=acc?next_acc:next;auto base=cursor;cursor+=uint64_t(m)*n*2;regions.push_back(region(base,m,n,name));views[name]=view(base,0,0,m,n);};
   for(auto in:graph["inputs"])alloc(in.at("name"),in.at("shape"),false);
   for(auto node:graph["nodes"]) {std::string id=node.at("id");if(views.count(id))throw std::runtime_error("duplicate NN tensor");alloc(id,node.at("shape"),true);}
   for(uint32_t core=0;core<24;++core)
    if(uint64_t(next-SPAD_BASE)+32768>uint64_t(_config.core_config[core].spad_size)*1024/2 || uint64_t(next_acc-ACCUM_SPAD_BASE)>uint64_t(_config.core_config[core].accum_spad_size)*1024/2)
     throw std::runtime_error("NN graph exceeds actual bank capacity");
   addr_type sync=next;regions.push_back(region(sync,1,16384,"sync"));
   record_source_regions(regions,1,1);_numeric_header["schema"]="source-views-program-v1";_numeric_header["profile"]="modeled_fp16_fp32_nn_v1";
   _numeric_header["precision_policy"]=cfg_.at("precision_policy");_numeric_header["task_contract"]=cfg_.at("task_contract");_numeric_header["algorithm"]=cfg_.at("algorithm");
   std::string prefix="batch"+std::to_string(b)+".";uint32_t serial=0;
   auto barrier=[&](addr_type written){Instruction i{.opcode=Opcode::PIPE_BARRIER,.id=prefix+"sync"+std::to_string(serial++),.dest_addr=sync,.size=1,.src_addrs={written},.src_from_accum=true,.my_tile=tile.get(),.is_barrier=true};tile->instructions.push_back(source_instruction(std::move(i),json::array(),nullptr,"nn.sync"));sync+=64;};
   uint32_t slot=0;
   for(auto in:graph["inputs"]){std::string name=in["name"];auto v=views.at(name);uint32_t m=v["rows"],n=v["cols"];addr_type dram=get_operand_addr(_INPUT_OPERAND+slot)+(in.value("batched",false)?uint64_t(b)*m*n*2:0);
    Instruction i{.opcode=Opcode::MOVIN,.id=prefix+"load."+name,.dest_addr=v["base"],.size=m*n*2/64,.operand_id=_INPUT_OPERAND+slot,.base_addr=0,.tile_m=m,.tile_k=n,.my_tile=tile.get()};
    for(uint32_t q=0;q<i.size;++q)i.src_addrs.push_back(dram+q*64);
    tile->instructions.push_back(source_instruction(std::move(i),json::array({view(dram,0,0,m,n)}),v,"nn.load."+name));barrier(v["base"]);++slot;
   }
   for(auto node:graph["nodes"]){std::string id=node["id"],op=node["op"];auto dst=views.at(id);uint32_t m=dst["rows"],n=dst["cols"];json src=json::array();for(auto name:node["src"])src.push_back(views.at(name.get<std::string>()));
    Instruction i{.id=prefix+id,.dest_addr=dst["base"],.size=m*n*2/64,.compute_size=m*n*4,.tile_m=m,.tile_k=n,.tile_n=n,.src_from_accum=true,.my_tile=tile.get()};
    if(op=="SOFTMAX"||op=="GELU") {
     if(src.size()!=1)throw std::runtime_error("NN unary arity");
     tile->instructions.push_back(op=="SOFTMAX"?nn_units::softmax(std::move(i),src[0],dst):nn_units::gelu(std::move(i),src[0],dst));
    }else{
     if(src.size()!=2)throw std::runtime_error("NN binary arity");
     if(op=="GEMM"){i.opcode=Opcode::GEMM_PRELOAD;i.compute_size=n;i.tile_k=src[0]["cols"];/* owner negative test: deliberately omit K transpose */}
     else if(op=="ADD")i.opcode=Opcode::ADD;else if(op=="MUL")i.opcode=Opcode::MUL;else throw std::runtime_error("unsupported NN node");
     for(auto v:src)i.src_addrs.push_back(v["base"].get<addr_type>());
     auto ins=source_instruction(std::move(i),src,dst,"nn.node."+id);ins->numeric_semantics["broadcast"]=node.value("broadcast",std::string("none"));tile->instructions.push_back(std::move(ins));
    }barrier(dst["base"]);
   }
   auto v=views.at(graph.at("output").get<std::string>());uint32_t m=v["rows"],n=v["cols"];addr_type dram=get_operand_addr(_OUTPUT_OPERAND)+uint64_t(b)*m*n*2;
   Instruction i{.opcode=Opcode::MOVOUT,.id=prefix+"store",.dest_addr=v["base"],.size=m*n*2/64,.operand_id=_OUTPUT_OPERAND,.base_addr=0,.tile_m=m,.tile_k=n,.src_from_accum=true,.last_inst=true,.my_tile=tile.get()};for(uint32_t q=0;q<i.size;++q)i.src_addrs.push_back(dram+q*64);
   tile->instructions.push_back(source_instruction(std::move(i),json::array({v}),view(dram,0,0,m,n),"nn.store"));_tiles.push_back(std::move(tile));
  }
 }
};
}
void NNGraphModel::initialize_model(std::vector<std::unique_ptr<Tensor>>&) {
 validate_graph(_model_config);
 std::map<std::string,std::string> attrs;auto op=std::make_unique<NNGraphOperation>(_config,this,_model_config,attrs);uint32_t B=_model_config.at("batch_size");auto g=_model_config.at("graph");
 for(auto in:g["inputs"]){std::vector<uint32_t> dims;if(in.value("batched",false))dims.push_back(B);dims.push_back(in["shape"][0]);dims.push_back(in["shape"][1]);auto t=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),in["name"].get<std::string>()),dims,_config.precision,true);t->set_produced();auto id=t->get_id();add_tensor(std::move(t));op->add_input(id);}
 auto dims=_model_config.at("output_shape").get<std::vector<uint32_t>>();auto t=std::make_unique<Tensor>(get_root_node_id(),name_gen(get_name(),"Y"),dims,_config.precision,false);op->add_output(t->get_id());add_tensor(std::move(t));op->initialize_tiles(_mapping_table);_executable_layer.push_back(op.get());_operation_map[op->get_id()]=std::move(op);
}
