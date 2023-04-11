#include "codegen_oraa.h"

#include <tvm/arith/analyzer.h>

#include <cctype>
#include <iomanip>

#include "../../arith/pattern_match.h"
#include "../../printer/meta_data.h"
#include "../../printer/text_printer.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenORAA::Init() {
  decl_stream << "from dla2.client import get_remote_runtime\n";
  decl_stream << "from dla2.core.flags import TensorFormat\n";
  decl_stream << "api = get_remote_runtime()\n";
}

void CodeGenORAA::InitFuncState(const PrimFunc& f) {
  CodeGenSourceBase::ClearFuncState();
  // TODO(Chunwei Xia) For now we hack the blockIdx and threadIdx to variable map
  // auto n = make_object<VarNode>("blockIdx.x", GetRef<VarType>);
  // var_idmap_[] = "blockIdx.x";
}

void CodeGenORAA::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->ReserveName("_");
  name_supply_->ReserveName("and");
  name_supply_->ReserveName("as");
  name_supply_->ReserveName("assert");
  name_supply_->ReserveName("break");
  name_supply_->ReserveName("class");
  name_supply_->ReserveName("continue");
  name_supply_->ReserveName("def");
  name_supply_->ReserveName("del");
  name_supply_->ReserveName("elif");
  name_supply_->ReserveName("else");
  name_supply_->ReserveName("except");
  name_supply_->ReserveName("False");
  name_supply_->ReserveName("finally");
  name_supply_->ReserveName("for");
  name_supply_->ReserveName("from");
  name_supply_->ReserveName("global");
  name_supply_->ReserveName("if");
  name_supply_->ReserveName("import");
  name_supply_->ReserveName("in");
  name_supply_->ReserveName("is");
  name_supply_->ReserveName("lambda");
  name_supply_->ReserveName("None");
  name_supply_->ReserveName("nonlocal");
  name_supply_->ReserveName("not");
  name_supply_->ReserveName("or");
  name_supply_->ReserveName("pass");
  name_supply_->ReserveName("raise");
  name_supply_->ReserveName("return");
  name_supply_->ReserveName("True");
  name_supply_->ReserveName("try");
  name_supply_->ReserveName("while");
  name_supply_->ReserveName("with");
  name_supply_->ReserveName("yield");
}

void CodeGenORAA::PrintFuncPrefix(std::ostream& os) { os << "def"; }

void CodeGenORAA::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenORAA: Expect PrimFunc to have the global_symbol attribute";
  // bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  // this->PrintExtraAttrs(f);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  // For now, we only print the name of params
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    stream << " " << vid;
  }
  stream << "):\n";
  // this->PrintIndent();
  TextMetaDataContext meta;
  tir::TIRTextPrinter printer(false, &meta);

  int func_scope = this->BeginScope();
  // Allocate Tensor
  // const auto* op = f.operator->();
  // uint32_t offset = 0;
  // for (Var v : op->params) {
  //   std::string vid = AllocVarID(v.get());
  //   auto it = op->buffer_map.find(v);
  //   if(it != op->buffer_map.end()){
  //     stream << vid << " = " << "api.declare_tensor(mem, " << offset;
  //     stream << printer.Print((*it).second).str();
  //     stream << ", TensorFormat.NCHW)\n";
  //   }
  // }
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
}

void CodeGenORAA::PrintExpr(const PrimExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.dtype());
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenORAA::PrintSSAAssign(const std::string& target, const std::string& src, DataType t) {}

void CodeGenORAA::PrintFinalReturn() {
  decl_stream << "buf = api.malloc(" << current_malloc_size_ << ")\n";
  PrintIndent();
  stream << "api.synchronize()\n";
}
std::string CodeGenORAA::Finish() { return decl_stream.str() + stream.str(); }

void CodeGenORAA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  // ICHECK_EQ(scope, "global");
}

void CodeGenORAA::RegisterHandleType(const VarNode* buf_var, DataType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

void CodeGenORAA::PrintStorageSync(const CallNode* op) {  // NOLINT(*)
}

// Print a reference expression to a buffer.
std::string CodeGenORAA::GetBufferRef(DataType t, const BufferNode* buffer, PrimExpr index) {
  const VarNode* buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  // DataType buffer_element_dtype = buffer->dtype;
  std::string buffer_str = vid;
  std::string index_str = PrintExpr(index);
  os << buffer_str << "[" << index_str << "]";

  return os.str();
}

std::string CodeGenORAA::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "(";
  this->PrintType(target, os);
  os << "(" << value << "))";
  return os.str();
}

// void CodeGenORAA::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
//   std::stringstream value;
//   this->PrintExpr(op->value, value);
//   os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
// }

inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenORAA* p) {  // NOLINT(*)
  std::ostringstream temp;
  temp << std::to_string(op->value);
  p->MarkConst(temp.str());
  os << temp.str();
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenORAA* p) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenORAA* p) {
  if (op->dtype.lanes() == 1) {
    std::string a = p->PrintExpr(op->a);
    std::string b = p->PrintExpr(op->b);
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      os << a;
      os << ", ";
      os << b;
      os << ')';
    } else {
      os << '(';
      os << a;
      os << ' ' << opstr << ' ';
      os << b;
      os << ')';
    }
  } else {
    LOG(FATAL) << "Vector load not implemented";
  }
}

void CodeGenORAA::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenORAA::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenORAA::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "\"" << op->value << "\"";
}

void CodeGenORAA::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
}

void CodeGenORAA::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}

void CodeGenORAA::VisitExpr_(const AddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenORAA::VisitExpr_(const SubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenORAA::VisitExpr_(const MulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenORAA::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenORAA::VisitExpr_(const ModNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinaryExpr(op, "%", os, this);
  } else {
    ICHECK(op->dtype.is_float()) << "Expected floating point or integer dtype in Mod, but got "
                                 << op->dtype;
    if (op->dtype.bits() == 32) {
      PrintBinaryExpr(op, "fmodf", os, this);
    } else if (op->dtype.bits() == 64) {
      PrintBinaryExpr(op, "fmod", os, this);
    } else {
      ICHECK(false)
          << "Non single or double precision floating point in Mod, expected 32 or 64 bits but got "
          << op->dtype.bits() << " bits.";
    }
  }
}
void CodeGenORAA::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenORAA::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenORAA::VisitExpr_(const EQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenORAA::VisitExpr_(const NENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenORAA::VisitExpr_(const LTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenORAA::VisitExpr_(const LENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenORAA::VisitExpr_(const GTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenORAA::VisitExpr_(const GENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenORAA::VisitExpr_(const AndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, " and ", os, this);
}
void CodeGenORAA::VisitExpr_(const OrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, " or ", os, this);
}
void CodeGenORAA::VisitExpr_(const NotNode* op, std::ostream& os) {  // NOLINT(*)
  os << "not";
  PrintExpr(op->a, os);
}

void CodeGenORAA::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::oraa_slice_tensor())) {
    ICHECK_EQ(op->args.size(), 11U);
    // shared_buf = api.slice(global_buf,:,:,:,:)
    // sliced_buf(global) = global_buf[][][][]
    this->PrintExpr(op->args[0], os);
    os << " = ";
    this->PrintExpr(op->args[1], os);
    os << "[";
    for (int i = 0; i < 4; ++i) {
      os << this->PrintExpr(op->args[3 + i]);
      os << ":";
      os << this->PrintExpr(op->args[3 + i]) << "+" << this->PrintExpr(op->args[3 + 4 + i]);
      if (i < 3) {
        os << ",";
      }
    }
    os << "]";
    std::stringstream ss;
    ss << os.rdbuf();
    VLOG(2) << ss.str();
  } else if (op->op.same_as(builtin::call_extern())) {
    auto func_name = this->PrintExpr(op->args[0]);
    if (func_name == "\"write_to_tensor\"") {
      ICHECK_EQ(op->args.size(), 5U);
      // @need structural?
      os << "api.write_to_tensor(";
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ",";
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << ")\n";
    } else if (func_name == "\"read_from_tensor\"") {
      ICHECK_EQ(op->args.size(), 5U);
      // @need structural?
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << " = api.read_from_tensor(";
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ")\n";
    } else if (func_name == "\"relu\"") {
      ICHECK_EQ(op->args.size(), 5U);
      os << "api.relu";
      os << "(";
      // core_id;
      os << "0";
      os << ",";
      // in
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ",";
      // out
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << ")\n";
    } else if (func_name == "\"binary_add\"") {
      os << "api.add";
      os << "(";
      // core_id;
      os << "0";
      os << ", in0=";
      // in
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ", in1=";
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << ", in2=None";
      os << ", in3=None";
      // out
      os << ", out=";
      os << this->PrintExpr(op->args[5]) << "[" << this->PrintExpr(op->args[6]) << "]";
      os << ")\n";
    } else if (func_name == "\"binary_sub\"") {
      os << "api.sub";
      os << "(";
      // core_id;
      os << "0";
      os << ", in0=";
      // in
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ", in1=";
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      // out
      os << ", out=";
      os << this->PrintExpr(op->args[5]) << "[" << this->PrintExpr(op->args[6]) << "]";
      os << ")\n";
    } else if (func_name == "\"add3\"") {
      os << "api.add";
      os << "(";
      // core_id;
      os << "0";
      os << ", in0=";
      // in
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ", in1=";
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << ", in2=";
      os << this->PrintExpr(op->args[5]) << "[" << this->PrintExpr(op->args[6]) << "]";
      os << ", in3=None";
      // out
      os << ", out=";
      os << this->PrintExpr(op->args[7]) << "[" << this->PrintExpr(op->args[8]) << "]";
      os << ")\n";
    } else if (func_name == "\"add4\"") {
      ICHECK_EQ(op->args.size(), 11U);
      os << "api.add";
      os << "(";
      // core_id;
      os << "0";
      os << ", in0=";
      // in
      os << this->PrintExpr(op->args[1]) << "[" << this->PrintExpr(op->args[2]) << "]";
      os << ", in1=";
      os << this->PrintExpr(op->args[3]) << "[" << this->PrintExpr(op->args[4]) << "]";
      os << ", in2=";
      os << this->PrintExpr(op->args[5]) << "[" << this->PrintExpr(op->args[6]) << "]";
      os << ", in3=";
      os << this->PrintExpr(op->args[7]) << "[" << this->PrintExpr(op->args[8]) << "]";
      // out
      os << ", out=";
      os << this->PrintExpr(op->args[9]) << "[" << this->PrintExpr(op->args[10]) << "]";
      os << ")\n";
    } else if (func_name == "\"pixel_shuffle\"") {
      ICHECK_EQ(op->args.size(), 13U);
      // api.pixel_shuffle(in, out)
      os << "api.pixel_shuffle(";
      // core_id
      os << "0";
      os << ", ";
      // input
      os << this->PrintExpr(op->args[1]);
      if (PrintExpr(op->args[2]) != "0") {
        ;
      }
      os << ", ";
      // output
      os << this->PrintExpr(op->args[3]);
      if (PrintExpr(op->args[4]) != "0") {
        ;
      }
      os << ")\n";
    } else if (func_name == "\"pixel_unshuffle\"") {
      ICHECK_EQ(op->args.size(), 13U);
      // api.pixel_shuffle(in, out)
      os << "api.pixel_unshuffle(";
      // core_id
      os << "0";
      os << ", ";
      // input
      os << this->PrintExpr(op->args[1]);
      if (PrintExpr(op->args[2]) != "0") {
        ;
      }
      os << ", ";
      // output
      os << this->PrintExpr(op->args[3]);
      if (PrintExpr(op->args[4]) != "0") {
        ;
      }
      os << ")\n";
    }
  }
}

void CodeGenORAA::VisitStmt_(const AllocateConstNode* op) { LOG(FATAL) << "To be implemented"; }

void CodeGenORAA::VisitStmt_(const DeclBufferNode* op) { this->PrintStmt(op->body); }

void CodeGenORAA::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Unexpected deprecated LoadNode.  Use BufferLoadNode instead.";
}

void CodeGenORAA::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;
  // delcare type.
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
    os << ref;
  }

  // stream << "BufferLoadNotImplemented\n";
}

void CodeGenORAA::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected deprecated StoreNode.  Use BufferStoreNode instead.";
}

void CodeGenORAA::VisitStmt_(const BufferStoreNode* op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->buffer->data;
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(value_dtype, op->buffer.get(), index_expr);
    this->PrintIndent();
    stream << ref << " = " << value << "\n";
  }

  // stream << "BufferStoreNotImplemented\n";
}

void CodeGenORAA::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  std::string value = PrintExpr(op->value);
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

void CodeGenORAA::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "[";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "]";
}

void CodeGenORAA::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  LOG(FATAL) << "Shuffle: not supported ";
}

void CodeGenORAA::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenORAA::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  PrintExpr(op->true_value, os);
  os << "if ";
  PrintExpr(op->condition, os);
  os << "else ";
  PrintExpr(op->false_value, os);
  os << "\n";
}

void CodeGenORAA::VisitStmt_(const LetStmtNode* op) { LOG(FATAL) << "LetStmt: to be implemented"; }

void CodeGenORAA::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  PrintStorageScope(scope, stream);
  // relu_a_shared = api.alloc_tensor(0,16384,dtype="int8")
  malloc_pair_[op] = std::make_pair(current_malloc_size_, constant_size);
  current_malloc_size_ += constant_size;
  stream << vid << " = api.declare_tensor(buf, " << malloc_pair_[op].first << ", "
         << malloc_pair_[op].second << ", TensorFormat.NCHW)\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenORAA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::pragma_import_c) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    ICHECK(value != nullptr);
    decl_stream << value->value;
  }
  this->PrintStmt(op->body);
}

void CodeGenORAA::VisitStmt_(const AssertStmtNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    stream << "ICHECK(" << cond << ") << \"" << str->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenORAA::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for " << vid << " in range(" << extent << "):\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  // PrintIndent();
}

void CodeGenORAA::VisitStmt_(const WhileNode* op) {
  PrintIndent();
  stream << "while " << PrintExpr(op->condition) << " :\n";
  int while_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
}

void CodeGenORAA::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenORAA::VisitStmt_(const SeqStmtNode* op) {
  for (Stmt stmt : op->seq) {
    PrintStmt(stmt);
  }
}

void CodeGenORAA::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->op.same_as(builtin::tvm_storage_sync())) {
      this->PrintStorageSync(call);
      return;
    }
  }
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << vid << "\n";
  }
}

}  // namespace codegen
}  // namespace tvm