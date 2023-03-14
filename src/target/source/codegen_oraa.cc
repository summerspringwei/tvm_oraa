#include "codegen_oraa.h"

#include <tvm/arith/analyzer.h>

#include "../../printer/text_printer.h"
#include "../../printer/meta_data.h"

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenORAA::Init() {}

void CodeGenORAA::InitFuncState(const PrimFunc& f) {
  CodeGenSourceBase::ClearFuncState();
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
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

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
  const auto* op = f.operator->();
  uint32_t offset = 0;
  for (Var v : op->params) {
    std::string vid = AllocVarID(v.get());
    auto it = op->buffer_map.find(v);
    if(it != op->buffer_map.end()){
      stream << vid << " = " << "api.declare_tensor(mem, " << offset;
      stream << printer.Print((*it).second).str();
      stream << ", TensorFormat.NCHW)\n";
    }
  }
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
}


}  // namespace codegen
}  // namespace tvm