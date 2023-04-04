/*!
 *  Build ORAA modules from source.
 *  Only generate source code for ORAA.
 *
 * \file build_oraa.cc
 */

#include "../build_common.h"
#include "../source/codegen_oraa.h"
#include "../../runtime/oraa/oraa_module.h"

namespace tvm {
namespace codegen {


runtime::Module BuildORAA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenORAA cg;
  cg.Init();
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenCUDA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();
  VLOG(0) << code;
  return ORAAModuleCreate("", "", ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.oraa").set_body_typed(BuildORAA);

}  // namespace codegen
}  // namespace tvm
