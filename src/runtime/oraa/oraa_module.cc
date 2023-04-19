#include "oraa_module.h"
#include <tvm/runtime/registry.h>
#include <string>
#include "../pack_args.h"

namespace tvm {
namespace runtime {
class ORAAModuleNode : public runtime::ModuleNode {
 public:
  explicit ORAAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    // std::fill(module_.begin(), module_.end(), nullptr);
  }
  const char* type_key() const final { return "oraa"; }
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);
  std::string GetSource(const std::string& format) final {
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    }
    return "oraa_module.cc: Source code empty!";
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  // which contains several/one function(s)
  std::string cuda_source_;
};

class ORAAWarppedFunc {
 public:
  // initialize the ORAA function.
  void Init(ORAAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    using tvm::runtime::Registry;
    if (const auto* f = Registry::Get("tvm_callback_oraa_compile")) {
      (*f)(m_->GetSource("py"));
    }
  }

 private:
  // internal module
  ORAAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
};

PackedFunc ORAAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  ORAAWarppedFunc f;
  f.Init(this,sptr_to_self,name);
  return PackFuncVoidAddr(f, std::vector<DLDataType>());
  // return PackedFunc();
}


Module ORAAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source) {
  auto n = make_object<ORAAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm