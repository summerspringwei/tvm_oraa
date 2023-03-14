
#include "oraa_module.h"

#include <string>

#include "metadata.h"

namespace tvm {
namespace runtime {
class ORAAModuleNode : public runtime::ModuleNode {
 public:
  explicit ORAAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
}

Module ORAAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source){
  auto n = make_object<ORAAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}


}  // namespace runtime
}  // namespace tvm