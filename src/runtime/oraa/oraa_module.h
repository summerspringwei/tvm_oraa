
#ifndef TVM_RUNTIME_ORAA_ORAA_MODULE_H_
#define TVM_RUNTIME_ORAA_ORAA_MODULE_H_

#include <tvm/runtime/module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief create a oraa module from data.
 *
 * \param data The module data, can be ptx, cubin
 * \param fmt The format of the data, can be "ptx", "cubin"
 * \param fmap The map function information map of each function.
 * \param oraa_source Optional, cuda source file
 */
Module ORAAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source);

}  // namespace runtime
}  // namespace tvm

#endif