
/*!
 * \file src/relay/op/contrib/oraa/common.h
 * \brief Functions for all Open Research AI Architecture operators to use.
*/

#ifndef TVM_REALY_OP_CONTRIB_ORAA_COMMON_H
#define TVM_REALY_OP_CONTRIB_ORAA_COMMON_H

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace oraa {

/*! \brief Check the data type for a given input matches one given in allowed_data_types. Raise a
 * type inference error if not.
 * \param reporter The infer type reporter.
 * \param data_type The data type to check.
 * \param allowed_data_types An initializer list of allowed data types.
 * \param operator_name The name of the operator to report.
 * \param tensor_name The name of the tensor to report e.g. "ifm", "ofm".
 * \param operator_type The type of the operator to report e.g. "ADD" for binary_elementwise.
 */
void CheckDataType(const TypeReporter& reporter, const DataType& data_type,
                   const std::initializer_list<DataType>& allowed_data_types,
                   const String& operator_name, const String& tensor_name,
                   const String& operator_type = "");



}  // namespace oraa
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

#endif