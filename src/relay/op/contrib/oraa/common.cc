
#include "common.h"

#include <sstream>

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace oraa {

void CheckDataType(const TypeReporter& reporter, const DataType& data_type,
                   const std::initializer_list<DataType>& allowed_data_types,
                   const String& operator_name, const String& tensor_name,
                   const String& operator_type) {
  for (const auto& i : allowed_data_types) {
    if (data_type == i) {
      return;
    }
  }

  std::ostringstream message;
  message << "Invalid operator: expected " << operator_name << " ";
  if (operator_type != "") {
    message << operator_type << " ";
  }
  message << "to have type in {";
  for (auto it = allowed_data_types.begin(); it != allowed_data_types.end(); ++it) {
    message << *it;
    if (std::next(it) != allowed_data_types.end()) {
      message << ", ";
    }
  }
  message << "}";
  message << " for " << tensor_name << " but was " << data_type << ".";

  reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan()) << message.str());
}

}  // namespace oraa
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm