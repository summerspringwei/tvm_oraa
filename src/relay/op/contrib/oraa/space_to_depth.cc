
/*!
 * \file src/relay/op/contrib/oraa/space_to_depth.cc
 * \brief Operator definitions for the Open Research AI Architecture space_to_depth ops.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/data_layout.h>

#include "common.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace oraa{

Array<IndexExpr> ORAASpaceToDepthOutputShape(Array<IndexExpr> ifm_shape, Array<IndexExpr> downscale_factor, String ifm_layout, String ofm_layout){
    // NCHW -> N 4C H/2 W/2
  Array<IndexExpr> output_shape{
    ifm_shape[0], 
    (ifm_shape[1] * downscale_factor[0] * downscale_factor[1]),
    indexdiv(ifm_shape[2], downscale_factor[0]),
    indexdiv(ifm_shape[3], downscale_factor[1]),
    };

  return output_shape;
}

struct ORAASpaceToDepthAttrs: public tvm::AttrsNode<ORAASpaceToDepthAttrs> {
  Array<IndexExpr> downscale_factor;
  String ifm_layout;
  String ofm_layout;
  TVM_DECLARE_ATTRS(ORAASpaceToDepthAttrs, "relay.attrs.ORAASpaceToDepthAttrs"){
    TVM_ATTR_FIELD(downscale_factor).describe("The height and width shuffle factor for the input feature map tensor.");
    TVM_ATTR_FIELD(ifm_layout)
        .set_default("NCHW")
        .describe("The layout of the Input Feature Map tensor. Can be 'NCHW'.");
    TVM_ATTR_FIELD(ofm_layout)
        .set_default("NCHW")
        .describe("The layout of the Output Feature Map tensor. Can be 'NCHW'.");
  }
};

TVM_REGISTER_NODE_TYPE(ORAASpaceToDepthAttrs);

bool ORAASpaceToDepthRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter){
  if(2!=types.size())
    return false;
  CHECK_EQ(types.size(), 2);
  const auto* ifm = types[0].as<TensorTypeNode>();
  if(ifm == nullptr) return false;

  const auto* param = attrs.as<ORAASpaceToDepthAttrs>();
  CHECK(param != nullptr) << "ORAASpaceToDepthAttrs cannot be nullptr.";
  const String operator_name = "oraa_space_to_depth";
  CheckDataType(reporter, ifm->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "ifm");
  Array<IndexExpr> ifm_shape = ifm->shape;

  // Assign ofm type
  auto ofm_shape = ORAASpaceToDepthOutputShape(ifm_shape, param->downscale_factor, param->ifm_layout, param->ofm_layout);

  reporter->Assign(types[1], TensorType(ofm_shape, ifm->dtype));
  return true;
}

Expr MakeSpaceToDepth(Expr input, Array<IndexExpr> downscale_factor, String ifm_layout, String ofm_layout){
  auto attrs = make_object<ORAASpaceToDepthAttrs>();
  attrs->downscale_factor = downscale_factor;
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.oraa.space_to_depth");
  return Call(op, {input,}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.oraa_space_to_depth").set_body_typed(MakeSpaceToDepth);

RELAY_REGISTER_OP("contrib.oraa.space_to_depth")
    .describe(R"code(Open Research AI Architecture space_to_depth operator.

This Relay operator corresponds to the hardware-implemented space_to_depth operator
found on Open Research AI Architecture. It accepts NCHW
format for the input data (Input Feature Map, or IFM).

- **ifm**: NCHW - (1, ifm_channels, ifm_height, ifm_width)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ORAASpaceToDepthAttrs>()
    .set_num_inputs(1)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .set_support_level(11)
    .add_type_rel("ORAA", ORAASpaceToDepthRel);


}  // namespace oraa
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

