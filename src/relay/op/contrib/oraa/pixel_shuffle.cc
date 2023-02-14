
/*!
 * \file src/relay/op/contrib/oraa/pixel_shuffle.cc
 * \brief Operator definitions for the Open Research AI Architecture pixel_shuffle ops.
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

Array<IndexExpr> ORAAPixelShuffleOutputShape(Array<IndexExpr> ifm_shape, Array<IndexExpr> upscale_factor, String ifm_layout, String ofm_layout){
  Array<IndexExpr> output_shape{ifm_shape[0], 
    indexdiv(indexdiv(ifm_shape[1], upscale_factor[0]), upscale_factor[1]), 
    (ifm_shape[2] * upscale_factor[0]),
    ifm_shape[3] * upscale_factor[1]};

  return output_shape;
}

struct ORAAPixelShuffleAttrs: public tvm::AttrsNode<ORAAPixelShuffleAttrs> {
  Array<IndexExpr> upscale_factor;
  String ifm_layout;
  String ofm_layout;
  TVM_DECLARE_ATTRS(ORAAPixelShuffleAttrs, "relay.attrs.ORAAPixelShuffleAttrs"){
    TVM_ATTR_FIELD(upscale_factor).describe("The height and width shuffle factor for the input feature map tensor.");
    TVM_ATTR_FIELD(ifm_layout)
        .set_default("NCHW")
        .describe("The layout of the Input Feature Map tensor. Can be 'NCHW'.");
    TVM_ATTR_FIELD(ofm_layout)
        .set_default("NCHW")
        .describe("The layout of the Output Feature Map tensor. Can be 'NHWC'.");
  }
};

TVM_REGISTER_NODE_TYPE(ORAAPixelShuffleAttrs);

bool ORAAPixelShuffleRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter){
  CHECK_EQ(types.size(), 2);
  const auto* ifm = types[0].as<TensorTypeNode>();
  if(ifm == nullptr) return false;

  const auto* param = attrs.as<ORAAPixelShuffleAttrs>();
  CHECK(param != nullptr) << "ORAAPixelShuffleAttrs cannot be nullptr.";
  const String operator_name = "oraa_pixel_shuffle";
  CheckDataType(reporter, ifm->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "ifm");
  Array<IndexExpr> ifm_shape = ifm->shape;

  // Assign ofm type
  auto ofm_shape = ORAAPixelShuffleOutputShape(ifm_shape, param->upscale_factor, param->ifm_layout, param->ofm_layout);

  reporter->Assign(types[1], TensorType(ofm_shape, ifm->dtype));
  return true;
}

Expr MakePixelShuffle(Expr input, Array<IndexExpr> upscale_factor, String ifm_layout, String ofm_layout){
  auto attrs = make_object<ORAAPixelShuffleAttrs>();
  attrs->upscale_factor = upscale_factor;
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.oraa.pixel_shuffle");
  return Call(op, {input,}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.oraa_pixel_shuffle").set_body_typed(MakePixelShuffle);

RELAY_REGISTER_OP("contrib.oraa.pixel_shuffle")
    .describe(R"code(Open Research AI Architecture pixel shuffle operator.

This Relay operator corresponds to the hardware-implemented pixel shuffle operator
found on Open Research AI Architecture. It accepts NCHW
format for the input data (Input Feature Map, or IFM).

Reference: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html

- **ifm**: NCHW - (1, ifm_channels, ifm_height, ifm_width)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ORAAPixelShuffleAttrs>()
    .set_num_inputs(1)
    .add_argument("ifm", "Tensor", "The Input Feature Map tensor (IFM).")
    .set_support_level(11)
    .add_type_rel("ORAA", ORAAPixelShuffleRel);


}  // namespace oraa
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

