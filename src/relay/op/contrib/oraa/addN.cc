
/*!
 * \file src/relay/op/contrib/oraa/addN.cc
 * \brief Operator definitions for the Open Research AI Architecture add3/add4 ops.
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
namespace oraa {

// Add3 is a elemwise operator
Array<IndexExpr> ORAAAdd3OutputShape(Array<IndexExpr> in0_shape, Array<IndexExpr> in1_shape,
                                     Array<IndexExpr> in2_shape, String ifm_layout,
                                     String ofm_layout) {
  Array<IndexExpr> output_shape{
      max(in0_shape[0], max(in1_shape[0], in2_shape[0])),
      max(in0_shape[1], max(in1_shape[1], in2_shape[1])),
      max(in0_shape[2], max(in1_shape[2], in2_shape[2])),
      max(in0_shape[3], max(in1_shape[3], in2_shape[3])),
  };

  return output_shape;
}

// Add4 is a elemwise operator
Array<IndexExpr> ORAAAdd4OutputShape(Array<IndexExpr> in0_shape, Array<IndexExpr> in1_shape,
                                     Array<IndexExpr> in2_shape, Array<IndexExpr> in3_shape,
                                     String ifm_layout,
                                     String ofm_layout) {
  Array<IndexExpr> output_shape{
      max(in0_shape[0], max(in1_shape[0], max(in2_shape[0],in3_shape[0]))),
      max(in0_shape[1], max(in1_shape[1], max(in2_shape[1],in3_shape[1]))),
      max(in0_shape[2], max(in1_shape[2], max(in2_shape[2],in3_shape[2]))),
      max(in0_shape[3], max(in1_shape[3], max(in2_shape[3],in3_shape[3]))),
  };

  return output_shape;
}

struct ORAAAdd3Attrs : public tvm::AttrsNode<ORAAAdd3Attrs> {
  String ifm_layout;
  String ofm_layout;
  TVM_DECLARE_ATTRS(ORAAAdd3Attrs, "relay.attrs.ORAAAdd3Attrs") {
    TVM_ATTR_FIELD(ifm_layout)
        .set_default("NCHW")
        .describe("The layout of the Input Feature Map tensor. Can be 'NCHW'.");
    TVM_ATTR_FIELD(ofm_layout)
        .set_default("NCHW")
        .describe("The layout of the Output Feature Map tensor. Can be 'NCHW'.");
  }
};

TVM_REGISTER_NODE_TYPE(ORAAAdd3Attrs);

struct ORAAAdd4Attrs : public tvm::AttrsNode<ORAAAdd4Attrs> {
  String ifm_layout;
  String ofm_layout;
  TVM_DECLARE_ATTRS(ORAAAdd4Attrs, "relay.attrs.ORAAAdd4Attrs") {
    TVM_ATTR_FIELD(ifm_layout)
        .set_default("NCHW")
        .describe("The layout of the Input Feature Map tensor. Can be 'NCHW'.");
    TVM_ATTR_FIELD(ofm_layout)
        .set_default("NCHW")
        .describe("The layout of the Output Feature Map tensor. Can be 'NCHW'.");
  }
};

TVM_REGISTER_NODE_TYPE(ORAAAdd4Attrs);

bool ORAAAdd3Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // CHECK_EQ(types.size(), 4);
  if(types.size()!=4)
    return false;
  const auto* in0 = types[0].as<TensorTypeNode>();
  const auto* in1 = types[1].as<TensorTypeNode>();
  const auto* in2 = types[2].as<TensorTypeNode>();
  if (in0 == nullptr || in1 == nullptr || in2 == nullptr) return false;

  const auto* param = attrs.as<ORAAAdd3Attrs>();
  CHECK(param != nullptr) << "ORAAAdd3Attrs cannot be nullptr.";
  const String operator_name = "oraa_add3";
  CheckDataType(reporter, in0->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in0");
  CheckDataType(reporter, in1->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in1");
  CheckDataType(reporter, in2->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in2");
  Array<IndexExpr> in0_shape = in0->shape;
  Array<IndexExpr> in1_shape = in1->shape;
  Array<IndexExpr> in2_shape = in2->shape;

  // Assign ofm type
  auto ofm_shape =
      ORAAAdd3OutputShape(in0_shape, in1_shape, in2_shape, param->ifm_layout, param->ofm_layout);

  reporter->Assign(types[3], TensorType(ofm_shape, in0->dtype));
  return true;
}

bool ORAAAdd4Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  // CHECK_EQ(types.size(), 5);
  if(types.size()!=5)
    return false;
  const auto* in0 = types[0].as<TensorTypeNode>();
  const auto* in1 = types[1].as<TensorTypeNode>();
  const auto* in2 = types[2].as<TensorTypeNode>();
  const auto* in3 = types[3].as<TensorTypeNode>();
  if (in0 == nullptr || in1 == nullptr || in2 == nullptr || in3 == nullptr) return false;

  const auto* param = attrs.as<ORAAAdd4Attrs>();
  CHECK(param != nullptr) << "ORAAAdd4Attrs cannot be nullptr.";
  const String operator_name = "oraa_add4";
  CheckDataType(reporter, in0->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in0");
  CheckDataType(reporter, in1->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in1");
  CheckDataType(reporter, in2->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in2");
  CheckDataType(reporter, in3->dtype, {DataType::UInt(8), DataType::Int(8)}, operator_name, "in3");
  Array<IndexExpr> in0_shape = in0->shape;
  Array<IndexExpr> in1_shape = in1->shape;
  Array<IndexExpr> in2_shape = in2->shape;
  Array<IndexExpr> in3_shape = in3->shape;

  // Assign ofm type
  auto ofm_shape =
      ORAAAdd4OutputShape(in0_shape, in1_shape, in2_shape, in3_shape, param->ifm_layout, param->ofm_layout);

  reporter->Assign(types[4], TensorType(ofm_shape, in0->dtype));
  return true;
}

Expr MakeAdd3(Expr in0, Expr in1, Expr in2, String ifm_layout, String ofm_layout) {
  auto attrs = make_object<ORAAAdd3Attrs>();
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.oraa.add3");
  return Call(op, {in0, in1, in2}, Attrs(attrs), {});
}

Expr MakeAdd4(Expr in0, Expr in1, Expr in2, Expr in3, String ifm_layout, String ofm_layout) {
  auto attrs = make_object<ORAAAdd4Attrs>();
  attrs->ifm_layout = std::move(ifm_layout);
  attrs->ofm_layout = std::move(ofm_layout);
  static const Op& op = Op::Get("contrib.oraa.add4");
  return Call(op, {in0, in1, in2, in3}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.oraa_add3").set_body_typed(MakeAdd3);
TVM_REGISTER_GLOBAL("relay.op._make.oraa_add4").set_body_typed(MakeAdd4);

RELAY_REGISTER_OP("contrib.oraa.add3")
    .describe(R"code(Open Research AI Architecture add3 operator.

This Relay operator corresponds to the hardware-implemented add3 operator
found on Open Research AI Architecture. It accepts NCHW
format for the input data (Input Feature Map, or IFM).

- **in0,in1,in2**: NCHW - (1, ifm_channels, ifm_height, ifm_width)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ORAAAdd3Attrs>()
    .set_num_inputs(3)
    .add_argument("in0", "Tensor", "The Input tensor 0.")
    .add_argument("in1", "Tensor", "The Input tensor 1.")
    .add_argument("in2", "Tensor", "The Input tensor 2.")
    .set_support_level(11)
    .add_type_rel("ORAA", ORAAAdd3Rel);

RELAY_REGISTER_OP("contrib.oraa.add4")
    .describe(R"code(Open Research AI Architecture add4 operator.

This Relay operator corresponds to the hardware-implemented add4 operator
found on Open Research AI Architecture. It accepts NCHW
format for the input data (Input Feature Map, or IFM).

- **in0,in1,in2,in3**: NCHW - (1, ifm_channels, ifm_height, ifm_width)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ORAAAdd4Attrs>()
    .set_num_inputs(4)
    .add_argument("in0", "Tensor", "The Input tensor 0.")
    .add_argument("in1", "Tensor", "The Input tensor 1.")
    .add_argument("in2", "Tensor", "The Input tensor 2.")
    .add_argument("in3", "Tensor", "The Input tensor 3.")
    .set_support_level(11)
    .add_type_rel("ORAA", ORAAAdd4Rel);

}  // namespace oraa
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
