"""A set of passes to legalize some of operations for the Open-Research-AI-Architecture"""
import tvm  # type: ignore
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback  # type: ignore
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.backend.contrib.oraa import op as oraa_op
import numpy as np  # type: ignore

class PixelShuffleRewriter(DFPatternCallback):
    """Convert reshape-transpose-reshape related composite functions
    to pixel_shuffle operators.
    """

    def __init__(self, require_type=True, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "pixel_shuffle"}))(wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        input_tensor = post.args[0]
        # print("*" * 10)
        # print(input_tensor)
        return oraa_op.oraa_pixel_shuffle(input_tensor)

class SpaceToDepthRewriter(DFPatternCallback):
    """Convert reshape-transpose-reshape related composite functions
    to pixel_shuffle operators.
    """

    def __init__(self, require_type=True, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "space_to_depth"}))(wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        input_tensor = post.args[0]
        # print("*" * 10)
        # print(input_tensor)
        return oraa_op.oraa_space_to_depth(input_tensor)

class Add2Rewriter(DFPatternCallback):
    """Convert add2 composite function
    to oraa_add2 operator.
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "add2"}))(wildcard(),wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        in0 = post.args[0]
        in1 = post.args[1]
        return oraa_op.oraa_add2(in0,in1)

class Add3Rewriter(DFPatternCallback):
    """Convert add3 related composite functions
    to oraa_add3 operators.
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "add3"}))(wildcard(),wildcard(),wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        in0 = post.args[0]
        in1 = post.args[1]
        in2 = post.args[2]
        # print("*" * 10)
        # print(in0)
        # print(in1)
        # print(in2)
        return oraa_op.oraa_add3(in0,in1,in2)

class Add3GraphRewriter(DFPatternCallback):
    """Convert add3 composite functions
    to concat+conv1x1 composite functions
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "add3"}))(wildcard(),wildcard(),wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        in0 = post.args[0]
        in1 = post.args[1]
        in2 = post.args[2]
        ishape = in2.checked_type.concrete_shape
        dtype=in2.checked_type.dtype
        concat = relay.concatenate([in0,in1,in2], axis=1)
        in_channel = ishape[1]*3
        out_channel = ishape[1]
        # weight  OIHW
        weight_unit_np = np.identity(out_channel)[:,:,None,None]
        # print(weight_unit_np)
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        # print(weight_np)
        weight = relay.const(weight_np.astype(dtype))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return conv

class Add4Rewriter(DFPatternCallback):
    """Convert add4 related composite functions
    to oraa_add4 operators.
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "add4"}))(wildcard(),wildcard(),wildcard(),wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        in0 = post.args[0]
        in1 = post.args[1]
        in2 = post.args[2]
        in3 = post.args[3]
        # print("*" * 10)
        # print(in0)
        # print(in1)
        # print(in2)
        # print(in3)
        return oraa_op.oraa_add4(in0,in1,in2,in3)

class Add4GraphRewriter(DFPatternCallback):
    """Convert add4 composite functions
    to concat+conv1x1 composite functions
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)
        self.pattern = (wildcard().has_attr({"Composite":
                                             "add4"}))(wildcard(),wildcard(),wildcard(),wildcard())

    def callback(self, pre: tvm.relay.Expr, post: tvm.relay.Expr,
                 node_map: tvm.ir.container.Map) -> tvm.relay.Expr:
        in0 = post.args[0]
        in1 = post.args[1]
        in2 = post.args[2]
        in3 = post.args[3]
        ishape = in3.checked_type.concrete_shape
        dtype=in3.checked_type.dtype
        concat = relay.concatenate([in0,in1,in2,in3], axis=1)
        in_channel = ishape[1]*4
        out_channel = ishape[1]
        # weight  OIHW
        weight_unit_np = np.identity(out_channel)[:,:,None,None]
        # print(weight_unit_np)
        weight_np = np.concatenate((weight_unit_np,weight_unit_np,weight_unit_np,weight_unit_np),axis=1)
        # print(weight_np)
        weight = relay.const(weight_np.astype(dtype))
        conv = relay.nn.conv2d(concat, weight, kernel_size=(1, 1))
        return conv

def transform_oraa_function(func: relay.Function) -> relay.Function:
    """This is the method that replace the operations
    with hardware/codegen supported operations by oraa
    """
    rewriters = [
        PixelShuffleRewriter(),
        SpaceToDepthRewriter(),
        # Add2Rewriter(),
        # Add3Rewriter(),
        # Add4Rewriter(),
        Add4GraphRewriter(),
        Add3GraphRewriter(),
        
    ]

    for rewriter in rewriters:
        func = rewrite(rewriter, func)
        # func = run_opt_pass(func, relay.transform.InferType())

    return func
