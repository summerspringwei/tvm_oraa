"""A set of passes to legalize some of operations for the Open-Research-AI-Architecture"""
import tvm  # type: ignore
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback  # type: ignore
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.backend.contrib.oraa import op as oraa_op


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

class Add3Rewriter(DFPatternCallback):
    """Convert add3 related composite functions
    to oraa_add3 operators.
    """
    def __init__(self, require_type=True, rewrite_once=False):
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

class Add4Rewriter(DFPatternCallback):
    """Convert add4 related composite functions
    to oraa_add4 operators.
    """
    def __init__(self, require_type=True, rewrite_once=False):
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

def transform_oraa_function(func: relay.Function) -> relay.Function:
    """This is the method that replace the operations
    with hardware/codegen supported operations by oraa
    """
    rewriters = [
        PixelShuffleRewriter(),
        Add3Rewriter(),
        Add4Rewriter(),
        
    ]

    for rewriter in rewriters:
        func = rewrite(rewriter, func)

    return func
