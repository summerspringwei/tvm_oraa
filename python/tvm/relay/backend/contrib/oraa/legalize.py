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
        print("*" * 10)
        print(input_tensor)
        return oraa_op.oraa_pixel_shuffle(input_tensor)


def transform_oraa_function(func: relay.Function) -> relay.Function:
    """This is the method that replace the operations
    with hardware/codegen supported operations by oraa
    """
    rewriters = [
        PixelShuffleRewriter(),
    ]

    for rewriter in rewriters:
        func = rewrite(rewriter, func)

    return func
