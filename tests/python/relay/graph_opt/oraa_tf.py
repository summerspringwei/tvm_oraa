
from typing import Union, List
import numpy as np


class TensorFormat:
    NCHW=0
    OCICHW=1


class TensorBuffer:
    def __init__(self, shape:list, dtype: str, scope: str):
        self.shape = shape
        self.dtype = dtype
        self.scope = scope


def talloc(shape: list, layout: TensorFormat, offset: int, align: int):
    pass

def free(tensor: TensorBuffer):
    pass


def write_to_tensor(dst: TensorBuffer, src: np.ndarray, dtype: str):
    pass


def read_from_tensor(src: TensorBuffer):
    pass


def write_barrier(core: int, tensor: TensorBuffer):
    pass


def synchronize(dry_run=False):
    pass


def add(core: int, dst: TensorBuffer, src_0: TensorBuffer, src_1: TensorBuffer):
    pass


def reshape(core: int, dst: TensorBuffer, src: TensorBuffer, new_shape: list):
    pass


def conv(core: int, dst: TensorBuffer, activation: TensorBuffer, filter: TensorBuffer, stride=[1,1,1,1], pad=[1,1,1,1]):
    pass


def softmax(core: int, dst: TensorBuffer, src: TensorBuffer, axis=-1):
    pass


def transpose(core: int, dst: TensorBuffer, src: TensorBuffer, permute: list):
    pass


def pixel_shuffle(core: int, dst: TensorBuffer, src: TensorBuffer, upscale_factor: int):
    pass


def concat(core: int, dst, src: List[TensorBuffer], axis=1):
    pass