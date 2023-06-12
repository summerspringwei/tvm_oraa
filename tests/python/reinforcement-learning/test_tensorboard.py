import time
import logging
import torch
import numpy as np
print(torch.__version__)
print(torch.__file__)

a = torch.from_numpy(np.array([1,2,3,4]))
print(a.dim())
print(a[None, :])
b = torch.from_numpy(np.array([1]))
print(b.dim())
print(b[None, :])
c = torch.tensor(1)
print(c.dim())
c = torch.reshape(c, [1])
print(c[None, :])
exit(0)
import tensorflow as tf


def test_tensorboard():
    writer = tf.summary.create_file_writer("/tmp/board")
    for i in range(100):
        with writer.as_default():
            tf.summary.scalar('learning rate', i, step=i)
        time.sleep(1)
        writer.flush()


def test_logger():
    logger = logging.getLogger("TTTT")
    logger.warn("aaaaa")

