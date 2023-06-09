import time
import logging

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

