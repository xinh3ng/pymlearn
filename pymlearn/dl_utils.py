"""
"""
from pdb import set_trace as debug
from keras.callbacks import Callback
from pydsutils.generic import create_logger
from pydsutils.system import get_memory_usage

logger = create_logger(__name__)


class TfMemoryUsage(Callback):
    """Keras callback on memory usage
    """
    def on_train_begin(self, logs={}):
        self.mem_usage = []
    
    def on_epoch_begin(self, epoch, logs={}):
        denom = 1024.0
        usage = get_memory_usage() / denom
        self.mem_usage.append('on_epoch_begin: %.2f Gb' % usage)
        logger.info('on_epoch_begin: memory usage: %.2f Gb' % usage)

    def on_epoch_end(self, epoch, logs={}):
        denom = 1024.0
        usage = get_memory_usage() / denom
        self.mem_usage.append('on_epoch_end: %.2f Gb' % usage)
        logger.info('on_epoch_end: memory usage: %.2f Gb' % usage)

    def on_batch_end(self, batch, logs={}):
        return
        denom =  1024.0
        usage = get_memory_usage() / denom
        self.mem_usage.append('on_batch_end: %.2f Gb' % usage)
        logger.info('on-batch_end: memory usage: %.2f Gb' % usage)

