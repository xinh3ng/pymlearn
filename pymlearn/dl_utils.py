"""
"""
from pdb import set_trace as debug
from keras.callbacks import Callback
from pydsutils.system import get_memory_usage
from pydsutils.generic import create_logger

logger = create_logger(__name__)


class TfMemoryUsage(Callback):
    """Keras callback on memory usage
    """
    def __init__(self, show_batch_begin=False, show_batch_end=False):
        super(TfMemoryUsage, self).__init__()
        self.show_batch_begin = show_batch_begin
        self.show_batch_end = show_batch_end
    
    def _get_mem_usage(self, pattern):
        denom = 1024.0
        usage = get_memory_usage() / denom
        self.mem_usage.append('%s: %.2f Gb' % (pattern, usage))
        logger.info('%s: memory usage: %.2f Gb' % (pattern, usage))

    def on_train_begin(self, logs={}):
        self.mem_usage = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self._get_mem_usage(pattern='on_epoch_begin')

    def on_epoch_end(self, epoch, logs={}):
        self._get_mem_usage(pattern='on_epoch_end')

    def on_batch_begin(self, batch, logs={}):
        if self.show_batch_begin:
            self._get_mem_usage(pattern='on_batch_begin')
    
    def on_batch_end(self, batch, logs={}):
        if self.show_batch_end:
            self._get_mem_usage(pattern='on_batch_end')
