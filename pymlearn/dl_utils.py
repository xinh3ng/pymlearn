"""
"""
from pdb import set_trace as debug
import resource
from keras.callbacks import Callback
from pydsutils.generic import create_logger

logger = create_logger(__name__)


class TfMemoryUsage(Callback):
    """Keras callback on memory usage
    """
    def on_epoch_end(self, logs={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


    def on_batch_end(self, logs={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
