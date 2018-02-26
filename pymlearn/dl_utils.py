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


class TfMetrics(Callback):
    """Keras callback on metrics

    https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    """
    def __init__(self):
        super(TfMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.f1_scores = []
        self.recalls = []
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        y_true = self.model.validation_data[1]
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        self.f1_scores.append(f1)
        self.recalls.append(recall)
        self.precisions.append(precision)

        logger.info('on_epoch_end: f1: %.4f, precision: %.4f, recall %.4f' %\
                    (f1, precision, recall))
        return
