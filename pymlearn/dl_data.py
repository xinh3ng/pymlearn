"""
"""
from pdb import set_trace as debug
from pydsutils.generic import create_logger

logger = create_logger(__name__)


class BaseModelDataValidator(object):
    """Validator of Model Data

    """
    def __init__(self, num_classes, num_rows, num_columns, num_channels):
        self.num_classes = num_classes
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_channels = num_channels

    def validate_X(self, X):
        raise NotImplementedError

    def validate_y(self, y):
        is_valid = True
        reasons = []
        if y.shape[1] != self.num_classes:
            is_valid = False
            reasons.append('input y.shape[1] does not match requirements')

        return {'is_valid': is_valid,
                'reasons': reasons}


class TfModelDataValidator(BaseModelDataValidator):
    def __init__(self, num_classes, num_rows, num_columns, num_channels):
        """

        Args:
            include_top: whether to include the 3 fully-connected layers at the top of the network.
            weights:
        """
        super(TfModelDataValidator, self).__init__(num_classes, num_rows, num_columns, num_channels)

    def validate_X(self, X):
        """
        Args:
            X:feature matrix that will feed into model training
        """
        is_valid = True
        reasons = []
        if X.shape[1:] != (self.num_rows, self.num_columns, self.num_channels):
            is_valid = False
            reasons.append('input X[1:] shape does not match requirements')

        return {'is_valid': is_valid,
                'reasons': reasons}


class TorchModelDataValidator(BaseModelDataValidator):
    def __init__(self, num_classes, num_rows, num_columns, num_channels):
        """

        Args:
            include_top: whether to include the 3 fully-connected layers at the top of the network.
            weights:
        """
        super(TorchModelDataValidator, self).__init__(num_classes, num_rows, num_columns, num_channels)

    def validate_X(self, X):
        """
        Args:
            X:feature matrix that will feed into model training
        """
        is_valid = True
        reasons = []
        if X.shape[1:] != (self.num_columns, self.num_channels, self.num_rows):
            is_valid = False
            reasons.append('input X[1:] shape does not match requirements')

        if (X.max() != 1) or (X.min() != 0):
            is_valid = False
            reasons.append('X is not scaled to [0, 1]')
        return {'is_valid': is_valid,
                'reasons': reasons}

