# -*- coding: utf-8 -*-
"""
Unit tests on cross validate, time series

"""
from pdb import set_trace as debug
import unittest
import numpy as np
import pandas as pd
from pymlearn.cv_ts import TimeSeriesSplitter


class TimeSeriesSplitterTest(unittest.TestCase):
    
    def test_split(self):
        """
        """
        data = pd.DataFrame.from_dict({
            'x': [i for i in range(0, 200)],
            'y': [1 for _ in range(0, 200)]
        })

        # TODO: don't know how to check this error yet
        splitter = TimeSeriesSplitter(train_size=199, n_ahead=2)
        #for _ in splitter.split(data):
        #    self.assertRaises(AssertionError, splitter.split(data))

        splitter = TimeSeriesSplitter(train_size=198, n_ahead=2)
        cnt = 0
        for train_data, test_data in splitter.split(data):
            cnt += 1
            self.assertEqual(test_data['x'].values.tolist(),
                             [- 1 + cnt + splitter.train_size, cnt + splitter.train_size]
                             )

        self.assertEqual(cnt, 1)  # Should have only 1 sample


if __name__ == '__main__':
    unittest.main()
