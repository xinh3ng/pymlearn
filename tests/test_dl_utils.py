# -*- coding: utf-8 -*-
"""Unit tests on DL util functions
"""
import unittest
from pymlearn.dl_utils import TfMemoryUsage, TfMetrics


class TfMemoryUsageTest(unittest.TestCase):
    
    def test_usage_count(self):
        """
        """
        usage = TfMemoryUsage()
        usage.on_train_begin()

        self.assertTrue(usage.mem_usage == [])
        return


class TfMetricsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_on_begin(self):
        """
        """
        m = TfMetrics()
        m.on_train_begin()
        self.assertTrue(m.f1_scores == [])
        self.assertTrue(m.precisions == [])

    def test_metrics(self):
        m = TfMetrics()
        m.on_train_begin()


if __name__ == '__main__':
    unittest.main()
