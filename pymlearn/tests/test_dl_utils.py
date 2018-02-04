# -*- coding: utf-8 -*-
"""Unit tests on DL util functions
"""
import unittest
from dl_utils import TfMemoryUsage
        
class TfMemoryUsageTest(unittest.TestCase):
    
    def test_usage_count(self):
        """
        """
        usage = TfMemoryUsage()
        usage.on_train_begin()

        self.assertTrue(usage.mem_usage == [])
        return


if __name__ == '__main__':
    unittest.main()
