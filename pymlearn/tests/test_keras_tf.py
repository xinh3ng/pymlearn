# -*- coding: utf-8 -*-
"""Unit tests of keras TF functions
"""
import unittest
import numpy as np

        
class ParallelImage2ArrayTest(unittest.TestCase):
    
    def test_shape_is_preserved(self):
        """Test if the numpy array shape is perserved during parallel processing
        
        This usage is similar to my usage in image to array conversion
        """
        input_shape = (20, 10)
        self.assertEqual(input_shape, 0)
        return


if __name__ == '__main__':
    unittest.main()
