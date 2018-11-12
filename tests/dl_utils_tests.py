# -*- coding: utf-8 -*-
"""Unit tests on deep learning utility functions
"""
return

from pymlearn.dl_utils import TfMemoryUsage, TfMetrics


def test_usage_count():
    """"""
    usage = TfMemoryUsage()
    usage.on_train_begin()
    assert (usage.mem_usage == [])


def test_tfmetrics():
    m = TfMetrics()
    m.on_train_begin()


def test_on_begin():
    """"""
    m = TfMetrics()
    m.on_train_begin()
    assert m.f1_scores == []
    assert m.precisions == []
