# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

import sklearn.metrics as sk_metrics

def cal_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    return sk_metrics.mean_squared_error(y_true, y_pred, sample_weight, multioutput)
