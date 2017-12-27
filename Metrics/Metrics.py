# -*- coding:utf-8 -*-
"""
    Description: 根据预测结果计算模型效果指标
    Author: shelldream
    Date: 2017.11.23
"""
import sys
reload(sys).setdefaultencoding("utf-8")
sys.path.append("..")

from utils.common import colors
from ranking_metrics import *
from classification_metrics import *
from regression_metrics import *

class Metrics:
    def __init__(self, y_real, y_pred, grp_len_list):
        """
            初始化
            Args:
                y_real: list or numpy array, 真实的y
                y_pred: list or numpy array, 预测的y
                grp_len_list: 
            Returns:
                None
        """
        if len(y_pred) != len(y_real):
            raise ValueError(colors.RED + "The length of y_real and y_pred are not same!! " + colors.ENDC)
        
        self.y_real = y_real
        self.y_pred = y_pred

        self.grp_len_list = grp_len_list
        self.instance_cnt = len(y_real)
        self.grp_cnt = len(grp_len_list)

        self.y_real_grp_list = None
        self.y_pred_grp_list = None
        self.split_grp_data()

    def split_grp_data(self):
        """
            按照group length 信息将实际y和预测y进行分组
        """
        self.y_real_grp_list = []
        self.y_pred_grp_list = []
        
        begin = 0
        for grp_len in self.grp_len_list:
            sub_real_list = self.y_real[begin: begin+grp_len]
            sub_pred_list = self.y_pred[begin: begin+grp_len]
            begin += grp_len
            self.y_real_grp_list.append(sub_real_list)
            self.y_pred_grp_list.append(sub_pred_list)
        
    def cal_grp_avg_classification_metric(self):
        pair_list = zip(self.y_real_grp_list, self.y_pred_grp_list) 
        auc_sum = 0.0
        auc_pr_sum = 0.0
        
        grp_cnt = self.grp_cnt

        for pair in pair_list:
            if len(set(pair[0])) == 1: #对于一个group中只有一个label的数据不计算该group的指标 
                grp_cnt -= 1
                continue
            sub_auc = cal_auc_v2(pair[0], pair[1])
            auc_sum += sub_auc
            auc_pr_sum += cal_auc_pr(pair[0], pair[1])
        
        print colors.GREEN + "average auc: %f"%(auc_sum/grp_cnt) + colors.ENDC
        print colors.GREEN + "average auc_pr: %f"%(auc_pr_sum/grp_cnt) + colors.ENDC

    def cal_grp_avg_ranking_metric(self):
        pair_list = zip(self.y_real_grp_list, self.y_pred_grp_list) 
        ndcg_at5_sum = 0.0
        ndcg_at10_sum = 0.0
        err_at5_sum = 0.0
        err_at10_sum = 0.0

        for pair in pair_list:
            ndcg_at5_sum += cal_ndcg(pair[0], pair[1], 5)
            ndcg_at10_sum += cal_ndcg(pair[0], pair[1], 10)
            err_at5_sum += cal_err(pair[0], pair[1], 5)
            err_at10_sum += cal_err(pair[0], pair[1], 10)
        
        print colors.GREEN + "average ndcg@5: %f"%(ndcg_at5_sum/self.grp_cnt) + colors.ENDC
        print colors.GREEN + "average ndcg@10: %f"%(ndcg_at10_sum/self.grp_cnt) + colors.ENDC
        print colors.GREEN + "average err@5: %f"%(err_at5_sum/self.grp_cnt) + colors.ENDC
        print colors.GREEN + "average err@10: %f"%(err_at10_sum/self.grp_cnt) + colors.ENDC
    
    def cal_grp_avg_regression_metric(self):
        pair_list = zip(self.y_real_grp_list, self.y_pred_grp_list)
        mse_sum = 0.0

        for pair in pair_list:
            mse_sum += cal_mean_squared_error(pair[0], pair[1])

        print colors.GREEN + "average MSE: %f"%(mse_sum/self.grp_cnt) + colors.ENDC

