# -*- coding:utf-8 -*-
"""
    Description:
        利用Xgboost进行分类、回归、排序等机器学习任务
        目前分类只支持二分类；排序只支持pairwise的训练方式
    Author: shelldream
    Date: 2017-11-23
"""
import sys
reload(sys).setdefaultencoding("utf-8")
sys.path.append("..")

import os
import datetime
import copy
import xgboost as xgb
import numpy as np
import pandas as pd

from utils.common import colors
from Metrics.Metrics import Metrics

class XgbModel:
    def __init__(self, params, task_type):
        """
            初始化
            Args:
                params: dict, 
                task_type: str,
        """
        
        self.params = params
        if task_type not in ["classification", "regression", "ranking"]:
            raise ValueError(colors.RED + "Invalid task type. The valid options: classification, regression, ranking" + colors.ENDC)

        self.task_type = task_type
        self.booster = None
        self.default_text_model_saveto = "../model/xgb_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".txt"
        self.default_bin_model_saveto = "../model/xgb_" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + ".bin"
         
    def save_text_model(self, model_saveto=None):
        """Save model in text format"""
        if self.booster is None:
            raise ValueError(colors.RED + "Your XgbModel is null!" + colors.ENDC)
        
        if model_saveto is None:
            print colors.BLUE + "Save your model in default file path!" + colors.ENDC
            if not os.path.exists("../model"):
                os.popen("mkdir -p ../model")
        
        model_saveto = model_saveto if model_saveto is not None else self.default_text_model_saveto

        try:
            self.booster.dump_model(model_saveto)
            print colors.GREEN + "%s has been saved successfully in raw text format!!"%model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%model_saveto + colors.ENDC
    
    def save_model(self, model_saveto=None):
        """save model file in binary format"""
        if self.booster is None:
            raise ValueError(colors.RED + "Your XgbModel is null!" + colors.ENDC)
        
        if model_saveto is None:
            print colors.BLUE + "Save your model in default file path!" + colors.ENDC
            if not os.path.exists("../model"):
                os.popen("mkdir -p ../model")
        
        model_saveto = model_saveto if model_saveto is not None else self.default_bin_model_saveto
        
        try:
            self.booster.save_model(model_saveto)
            print colors.GREEN + "%s has been saved successfully in binary format!!"%model_saveto + colors.ENDC
        except:
            print colors.RED + "Warning: %s has not been saved!! Please ensure that the output directory of your model exists. "%model_saveto + colors.ENDC
    
    def load_model(self, model_load_from):    
        """
            load model file in binary format
            Args:
                model_load_from: str, 模型载入路径
            Returns:
                None
        """
        try:
            self.booster.load_model(model_load_from)
            print colors.GREEN + "%s has been loaded successfully in binary format!!"%model_load_from + colors.ENDC     
        except:
            raise ValueError(colors.RED+"Model %s faied to be loaded!!"%model_load_from + colors.ENDC)
    
    def cal_feature_importance(self, importance_type="gain"):
        """
            返回按照特征重要性的分析结果
            Args:
                importance_type: str, 特征重要性类型, 目前只支持 "weight"、"gain"、"cover"
            Rets:
                sorted_fscores:
                sorted_scores:
        """
        if self.booster is None:
            raise ValueError(colors.RED + "Your XgbModel is empty!!" + colors.ENDC)
        self.fscores = self.booster.get_fscore()
        sorted_fscores = sorted(self.fscores.items(), key=lambda x:x[1], reverse=True) 
        self.scores = self.booster.get_score(importance_type=importance_type)
        sorted_scores = sorted(self.scores.items(), key=lambda x:x[1], reverse=True)
        
        print colors.BLUE + "-"*30 + " The feature importance of each feature " + "-"*30 + colors.ENDC
        for (feature, value) in sorted_fscores:
            print colors.BLUE + "\t\t\t%s\t%f"%(feature, value) + colors.ENDC
        
        print colors.BLUE + "\n\n" + "-"*30 + " The feature importance score (%s) "%importance_type + "-"*30 + colors.ENDC
        for (feature, value) in sorted_scores:
            print colors.BLUE + "\t\t\t%s\t%f"%(feature, value) + colors.ENDC
        
        return sorted_fscores, sorted_scores
    
    def train(self, x_data, y_data, grp_len_list=None, importance_type="gain"):
        """
            训练接口
            对于ranking 任务，采用的是pairwise的训练方法,所以grp_len_list是必须的
            Args:
                x_data: pandas dataframe, 特征数据
                y_data: numpy array, 训练目标数据
                grp_len_list: list, 对应pairwise ranking训练任务的数据集每个group的长度
                importance_type: str, 
            Returns:
                average_metric: float
        """
        if self.task_type == "classification":
            self.booster = xgb.XGBClassifier(**self.params).fit(x_data, y_data).booster()
        elif self.task_type == "regression":
            self.booster = xgb.XGBRegressor(**self.params).fit(x_data, y_data).booster()
        else:
            if grp_len_list is None:
                raise ValueError(colors.RED + "You must set your group length infomation befor training!!" + colors.ENDC)
            dtrain = xgb.DMatrix(x_data, label = y_data)
            dtrain.set_group(grp_len_list)
            self.booster = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=150)
        sorted_fscores , sorted_scores = self.cal_feature_importance(importance_type)
        average_metric = self.evaluate(x_data, y_data, grp_len_list)
        return average_metric

    def predict(self, x_data, y_data=None, predict_result_saveto=None):
        """
            预测接口
            Args:
                x_data: pandas dataframe, 特征数据
                y_data: numpy array, 目标y数据
                predict_result_saveto: str, 预测结果保存路径
            Returns:
                y_pred: numpy array 预测值
                new_x_data: pandas dataframe 
        """
        if self.booster is None:
            raise ValueError("The model is empty!!")
        mdata = xgb.DMatrix(x_data, y_data)
        if self.task_type in ["regression", "ranking"]:
            y_pred = self.booster.predict(mdata)
        else:
            y_pred = self.booster.predict(mdata)
             
        new_x_data = copy.deepcopy(x_data) 
        if y_data is not None:
            new_x_data.insert(0, "real_score", y_data)
            
        new_x_data.insert(0, "predict_score", y_pred)
        
        if predict_result_saveto is not None:
            try:
                new_x_data.to_csv(predict_result_saveto, sep="\t", index=False)
            except:
                print colors.RED + "Fail to save the predict result as %s"%predict_result_saveto + colors.ENDC
        
        return y_pred, new_x_data

    def evaluate(self, x_data, y_data, grp_len_list=None):
        """
            效果评估接口
            Args:
                x_data: 
                y_data: 
                grp_len_list: list, a list of instance count of each group
            Returns:
                average_metric: float
        """
        if self.booster is None:
            raise ValueError(colors.RED + "Your XgbModel is empty!!" + colors.ENDC)
        y_pred, new_x_data = self.predict(x_data)
        
        if grp_len_list is None:
            grp_len_list = [len(y_pred)]

        group_cnt = len(grp_len_list)
        if group_cnt == 0:
            raise ValueError(colors.RED + "The group length list is empty!!"+colors.ENDC)
        
        instance_cnt = len(y_pred)
        if instance_cnt != sum(grp_len_list):
            raise ValueError(colors.RED + "The instance count does not match the group length info!!" + colors.ENDC)
        
        metrics = Metrics(y_data, y_pred, grp_len_list)
        
        if self.task_type == "regression":
            metrics.cal_grp_avg_regression_metric()
        elif self.task_type == "classification":
            metrics.cal_grp_avg_classification_metric()
        else:
            metrics.cal_grp_avg_ranking_metric() 
