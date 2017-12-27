# -*- coding:utf-8 -*-
"""
    Description:
        用来检查各维特征的有效性，包括检查
        1. 各维特征的数据分布
        2. 各维特征和目标的相关性
    Author: shelldream
    Date: 2017-11-23
"""
import sys
reload(sys).setdefaultencoding("utf-8")
from sklearn.cluster import KMeans
import numpy as np
import scipy.stats as stats
import pandas as pd

class FeatureChecker(object):
    def __init__(self, x_data, y_data):
        """
            Args:
                x_data: pandas dataframe, 特征数据
                y_data: numpy array, 目标数据y
            Returns:
                None
        """
        self.x_data = x_data
        self.y_data = y_data
    
    def check_label_distribution(self):
        """
            检查目标数据的分布情况
        """
        label_stats = pd.value_counts(self.y_data)
        label_total_cnt = len(self.y_data)
        print "The distribution of y: "
        for label, count in label_stats.iteritems():
            print "%f  %f  %f"%(label, count, count*1.0/label_total_cnt)

    def check_distribution_roughly(self, feat_name):
        """
            粗略计算某一维特征的数据分布情况
            Args:
                feat_name: str, 需要计算的特征名称
            Returns:
                feat_data: pandas Series, 该维特征对应的数据
        """
        if feat_name not in self.x_data.columns:
            raise ValueError("%s doesn't exist in the dataframe!!"%feat_name) 

        feat_data = self.x_data[feat_name]
        print "Statistical Key Figures of %s"%feat_name
        total_count = self.x_data.shape[0]
        not_nan_count = feat_data.count()
        not_nan_ratio = float(not_nan_count)/total_count
        print "not_nan_ratio\t%f" %not_nan_ratio
        print  feat_data.describe()
        return feat_data

    def check_distribution_precisely(self, feat_name, bin_num=7):
        """
            精细计算某一维特征的数据分布情况
            利用一维kmeans 对于该维特征进行聚类分bin,得到该维特征对应的统计直方图信息
            Args:
                feat_name: str,需要计算的特征名称
                bin_num: int, 对于该维特征的特征值进行分bin的个数
            Returns:
                None
        """
        feat_data = self.check_distribution_roughly(feat_name)
        print "\nThe histogram of %s"%feat_name
        raw_feat_data = np.array(feat_data)
        feat_data = zip(raw_feat_data, [0 for i in range(len(feat_data))])
        kmeans = KMeans(n_clusters=bin_num, random_state=0).fit(feat_data)  #1-D kmeans to split the data into specific bucket
        
        bin_min = {k:float("inf") for k in xrange(bin_num)} #每个bin的下界
        bin_max = {k:float("-inf") for k in xrange(bin_num)} #每个bin的上界
        bin_sample_cnt = {k:0 for k in xrange(bin_num)}  #每个bin中样本的个数
         
        labels = kmeans.labels_
        
        for i in xrange(len(raw_feat_data)):
            value = raw_feat_data[i]
            label_k = labels[i]
            bin_sample_cnt[label_k] += 1.0
            
            if value < bin_min[label_k]:
                bin_min[label_k] = value

            if value > bin_max[label_k]:
                bin_max[label_k] = value
            
        bin_min_max_cnt = []
        for k,v in bin_min.items():
            if bin_sample_cnt[k] > 0:
                bin_min_max_cnt.append((bin_min[k], bin_max[k], bin_sample_cnt[k]))

        bin_min_max_cnt = sorted(bin_min_max_cnt, key=lambda x:x[0], reverse=False)
        
        for item in bin_min_max_cnt: 
            print "%f——%f: %d  %f"%(item[0], item[1], item[2], item[2]/len(raw_feat_data))
        
    def cal_correlation(self, feat_name, groupby=None, metrics_func=None):
        """
            计算某维特征和目标的相关性
            1. 计算该维特征和目标的皮尔逊系数
            2. 按照给定的metric 计算函数，计算单维特征的metric
            Args:
                feat_name: str, 特征名称
                groupby: str, 计算metric时的groupby key
                metric_func: function, metric计算function
            Returns:
                None 
        """
        labels = self.y_data
        feats = self.x_data[feat_name].tolist()
        
        feat_label = zip(feats, labels)
         
        value_label_dict = {}
        
        for (value, label) in feat_label:
            value_label_dict[value] = value_label_dict.get(value, [])
            value_label_dict[value].append(float(label))
        
        x_list = []
        y_list = []
        for k,v_list in value_label_dict.items():
            if len(v_list) > 0:
                x_list.append(k)
                y_list.append(sum(v_list)/len(v_list))
        
        print "(Person correlation, p-value) of %s"%feat_name
        corrcoef = stats.pearsonr(x_list, y_list) 
        print corrcoef
        
        if metrics_func is not None:
            print "The metrics of %s"%feat_name
            if groupby is None:
                metrics = metrics_func(labels, feats)
            else:
                complete_data = self.x_data.insert(0, 'label', self.y_data)
                grouped = complete_data.groupby(groupby)
                metrics_list = []
                for key, data in grouped:
                    values = data[feat_name].tolist()
                    scores = data['label'].tolist()
                    if len(set(scores)) == 1:
                        continue
                    metrics_list.append(metrics_func(scores, values))
                
                if len(metrics_list) > 0:
                    metrics = sum(metrics_list)/len(metrics_list)  #average metric
            print metrics

