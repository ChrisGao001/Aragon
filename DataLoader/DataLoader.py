# -*- coding:utf-8 -*-
"""
    Description: 数据加载模块
    Author: shelldream
    Date: 2017-11-19
"""
import sys
reload(sys).setdefaultencoding("utf-8")
sys.path.append("./")
sys.path.append("../")
import os
import copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

import utils.common as common

class DataLoader:
    def __init__(self, filename_list, data_type, groupby=None, target_y="label", \
        black_feature_list=None, fmap_filename=None, delimiter="\t", query_id=False):
        """
            Args:
                filename_list: list, 待加载的数据文件名列表
                data_type: str,  取值为 “libsvm”、“csv_with_fmap”、"csv_with_table_header" 其中一个
                groupby: str, 数据分组的关键字
                target_y: str, 指定哪一列数据是y, csv 数据格式必须指定
                black_feature_list: list, 需要过滤的特征
                fmap_filename: str, fmap 文件名
                delimiter: str, 分隔符
                query_id: libsvm 带queryid格式参数
            Returns:
                None
        """
        self.filename_list = filename_list
        self.data_type = data_type
        self.target_y = target_y
        self.black_feature_list = black_feature_list
        self.fmap_filename = fmap_filename
        self.delimiter = delimiter
        self.groupby = groupby
        self.query_id = query_id
        
        self.ft_name_type_map = None
        self.ft_name_list = None
        self.queryid_list = []
        self.columns = None
        self.x_data = None
        self.y_data = None
        self.complete_data = None
        self.grp_len_list = []

    def load(self):
        """
            加载数据执行函数
            Args:
                None
            Returns:
                x_data: pandas dataframe, 特征黑名单过滤后的x数据
                y_data: numpy array, 目标数据y
                complete_data: pandas dataframe, 特征黑名单过滤前的完整数据
                grp_len_list: list, instance count for each group
        """
        if self.fmap_filename is not None:
            self._parse_schema()

        if self.data_type == "libsvm":
            self._load_libsvm_data()
        elif self.data_type == "csv_with_fmap":
            self._load_csv_with_fmap_data()
        elif self.data_type == "csv_with_table_header":
            self._load_csv_with_table_header()
        else:
            raise ValueError(common.colors.RED + "Please check the data_type parameter!" + common.colors.ENDC)
        
        try:
            self.y_data = self.complete_data[self.target_y].values  
        except:
            raise ValueError(common.colors.RED + "%s doesnot exist in the schema!!"%self.target_y + common.colors.ENDC)
        
        self.x_data = copy.deepcopy(self.complete_data)
        self.x_data.pop(self.target_y)
        
        if self.black_feature_list is not None:
            self._filter_feature()
        return self.x_data, self.y_data, self.complete_data, self.grp_len_list
    
    def _parse_schema(self):
        """
            解析fmap文件，获取数据的schema
        """
        if not os.path.exists(self.fmap_filename):
            raise ValueError(common.colors.RED + self.fmap_filename + "doesn't exist!!" + common.colors.ENDC)

        dtype_map = {
        "int": np.int32,
        "float": np.float64,
        "str": np.object,
        "string": np.object
        }
        
        self.ft_name_list = []
        self.ft_name_type_map = dict()
        with open(self.fmap_filename, "r") as f_map_r:
            for line in f_map_r:
                try:
                    if "#" not in line: #注释行
                        index, feature_name, data_type = line.rstrip().split("\t")
                        self.ft_name_list.append(feature_name)
                        self.ft_name_type_map[feature_name] = dtype_map[data_type]
                    else:
                        continue
                except:
                    print common.colors.YELLOW + "Invalid line in %s when parsing feature schema file!"%line + common.colors.ENDC 
    
    def _get_grp_len_list(self, group_key_list):
        grp_len_list = []
        instance_cnt = 0
        last_group_key = None
        for group_key in group_key_list:
            if group_key != last_group_key:
                if last_group_key is not None:
                    grp_len_list.append(instance_cnt)
                instance_cnt = 1
            else:
                instance_cnt += 1
            last_group_key = group_key
        
        if instance_cnt != 0:
            grp_len_list.append(instance_cnt)
        return grp_len_list
         
    def _load_libsvm_data(self):
        """加载libsvm格式原始数据""" 
        frames = []
        for filename in self.filename_list:
            try:
                data = load_svmlight_file(filename, query_id=self.query_id)
                print common.colors.GREEN + "Load datafile %s successfully!!"%filename + common.colors.ENDC 
            except:
                print common.colors.YELLOW + "Can not load datafile " + filename + common.colors.ENDC
                continue

            x = np.array(data[0].todense())
            y = data[1]
            x = np.column_stack((y, x))
            
            if self.query_id:
                self.queryid_list += data[2]
             
            if self.ft_name_list is not None: 
                x = pd.DataFrame(data=x, columns=self.ft_name_list)
            else:
                x = pd.DataFrame(data=x)
            frames.append(x)
            
        self.complete_data = pd.concat(frames, axis=0)            
        self.complete_data = self.complete_data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index
        
        if self.query_id:
            self.grp_len_list = self._get_grp_len_list(self.queryid_list) 
        else:
            self.grp_len_list.append(self.complete_data.shape[0]) 

    def _load_csv_with_fmap_data(self):
        """根据指定的feature map文件载入csv格式的数据"""
        if self.ft_name_list is None or self.ft_name_type_map is None or len(self.ft_name_list) == 0:
            raise ValueError(common.colors.RED + "Faild to parse The feature map file %s!!"%self.fmap_filename + common.colors.ENC)
        
        frames = []
        for filename in self.filename_list:
            try:
                df = pd.read_csv(filename, sep=self.delimiter, header=None, names=self.ft_name_list, dtype=self.ft_name_type_map) 
                print common.colors.GREEN + "Load datafile %s successfully!!"%filename + common.colors.ENDC 
                frames.append(df)
            except:
                print common.colors.YELLOW + "Can not load datafile " + filename + common.colors.ENDC
                continue
            
        self.complete_data = pd.concat(frames, axis=0)            
        self.complete_data = self.complete_data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index
        if self.groupby is not None:
            self.grp_len_list = self._get_grp_len_list(self.complete_data[self.groupby])
        else:
            self.grp_len_list.append(self.complete_data.shape[0]) 

    def _load_csv_with_table_header(self):
        """载入带表头的的文本数据，schema为文件第一行"""
        frames = []
        for filename in self.filename_list: 
            try:
                df = pd.read_table(filename, sep=self.delimiter)
                print common.colors.GREEN + "Load datafile %s successfully!!"%filename + common.colors.ENDC 
                frames.append(df)
            except:
                print common.colors.YELLOW + "Can not load datafile " + filename + common.colors.ENDC
                continue
        self.complete_data = pd.concat(frames, axis=0)            
        self.complete_data = self.complete_data.reset_index(drop=True)  #合并不同数据文件的数据，然后重置index
        if self.groupby is not None:
            self.grp_len_list = self._get_grp_len_list(self.complete_data[self.groupby])
        else:
            self.grp_len_list.append(self.complete_data.shape[0]) 

    def _filter_feature(self):
        """根据指定的特征黑名单过滤数据"""
        for feature in self.black_feature_list:
            try:
                self.x_data.pop(feature)
            except:
                print common.colors.YELLOW + "%s does not in the feature schema"%feature + common.colors.ENDC
        
