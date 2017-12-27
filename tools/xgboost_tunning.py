# -*- coding:utf-8 -*-
"""
    Description: 
        xgboost tunning 工具
        解析xgboost的原始text文件，调用tunning函数，调试模型
    Author: shelldream
    Date: 2017.11.19
"""
import sys
reload(sys).setdefaultencoding("utf-8")
import re
import numpy as np

sys.path.append("./")
sys.path.append("../")

import utils.xgboost_utils

def get_xgboost_score(all_nodes_info, total_tree_num, feature_dict):
    score = 0.0
    node_id = None
    operators = ["<", "<=", ">", ">=", "=="]
    for i in xrange(total_tree_num):
        tree_id = str(i)
        node_id = tree_id + "_0"
        while True: #traverse a singe tree
            if "condition" in all_nodes_info[node_id]:
                for operator in operators:
                    try:
                        condition_key, condition_value = all_nodes_info[node_id]["condition"].split(operator)
                        if condition_key not in feature_dict:
                            print "feature %s is missing in your feature dict. Set the value of the feature nan!"%condition_key
                            real_value = None
                        else:
                            real_value = feature_dict[condition_key]
                        try:
                            condition_value = float(condition_value)
                        except:
                            condition_value = int(condition_value)
                        
                        if operator == "<":
                            if real_value is None:
                                node_id = all_nodes_info[node_id]["missing"]
                            elif real_value < condition_value:
                                node_id = all_nodes_info[node_id]["yes"]
                            else:
                                node_id = all_nodes_info[node_id]["no"]
                        break
                    except:
                        continue
            elif "leaf" in all_nodes_info[node_id]:
                score += all_nodes_info[node_id]["leaf"]
                break  #Break the loop until reach the leaf node
    return score


def tunning():
    """
        Usage:
            python xgboost_tunning.py demo_model.txt
            please input feature dict or modify your feature:{"bm25_2_all_dump":5.32322, "bm25_12_all_dump":10.3232, "bm25_12_dump":3.322, "bm25_2_dump":0.3232}
            score:3.271109
            please input feature dict or modify your feature:"bm25_2_all_dump":0      
            Now the feature dict is  {'bm25_2_dump': 0.3232, 'bm25_12_dump': 3.322, 'bm25_12_all_dump': 10.3232, 'bm25_2_all_dump': 0.0}
            score:3.450927
            please input feature dict or modify your feature:"bm25_2_all_dump":0,'bm25_12_all_dump': 0     
            Now the feature dict is  {'bm25_2_dump': 0.3232, 'bm25_12_dump': 3.322, 'bm25_12_all_dump': 0.0, 'bm25_2_all_dump': 0.0}
            score:2.450927
    """
    filename = sys.argv[1]
    all_nodes_info, total_tree_num = utils.xgboost_utils.parse_raw_text_model_file(filename)
    feature_dict = dict()
    while True:
        raw_data = raw_input("please input feature dict or modify your feature:")
        try:
            feature_dict = eval(raw_data)
            xgboost_score = get_xgboost_score(all_nodes_info, total_tree_num, feature_dict)
            xgboost_score = 1.0/(1+np.exp(-xgboost_score))
            print "score:%f"%xgboost_score
        except:
            try:
                feature_value_list = raw_data.split(",")
                for item in feature_value_list:
                    feature, value = item.split(":")
                    feature = feature.strip('"').strip("'")
                    feature_dict[feature] = float(value)
                print "Now the feature dict is ", feature_dict
                xgboost_score = get_xgboost_score(all_nodes_info, total_tree_num, feature_dict)
                xgboost_score = 1.0/(1+np.exp(-xgboost_score))
                print "score:%f"%xgboost_score
            except:
                print "Wrong feature dict format!!"
                continue
        

if __name__ == "__main__":
    tunning()
