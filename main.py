#-*- coding:utf-8 -*-
"""
    Description: 
        main.py
    Author: shelldream
    Date: 2017-11-19
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import argparse

from utils.common import colors
from DataLoader.DataLoader import DataLoader
from XgbModel.XgbModel import  XgbModel

def main():
    parser = argparse.ArgumentParser(description="This is Aragon platform!")
    
    #Arguments about loading data
    parser.add_argument("--data_type", choices=["libsvm", "csv_with_fmap", "csv_with_table_header"], dest="data_type", default="libsvm", help="Choose the data type you want to use.\
        The choices include libsvm, csv_with_fmap, csv_with_table_header")
    parser.add_argument("--train_data_path", action="append", dest="train_data", default=None, help="Choose the \
        training data path. You can choose more than one file path.")
    parser.add_argument("--test_data_path", action="append", dest="test_data", default=None, help="Choose the \
        test data path. You can choose more than one file path.")
    parser.add_argument("--validation_data_path", action="append", dest="valid_data", default=None, help="Choose the \
        validation data path. You can choose more than one file path.")
    parser.add_argument("--fmap_filename", default=None, dest="fmap", help="Choose your feature schema file.You need one file if you choose csv_with_fmap load type.")
    parser.add_argument("--black_feature", action="append", dest="black_feature_list", default=None, help="Choose the \
        black feature you don't need!")
    parser.add_argument("--groupby", default=None, dest="groupby", help="The groupby key for your data.")
    parser.add_argument("--delimiter", default="\t", dest="delimiter", help="Choose the delimiter for your raw data.")
    parser.add_argument("--target_y", default="label", dest="target_y", help="Choose the target column for your raw data.")
    parser.add_argument("--query_id", default=False, dest="query_id", help="Set the query_id True if you want the query_id array!")

    #Arguments about task
    parser.add_argument("--task", choices=["regression", "classification", "ranking"], dest="task_type", default="classification",
        help="Choose the task type you want to use! The choices include regression, classification, ranking")
    parser.add_argument("--mode", choices=["train", "predict", "evaluation"], dest="mode", default="train", \
        help="Choost the task mode you want to use! The choices include train, predict, evaluation")

    #Arguments about model
    parser.add_argument("--model", choices=["xgboost", "vowpal_wabbit"], dest="model", default="xgboost", help="Choose the model you want to use.\
        The choices include xgboost, vowpal_wabbit.")
    parser.add_argument("--parameters", default="{}", dest="parameters", help="Choose the parameters for your model and \
        the format of your parameters is dict format!")
    parser.add_argument("--boost_round", default=100, dest="boost_round", help="Number of boosting iterations")
    parser.add_argument("--model_path", default="./model/default_model", dest="model_path", help="Set your model path for model saving or loading")
    parser.add_argument("--predict_result_path", default="./predict/predict_result", dest="predict_result_path", help="Set your file path to save the prediction result")
    parse_args(parser)


def parse_args(parser):
    """
        解析输入的参数
        Args:
            parser: argparse.ArgumentParser
        Rets:
            None
    """
    args = parser.parse_args()
    x_train, y_train, complete_train = None, None, None 
    x_test, y_test, complete_test = None, None, None 
    x_valid, y_valid, complete_valid = None, None, None 
    
    if args.train_data: #训练集非空
        train_data_loader = DataLoader(args.train_data, args.data_type, args.target_y, args.black_feature_list, args.fmap, args.delimiter, args.query_id) 
        x_train, y_train, complete_train, grp_len_list_train = train_data_loader.load()
        
    if args.test_data: #测试集非空
        test_data_loader = DataLoader(args.test_data, args.data_type, args.target_y, args.black_feature_list, args.fmap, args.delimiter, args.query_id) 
        x_test, y_test, complete_test, grp_len_list_test = test_data_loader.load()
      
    if args.valid_data: #验证集非空
        valid_data_loader = DataLoader(args.valid_data, args.data_type, args.target_y, args.black_feature_list, args.fmap, args.delimiter, args.query_id) 
        x_valid, y_valid, complete_valid, grp_len_list_valid = valid_data_loader.load()
     
    try:
        param_dict = eval(args.parameters)
        if args.mode == "train":
            print colors.BLUE + "param_dict:", param_dict , colors.ENDC
    except:
        raise ValueError(colors.RED + "Wrong parameters!!" + colors.ENDC)
    
    if args.model == "xgboost":
        model = XgbModel(param_dict, args.task_type)
    else:
        pass
    
    if args.mode == "train":
       model.train(x_train, y_train, )  

if __name__ == "__main__":
    main()
