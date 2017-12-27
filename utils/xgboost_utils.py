# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

import os
import pickle
import re
from common import *

class XgboostUtil(object):
    """
        Init the class with model file in binary format.
    """
    def __init__(self, fname, given_features=None):
        """
            Args:
                given_features: set
        """
        self.given_features = set() if given_features is None else given_features
        try:
            self.model = pickle.load(open(fname, "rb"))
            print colors.GREEN + "%s has been loaded successfully in binary format!!"%fname + colors.ENDC     
        except:
            raise ValueError(colors.RED+"Model %s faied to be loaded!!"%fname + colors.ENDC)
        self.fscores = self.model.booster().get_fscore()
        self.feature_set = set(self.fscores.keys())
        self.feature_set = self.feature_set | set(self.given_features)
        self.tmp_file = "tmp_model.txt"
    
    def conv2py(self, output_py_model="model.py", predict_function_name="gbdt_predict"):
        """Convert the raw text model to py file"""
        self.model.booster().dump_model(self.tmp_file)
        all_nodes_info, total_tree_num = parse_raw_text_model_file(self.tmp_file)
        fw = open(output_py_model, "w")
        fw.write('import sys\nreload(sys)\nsys.setdefaultencoding(\'utf8\')\n\n')

        def print_one_tree(tree_node_id):
            try:
                tree_id, node_id = tree_node_id.split("_")
                tree_id = int(tree_id)
                node_id = int(node_id)
            except:
                raise ValueError("Wrong model file!!")
            
            def traverse(tree_node_id):
                fw.write("\ndef nodeFunc_%s(feature_dict):\n"%tree_node_id) 
                node_info_dict = all_nodes_info[tree_node_id]
                if "leaf" in node_info_dict:
                    fw.write("\treturn %f\n"%node_info_dict["leaf"])
                else:
                    operators = ["<", "<=", ">", ">=", "=="]
                    condition = node_info_dict["condition"]
                    feature_name = ""
                    optor = ""
                    value = ""
                    for operator in operators:
                        content = condition.split(operator)
                        if len(content) == 2:
                            feature_name = content[0]
                            opter = operator
                            value = content[1]
                            break
                    if feature_name == "":
                        raise ValueError("Wrong model file!!")
                    
                    fw.write("\tif \"%s\" not in feature_dict or feature_dict[\"%s\"] is None:\n"%(feature_name, feature_name))
                    fw.write("\t\treturn nodeFunc_%s(feature_dict)\n"%node_info_dict["missing"])
                    fw.write("\tif feature_dict[\"%s\"] %s %s:\n\t\treturn nodeFunc_%s(feature_dict)\n"%(feature_name, opter, value, node_info_dict["yes"]))
                    fw.write("\telse:\n\t\treturn nodeFunc_%s(feature_dict)\n"%node_info_dict["no"])
                    
                    traverse(node_info_dict["yes"])
                    traverse(node_info_dict["no"])
            
            fw.write("\ndef treeFunc_%d(feature_dict):\n"%tree_id)
            fw.write("\treturn nodeFunc_%s(feature_dict)\n"%tree_node_id)
            traverse(tree_node_id) 
             
        fw.write("\ndef %s(feature_dict):\n"%predict_function_name)
        fw.write("\tresult = 0.0\n")
        for tree_id in xrange(total_tree_num):
            fw.write("\tresult += treeFunc_%d(feature_dict)\n"%tree_id)
        fw.write("\treturn result\n")
        
        for tree_id in xrange(total_tree_num):
            #DFS for the gbdt tree 
            print_one_tree("%d_0"%tree_id)
        
        fw.close()
        os.popen("rm %s"%self.tmp_file)
        
    def conv2cpp(self, output_cpp_model="model.hpp", output_cpp_struct="model_def.hpp", predict_function_name='gbdt_predict', struct_name='GBDT_FEATURE_INSTANCE'):
        """Convert the raw text model to cpp file.
        """
        self.model.booster().dump_model(self.tmp_file)
        all_nodes_info, total_tree_num = parse_raw_text_model_file(self.tmp_file)
         
        def generate_struct_cpp(output_cpp_struct): 
            fw = open(output_cpp_struct, "w")
            #generate the header file protection
            fw.write("#ifndef __GBDT_FEATURE_INSTANCE__\n#define __GBDT_FEATURE_INSTANCE__\n\n") 
            
            fw.write("struct %s{\n"%struct_name)
            for feature_name in self.feature_set:
                fw.write("\tfloat %s;\n"%feature_name)
            
            fw.write("//default construction: set all value to 0\n//DO NOT USE MEMSET to set zero, USE \"it = ZERO_INST\" instead, because the struct is not a POD now!\n")
            fw.write("\t%s(){\n"%struct_name)
            for feature_name in self.feature_set:
                fw.write("\t\t%s=0.0;\n"%feature_name)
            fw.write("\t}\n")

            fw.write("};\n\n")
            
            fw.write("static const %s ZERO_INST;\n"%struct_name)
            fw.write("#endif //__GBDT_FEATURE_INSTANCE__\n")
            fw.close() 
            
        def generate_model_cpp(output_cpp_model):
            fw = open(output_cpp_model, "w")
            #generate the header file protection
            fw.write("""#ifdef __GBDT_PREDICT__
#error this header file should not be include twice!!
#endif
#ifndef __GBDT_PREDICT__
#define __GBDT_PREDICT__\n""")
            
            #generate the include lines
            fw.write("#include <cmath>\n") 
            fw.write("#include \"model_def.hpp\"\n")
            fw.write("")
            #generate util function
            fw.write("""
#ifndef CHECK_NAN
#define CHECK_NAN
template<typename T>
inline bool CheckNAN(T v) {
#ifdef _MSC_VER
    return (_isnan(v) != 0);
#else
    return isnan(v);
#endif
}
#endif\n\n""")
         
            #generate the predict function
            fw.write("inline float %s(const %s &it) {\n"%(predict_function_name, struct_name))
            fw.write("float response = 0.0;\n")
            
            def traverse(tree_node_id):
                node_info_dict = all_nodes_info[tree_node_id]
                try:
                    tree_id, node_id = tree_node_id.split("_")
                    tree_id = int(tree_id)
                    node_id = int(node_id)
                except:
                    raise ValueError("Wrong model file!!")

                fw.write("N%s:\n"%(tree_node_id))
                if "leaf" in node_info_dict:
                    fw.write("\tresponse += %f;\n"%node_info_dict["leaf"])
                    fw.write("\tgoto T%d;\n"%(tree_id+1))
                else:
                    operators = ["<", "<=", ">", ">=", "=="]
                    condition = node_info_dict["condition"]
                    feature_name = ""
                    for operator in operators:
                        content = condition.split(operator)
                        if len(content) == 2:
                            feature_name = content[0]
                            break
                    if feature_name == "":
                        raise ValueError("Wrong model file!!")
                    fw.write("\tif(CheckNAN(it.%s)) goto N%s;\n"%(feature_name, node_info_dict["missing"]))
                    fw.write("\tif(it.%s) goto N%s; else goto N%s;\n"%(condition, node_info_dict["yes"], node_info_dict["no"]))
                    traverse(node_info_dict["yes"])
                    traverse(node_info_dict["no"])
             
            #generate the predict path according the tree struction
            for tree_id in xrange(total_tree_num):
                fw.write("T%d:\n"%tree_id)
                #DFS for the gbdt tree 
                traverse("%d_0"%tree_id)
            fw.write("T%d:\n"%total_tree_num)
            fw.write("\treturn response;\n")
             
            fw.write("}\n\n")
            fw.write("#endif //__GBDT_PREDICT__\n")
            fw.close()
        
        generate_struct_cpp(output_cpp_struct)
        generate_model_cpp(output_cpp_model)
        os.popen("rm %s"%self.tmp_file)


def parse_raw_text_model_file(filename):
    "解析xgboost 原始的text文件"
    all_nodes_info = dict()

    tree_id = None
    total_tree_num = 0

    with open(filename, "r") as fr:
        for line in fr:
            line = line.rstrip()
            root_match_res = re.match(re.compile(r"booster\[\d+\]"), line) #find the root node of a single tree
            if root_match_res is not None:
                tree_id = root_match_res.group().lstrip("booster[").rstrip("]")
                total_tree_num += 1
                 
            tree_node_match_res = re.search(re.compile(r"\d+:"), line)
            if tree_node_match_res is not None:
                node_id = tree_id + "_" + tree_node_match_res.group().rstrip(":")
                all_nodes_info[node_id] = dict()
            
                yes_match_res = re.search(re.compile("yes=\d+"), line)
                no_match_res = re.search(re.compile(r"no=\d+"), line)
                missing_match_res = re.search(re.compile(r"missing=\d+"), line)
                
                if yes_match_res is not None and no_match_res is not None and missing_match_res is not None:
                    condition_match_res = re.search(re.compile(r"\[.*\]"), line)
                    condition = condition_match_res.group().lstrip("[").rstrip("]")
                    yes_node_id = tree_id + "_" + re.search(re.compile(r"yes=\d+"), line).group().lstrip("yes=")
                    no_node_id = tree_id + "_" + re.search(re.compile(r"no=\d+"), line).group().lstrip("no=")
                    missing_node_id = tree_id + "_" + re.search(re.compile(r"missing=\d+"), line).group().lstrip("missing=")
            
                    all_nodes_info[node_id]["condition"] = condition
                    all_nodes_info[node_id]["yes"] = yes_node_id
                    all_nodes_info[node_id]["no"] = no_node_id
                    all_nodes_info[node_id]["missing"] = missing_node_id

                leaf_match_res = re.search(re.compile("leaf=.*"), line)
                
                if leaf_match_res is not None:
                    leaf_value = float(leaf_match_res.group().lstrip("leaf=") )
                    all_nodes_info[node_id]["leaf"] = leaf_value
    return all_nodes_info, total_tree_num


if __name__ == "__main__":
    bin_model_file = "../model/coupon_model.txt"
    xgb_util = XgboostUtil(bin_model_file, given_features=["user_sku_follow_cnt"])
    #xgb_util.conv2py()
    xgb_util.conv2cpp(predict_function_name='gbdt_predict_coupon')
