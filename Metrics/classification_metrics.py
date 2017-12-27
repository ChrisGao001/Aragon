# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

import sklearn.metrics as sk_metrics

def cal_confuse_matrix(labels, y_preds, theshold, pos_label=1):
    """
        Calculate the confusion matrix for the given score theshold
        If the predict score is higher than the theshold, the label should be postive label.
        Args:
            labels: list, The real label of the instances.
            y_preds: list, The predict score of the instances.
            theshold: float, The theshold for the classifier to determine the output label.
        Rets:
            confusion_matrix: dict,e.g: {"TP":100, "FP":100, "FN":100, "TN":100}
    """
    if len(labels) != len(y_preds):
        raise ValueError("The lengths of scores and y_preds are not the same!!")
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_preds)):
        score = y_preds[i]
        label = labels[i]
        if score >= theshold:
            if label == pos_label:
                TP += 1
            else:
                FP += 1
        else:
            if label != pos_label:
                TN += 1
            else:
                FN += 1 
    confusion_matrix = {"TP":TP, "FP":FP, "TN":TN, "FN":FN}
    return confusion_matrix


def cal_accuracy_score(labels, y_preds, normalize=True, sample_weight=None):
    """
        Calculate the accuracy score.
    """
    if len(labels) != len(y_preds):
        raise ValueError("The lengths of scores and y_preds are not the same!!")
    return sk_metrics.accuracy_score(labels, y_preds, normalize, sample_weight)


def cal_auc_pr(labels, y_preds, pos_label=1):
    """Precision Recall curve auc"""
    label_score_pairs = zip(labels, y_preds) 
    label_score_pairs =  sorted(label_score_pairs, key=lambda x:x[1], reverse=True) # sort by the score desc
    
    p_num = 0
    n_num = 0
    for pair in label_score_pairs:
        if abs(pair[0] - pos_label) < 10e-7:
            p_num += 1
        else:
            n_num += 1
    """
        Precision: P= TP/(TP+FP)
        Recall: R=TP/(TP+FN)
                R=TP/p_num
    """ 
    auc = 0.0
    TP = 0.0
    FP = 0.0
    P0 = 0.0
    R0 = 0.0

    precision_recall_list = []

    for (label, score) in label_score_pairs:
        if abs(label- pos_label) < 10e-7:
            TP += 1.0
        else:
            FP += 1.0
        
        P1 = TP/(TP+FP)
        R1 = TP/p_num
        
        auc += (R1-R0)*abs(P1-P0)*0.5
        
        P0 = P1
        R0 = R1
        
    return auc      


def cal_auc(labels, y_preds, pos_label=1):
    """
        ROC AUC
    """
    fpr, tpr, thresholds = sk_metrics.roc_curve(labels, y_preds, pos_label=pos_label)
    auc = sk_metrics.auc(fpr, tpr)
    return auc


def cal_auc_v2(labels, y_preds, pos_label=1):
    label_score_pairs = zip(labels, y_preds) 
    label_score_pairs =  sorted(label_score_pairs, key=lambda x:x[1], reverse=True) #按照score降序排序
    
    p_num = 0
    n_num = 0
    for pair in label_score_pairs:
        if pair[0] == pos_label:
            p_num += 1
        else:
            n_num += 1
    
    """
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
    """
    tpr0 = 0
    fpr0 = 0
    tp = 0
    fp = 0
    auc = 0
    for pair in label_score_pairs:
        if pair[0] == 1:
            tp += 1.0
            tpr1 = tp/p_num
            fpr1 = fp/n_num       
            auc += (tpr0+tpr1)*(fpr1-fpr0)/2
            tpr0 = tpr1
            fpr0 = fpr1
        else:
            fp += 1.0

    auc += 1.0-fpr0
    return auc    

