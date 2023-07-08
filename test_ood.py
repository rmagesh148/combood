import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, det_curve, average_precision_score, roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.datasets import cifar10, mnist
from sklearn import preprocessing

from confidenciator import Confidenciator, split_features
from data import distorted, calibration, out_of_dist, load_data, load_svhn_data, imagenet_validation, save_missing_indices_images_in_folder,save_missing_document_indices_images_in_folder
import data
from data import save_missing_cifar10_indices_images_in_folder_for_mnist_id
from utils import binary_class_hist, df_to_pdf
from models.load import load_model
import sys
import math
import seaborn as sns
from matplotlib import pyplot as plt2
import pickle
import time
import random
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def compute_confusion_metrix(in_dist, out_dist,dataset_name,featuretester_method):
    #=#
    print("compute_confusion_metrix()")
    print("flag 1.27 featuretester_method  :",featuretester_method)
    print("flag 1.27 dataset_name  :",dataset_name)
    print("np.shape(in_dist): ",np.shape(in_dist))
    print("np.shape(out_dist): ",np.shape(out_dist))
    # print("(in_dist): ",in_dist )
    # print("(out_dist): ",out_dist )
    y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
    y_pred = np.concatenate([in_dist, out_dist])
    print("np.shape(y_true): ",np.shape(y_true))
    print("np.shape(y_pred): ",np.shape(y_pred))
    # print("(y_true): ",y_true)
    # print("(y_pred): ",y_pred)
    
    optimal_threshold = calculate_optimal_threshold(y_true,y_pred,dataset_name,featuretester_method)
    
    # Convert the predicted scores to binary predictions using a threshold of 0.5
    # Convert probabilities to binary predictions
    y_pred_binary = np.where(y_pred >= optimal_threshold, 1, 0)
    # y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
    
    # compute confusion metrix only for ood daata
    
    # Compute confusion matrix
    y_true_ood = np.zeros(len(out_dist))
    y_pred_ood = out_dist
    y_pred_ood_binary = np.where(y_pred_ood >= optimal_threshold, 1, 0)
    # cm = confusion_matrix(y_true, y_pred_binary)
    print("flag 1.28 np.shape(y_true_ood): ",np.shape(y_true_ood))
    print("flag 1.28 np.shape(y_pred_ood): ",np.shape(y_pred_ood))
    cm = confusion_matrix(y_true_ood, y_pred_ood_binary)
    
    print("flag 1.29 cm: ",cm)
    
    tn = cm[0][0]  # True Negatives
    fp = cm[0][1]  # False Positives
    fn = cm[1][0]  # False Negatives
    tp = cm[1][1]  # True Positives
    # tn, fp, fn, tp = cm.ravel()
    
    # Compute other performance metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    tpr = recall
    fpr = fp / (fp + tn)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    # Print the results
    print("\nflag 1.27 Confusion Matrix:")
    print("cm :",cm)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("TPR (Sensitivity):", tpr)
    print("FPR (1 - Specificity):", fpr)
    print("AUC-ROC:", roc_auc)
    print("Optimal Threshold:", optimal_threshold)
    
    cm_scores = pd.Series({
        "Testimages": len(out_dist),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positive": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TPR (Sensitivity)": tpr,
        "FPR (1 - Specificity)": fpr,
        "AUC-ROC": roc_auc,
        "Optimal Threshold": optimal_threshold,
        
    })
    
    return cm_scores

def taylor_scores(in_dist, out_dist,dataset_name,featuretester_method):
    print("\ntest_ood.py ==> taylor_scores()")
    print("featuretester_method 1.2 :",featuretester_method)
    print("dataset_name 1.2 :",dataset_name)
    print("np.shape(in_dist): ",np.shape(in_dist))
    print("np.shape(out_dist): ",np.shape(out_dist))
    # print("(in_dist): ",in_dist )
    # print("(out_dist): ",out_dist )
    y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
    y_pred = np.concatenate([in_dist, out_dist])
    print("np.shape(y_true): ",np.shape(y_true))
    print("np.shape(y_pred): ",np.shape(y_pred))
    # print("(y_true): ",y_true)
    # print("(y_pred): ",y_pred)

    fpr, fnr, thr = det_curve(y_true, y_pred, pos_label=1)
    det_err = np.min((fnr + fpr) / 2)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    fpr95_sk = fpr[np.argmax(tpr >= .95)]
    
    scores = pd.Series({
        "FPR (95% TPR)": fpr95_sk,
        "Detection Error": det_err,
        "AUROC": roc_auc_score(y_true, y_pred),
        "AUPR In": average_precision_score(y_true, y_pred, pos_label=1),
        "AUPR Out": average_precision_score(y_true, 1 - y_pred, pos_label=0),
    })
    return scores

def mcnemar_test(a, b):
    mcnemar_dict = {}
    if len(a) != len(b): 
        return None
    true_true_a = 0
    true_false_b = 0
    false_true_c = 0
    false_false_d = 0
                
    for i in range(0, len(a)):
        # print("flag 1.8 i",i)
        # print("flag 1.8 a[i]",a[i])
        # print("flag 1.8 b[i]",b[i])

        if a[i] and b[i]:
            true_true_a+=1
        elif a[i] and not b[i]:
            true_false_b+=1
        elif not a[i] and b[i]:
            false_true_c+=1
        elif not a[i] and not b[i]: 
            false_false_d+=1
        else:
            pass
    print(true_true_a, true_false_b, false_true_c, false_false_d)
    print("mcnemar_test :","\ntrue_true_a:", true_true_a, "\ntrue_false_b:", true_false_b, "\nfalse_true_c:", false_true_c, "\nfalse_false_d:", false_false_d)
    # mcnemar = (true_false_b - false_true_c)**2 / (true_false_b + false_true_c)
    
    table = [[true_true_a, true_false_b], [false_true_c, false_false_d]]
    result = mcnemar(table, exact=False, correction = True)
    mcnemar_dict = {"pvalue":result.pvalue , 
                    "statistic" :result.statistic }
    # return float(result.pvalue), result.statistic
    return mcnemar_dict

  
def get_mcnemar_for_all_ood_data(id_dataset,df1,df2):
    print("get_mcnemar_for_all_ood_data()")
    print("flag 1.11 mcnemar test for","id dataset-",id_dataset)
    mcnemar_test_dict = {}
    print("flag 1.11 df1.index : ", df1.index)
    print("flag 1.11 df2.index : ", df2.index)

    # loop over each index and calculate the sum of loss column for that index in both dataframes
    for index in df1.index:
        a= df1.loc[index,'y_binary']
        b= df2.loc[index,'y_binary']
        # print("flag 1.21 a :",a)
        # print("flag 1.21 b :",b)
        mcnemar_test_dict[index] = mcnemar_test(a,b)
    
    # p_value, stat_value =mcnemar_test(ybinary_knn_cifar10,ybinary_xood_mahala_pen_knn_log_sq_cifar10)
    df3 = pd.DataFrame(mcnemar_test_dict).transpose()
    df3.to_csv(f"{id_dataset}_mcnemar_test_dict.txt", sep="\t")
    df3.to_csv(f"{id_dataset}_mcnemar_test_dict_df3.csv")
    print('flag 1.11 mcnemar_test_dict:',mcnemar_test_dict)

    
def calculate_optimal_threshold(y_test, y_prob,dataset_name,featuretester_method):
    print("calculate_optimal_threshold()")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    # print("flag 1.6 roc_auc :",roc_auc)
    # print("flag 1.6 fpr :",fpr)
    # print("flag 1.6 tpr :",tpr)
    # print("flag 1.6 thresholds :",thresholds)
    
    # Plot the ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    # plt.savefig(f'roc_curve_{featuretester_method}_{dataset_name}.png')
    plt.show()
    plt.clf()
    
    # Find the optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]   
    print('flag 1.6 The optimal threshold is:', optimal_threshold)
    
    return optimal_threshold
    
    
def get_incorrect_indices(in_dist, out_dist,dataset_name,featuretester_method):
    print("get_incorrect_indices()")
    print("\ntest_ood.py ==> taylor_scores()")
    print("featuretester_method 1.2 :",featuretester_method)
    print("dataset_name 1.2 :",dataset_name)
    print("np.shape(in_dist): ",np.shape(in_dist))
    print("np.shape(out_dist): ",np.shape(out_dist))
    # print("(in_dist): ",in_dist )
    # print("(out_dist): ",out_dist )
    y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
    y_pred = np.concatenate([in_dist, out_dist])
    print("np.shape(y_true): ",np.shape(y_true))
    print("np.shape(y_pred): ",np.shape(y_pred))
    # print("(y_true): ",y_true)
    # print("(y_pred): ",y_pred)
    ## ##
    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred)
    y_true_ood = np.zeros(len(out_dist))
    y_pred_ood = out_dist
    
    optimal_threshold = calculate_optimal_threshold(y_true,y_pred,dataset_name,featuretester_method)
    
    # Convert the predicted scores to binary predictions using a threshold of 0.5
    # assuming for out_dist the probability shoule be less than optimal_threshold
    y_binary = y_pred_ood < optimal_threshold
    
    num_true = np.count_nonzero(y_binary)
    num_false = y_binary.size - num_true
    print(f"Number of True values: {num_true}")
    print(f"Number of False values: {num_false}")
    
    # Get the indices where the values of y_binary are True and False, respectively
    # Find the indices of the correct and incorrect predictions
    correct_indices = np.where(y_binary)[0]
    incorrect_indices = np.where(~y_binary)[0]
    
    # Print the number of correct and incorrect predictions and their indices
    print(f"Number of correct predictions: {len(correct_indices)}")
    print(f"Number of incorrect predictions: {len(incorrect_indices)}")
    # print("Indices of correct predictions:", correct_indices)
    # print("Indices of incorrect predictions:", incorrect_indices)
    
    incorrect_indices = pd.Series({
        "incorrect_indices": incorrect_indices,
        "y_binary":y_binary,
        "y_pred_ood":y_pred_ood,
    })
    return incorrect_indices #, y_binary


class FeatureTester: 
    # def __init__(self, mahala_xood, knn_pen, dataset: str, model: str, feature_model, name="", mahala_xood, knn_pen):
    def __init__(self,dataset: str, model: str, feature_model, name="", extreme= True, pen= True):
        mahala_xood = extreme
        knn_pen = pen
        print("\n\n-------------test_ood.py ==> FeatureTester-------------")
        self.ood = {}
        self.dataset = dataset
        self.model = model
        # data.img_shape = (32, 32, 3)
        data.img_shape = (224, 224, 3)
        self.data = data.load_dataset(dataset)  # type(self.data) = dict type
        
        print(self.data.keys())
        if "Train" in self.data.keys():
            print(type(self.data["Train"]))

        # self.data["Train"] = self.data["Train"].iloc[:100, :]
        # self.data["Val"] = self.data["Val"].iloc[:100, :]
        # self.data["Test"] = self.data["Test"].iloc[:100, :]

        self.testset_data = self.data["Test"]
        m, transform = load_model(dataset, model)
        # print("load_model : ", m)
        self.path = Path(f"results/{dataset}_{model}")
        self.path = (self.path / name) if name else self.path
        self.path.mkdir(exist_ok=True, parents=True)
        
        # print("Creating Confidenciator", flush=True)
        # print(type(self.data["Train"]))
        self.conf = Confidenciator(m, transform, self.data["Train"],mahala_xood,knn_pen)
        # self.conf.plot_model(self.path) TODO implement this.

        print("\n\n   ##  Adding Feature Columns   ##  ")
        # print("feature_model :", feature_model)
        
        # ==============================================================
        # # add_prediction_and_features to ID train, val and test data
        # ==============================================================
        for name, df in self.data.items():  
            print("flag 3.6 ", type(self.data[name]))
            if feature_model == "mahala":
                print("It is goign in mahala")
                print("running set  :",name)
                if not mahala_xood: 
                    self.data[name] = self.conf.add_prediction_and_penultimate_features_dl_to_mahala(self.data[name])
                else:
                    self.data[name] = self.conf.add_prediction_and_features_dl(
                        self.data[name]) # name = Train, Test, Val 

                
            elif feature_model == "knn":
                print("KNN PART IS GETTING EXECUTED")
                print("running set  :",name)
                # if not knn_pen:
                if  knn_pen:
                    self.data[name] = self.conf.add_prediction_and_features_knn(
                        self.data[name])
                else:
                    self.data[name] = self.conf.add_prediction_and_extreme_features_dl_to_knn(self.data[name])
                print("knn extreme feature shape: ", self.data[name].shape)
                
            else:
                print("executing for another feature_model ")
                mahala_df = self.conf.add_prediction_and_features_dl(
                    self.data[name])
                knn_df = self.conf.add_prediction_and_features_knn(
                    self.data[name])
                print("flag 3.1 type(mahala_df):",type(mahala_df))
                print("flag 3.1 type(knn_df):",type(knn_df))

                print("mahala_df.shape :",mahala_df.shape)
                print("knn_df.shape :",knn_df.shape)
                self.data[name] = pd.concat([mahala_df, knn_df], ignore_index=True, axis=1)

                print("flag 3.7 self.data[name].shape :",self.data[name].shape)
                
        print("flag 3 self.data.keys() :", self.data.keys())
        
        # self.compute_accuracy(self.data)
                
        # ================================
        #  Creating Out-Of-Distribution Sets     
        # ================================
        print("\n\n  ##  Creating Out-Of-Distribution Sets  ##  ", flush=True)
        if feature_model == "mahala":
            print("OOD Data Collection For Mahala:")
            if not mahala_xood:
            # if mahala_xood:

                self.ood = {name: self.conf.add_prediction_and_penultimate_features_dl_to_mahala(
                    df) for name, df in out_of_dist(self.dataset).items()}
            else:
                self.ood = {name: self.conf.add_prediction_and_features_dl(
                    df) for name, df in out_of_dist(self.dataset).items()}
            
        elif feature_model == "knn":
            print("OOD Data Collection For KNN:")
            if knn_pen:
                self.ood = {name: self.conf.add_prediction_and_features_knn(
                    df) for name, df in out_of_dist(self.dataset).items()}
            else:
                self.ood = {name: self.conf.add_prediction_and_extreme_features_dl_to_knn(df) for name, df in out_of_dist(self.dataset).items()}
            
        else:
            for name, df in out_of_dist(self.dataset).items():
                mahala_ood = self.conf.add_prediction_and_features_dl(df)
                knn_ood = self.conf.add_prediction_and_features_knn(df)
                self.ood[name] = pd.concat([mahala_ood, knn_ood], ignore_index=True, axis=1)
            #self.ood = {name: self.conf.add_prediction_and_features_knn(
             #   df) for name, df in out_of_dist(self.dataset).items()}
            #self.ood = {name: self.conf.add_prediction_and_extreme_features_dl_to_knn(
             #   df) for name, df in out_of_dist(self.dataset).items()}
        print("Length of ood: ", self.ood.keys())
        # self.cal = None  # Training set for the logistic regression.

    def compute_accuracy(self, datasets):
        print("test_ood.py ==> FeatureTester.compute_accuracy()")
        try:
            accuracy = pd.read_csv(
                self.path / "accuracy.txt", sep=":", index_col=0)["Accuracy"]
        except FileNotFoundError:
            accuracy = pd.Series(name="Accuracy", dtype=float)
        for name, df in datasets.items():
            accuracy[name] = df["is_correct"].mean()
            print(f"Accuracy {name}: {accuracy[name]}")
        accuracy.sort_values(ascending=False).to_csv(
            self.path / "accuracy.txt", sep=":")
        print("Done", flush=True)

    def create_summary_combine(self, f, name="", corr=False):
        print("\n\ntest_ood.py ==> FeatureTester.create_summary_combine()")
        print("Creating Taylor Table", flush=True)
            
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        return pred, pred_clean

    def taylor_table(self, pred, pred_clean, name, method_name, corr=False):
        print("\n\n test_ood.py ==> FeatureTester.taylor_table()")

        all = np.concatenate(list(pred.values()) + [pred_clean])
        print("all :", all)
        p_min, p_max = np.min(all), np.max(all)

        # This function is used since some scores only support values between 0 and 1.
        def map_pred(x):
            print("test_ood.py ==> map_pred()")
            return (x - p_min) / (p_max - p_min)

        pred["All"] = np.concatenate(list(pred.values()))
        print("Until Taylor table everything is good")
        
        # ==========================
        # compute_taylor_scores
        # ==========================
        featuretester_method = name
        table = pd.DataFrame.from_dict(
            {name: taylor_scores(map_pred(pred_clean), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")
        
        table.to_csv(self.path / f"summary_{name}.csv")
        df_to_pdf(table, decimals=4, path=self.path /
                  f"summary_{name}.pdf", vmin=0, percent=True)
        # self.hist_plot(pred, pred_clean, method_name)
        print("taylor_table name 1.3", name)
        if corr:
            pred_corr = pred_clean[self.data["Test"]["is_correct"]]
            table = pd.DataFrame.from_dict(
                {name: taylor_scores(map_pred(pred_corr), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")
            table.to_csv(self.path / f"summary_correct_{name}.csv")
            df_to_pdf(table, decimals=4, path=self.path /
                      f"summary_correct_{name}.pdf", vmin=0, percent=True)
        
        # ========================================
        # get_indices of wrongly classified images
        # ========================================
        incorrect_indices_table = pd.DataFrame.from_dict(
            {name: get_incorrect_indices(map_pred(pred_clean), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")   
        
        print("flag 1.81 incorrect_indices_table :",incorrect_indices_table)
        
        # ==========================
        # compute_confusion_metrix scores
        # ==========================
        cm_table = pd.DataFrame.from_dict(
            {name: compute_confusion_metrix(map_pred(pred_clean), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")
        
        cm_table.to_csv(self.path / f"confusion_metrix_summary_{name}.csv")
        # df_to_pdf(cm_table, decimals=4, path=self.path /
        #           f"confusion_metrix_summary_{name}.pdf", vmin=0, percent=True)
        # self.hist_plot(pred, pred_clean, method_name)
        print("confusion_metrix_table name 1.28", name)
        
        return incorrect_indices_table
        
    def create_summary(self, f, name="", corr=False):
        print("test_ood.py ==> FeatureTester.create_summary()")
        print("Creating Taylor Table", flush=True)
        print(self.ood.keys())

        all = np.concatenate(list(pred.values()) + [pred_clean])
        print(all)
        p_min, p_max = np.min(all), np.max(all)

        # This function is used since some scores only support values between 0 and 1.
        def map_pred(x):
            print("test_ood.py ==> taylor_scores()")
            return (x - p_min) / (p_max - p_min)

        #pred["All"] = np.concatenate(list(pred.values()))
        print("Until Taylor table everything is good")
        table = pd.DataFrame.from_dict(
            {name: taylor_scores(map_pred(pred_clean), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")
        table.to_csv(self.path / f"summary_{name}.csv")
        df_to_pdf(table, decimals=4, path=self.path /
                  f"summary_{name}.pdf", vmin=0, percent=True)
        if corr:
            pred_corr = pred_clean[self.data["Test"]["is_correct"]]
            table = pd.DataFrame.from_dict(
                {name: taylor_scores(map_pred(pred_corr), map_pred(p),name,featuretester_method) for name, p in pred.items()}, orient="index")
            table.to_csv(self.path / f"summary_correct_{name}.csv")
            df_to_pdf(table, decimals=4, path=self.path /
                      f"summary_correct_{name}.pdf", vmin=0, percent=True)

    def test_separation(self, test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        print("test_ood.py ==> FeatureTester.test_separation()")
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(
                datasets.values()).reset_index(drop=True)
        summary_path = self.path / (f"{name}_split" if split else name)
        summary_path.mkdir(exist_ok=True, parents=True)
        summary = {dataset: {} for dataset in datasets.keys()}
        for feat in np.unique([c.split("_")[0] for c in self.conf.feat_cols]):
            feat_list = [f for f in self.conf.feat_cols if feat in f]
            if split & (feat != "Conf"):
                feat_list = list(
                    sorted([f + "-" for f in feat_list] + [f + "+" for f in feat_list]))
            fig, axs = plt.subplots(len(datasets), len(feat_list), squeeze=False,
                                    figsize=(2 * len(feat_list) + 3, 2.5 * len(datasets)), sharex="col")
            for i, (dataset_name, dataset) in enumerate(datasets.items()):
                if dataset_name != "Clean":
                    dataset = pd.concat([dataset, test_set]).reset_index()
                feats = pd.DataFrame(self.conf.pt.transform(
                    self.conf.scaler.transform(dataset[self.conf.feat_cols])), columns=self.conf.feat_cols)
                if split:
                    cols = list(feats.columns)
                    feats = pd.DataFrame(split_features(feats.to_numpy()),
                                         columns=[c + "+" for c in cols] + [c + "-" for c in cols])
                for j, feat_id in enumerate(feat_list):
                    summary[dataset_name][feat_id] = binary_class_hist(feats[feat_id], dataset["is_correct"],
                                                                       axs[i, j], "", bins=50,
                                                                       label_1="ID", label_0=dataset_name)
            for ax, col in zip(axs[0], feat_list):
                ax.set_title(f"Layer {col}")

            for ax, row in zip(axs[:, 0], datasets.keys()):
                ax.set_ylabel(row, size='large')
            plt.tight_layout(pad=.4)
            plt.savefig(summary_path / f"{feat}.pdf")
        if split:
            summary["LogReg Coeff"] = self.conf.coeff
        # save_corr_table(feature_table, self.path / f"corr_distorted", self.dataset_name)
        summary = pd.DataFrame(summary)
        summary.to_csv(f"{summary_path}.csv")
        df_to_pdf(summary, decimals=4,
                  path=f"{summary_path}.pdf", vmin=0, percent=True)

    def fit_knn(self, test: bool, c=None):
        print("test_ood.py ==> FeatureTester.fit_knn()")
        self.conf.fit_knn_faiss(self.data["Train"], c=c)

    def fit(self, c=None, new_cal_set=False):
        print("test_ood.py ==> FeatureTester.fit()")
        if new_cal_set or not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])
        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)

    def test_ood(self, split=False):
        print("test_ood.py ==> FeatureTester.test_ood()")
        print("\n==================   Testing features on Out-Of-Distribution Data   ==================\n",
              flush=True)
        self.test_separation(self.data["Test"].assign(
            is_correct=True), self.ood, "out_of_distribution", split)

    def test_distorted(self, split=False):
        print("test_ood.py ==> FeatureTester.test_distorted()")
        print("\n=====================   Testing features on Distorted Data   =====================\n", flush=True)
        dist = distorted(self.data["Test"])
        dist = {name: self.conf.add_prediction_and_features(
            df) for name, df in dist.items()}
        self.compute_accuracy(dist)
        self.test_separation(self.data["Test"], dist, "distorted", split)

    def plot_detection(self, f, name):
        print("test_ood.py ==> FeatureTester.plot_detection()")
        path = self.path / f"detection/{name}"
        path.mkdir(exist_ok=True, parents=True)
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        plt.figure(figsize=(4, 3))
        for key, p in pred.items():
            plt.clf()
            labels = pd.Series(np.concatenate(
                [np.ones(len(pred_clean), dtype=bool), np.zeros(len(p), dtype=bool)]))
            p = pd.Series(np.concatenate([pred_clean, p]))
            binary_class_hist(p, labels, plt.gca(), name,
                              label_0="OOD", label_1="ID")
            plt.tight_layout()
            plt.savefig(path / f"{key}.pdf")
            
    def hist_plot(self, pred_ood, pred_clean, method_name):
        if isinstance(pred_ood, dict):
          ood_df = {name: pd.DataFrame(-df, columns=[name]) for name, df in pred_ood.items()}
          print("this passes")
          in_df = pd.DataFrame(-pred_clean, columns=["clean"])
          result = {name: pd.concat([df, in_df]) for name, df in ood_df.items()}
          plt2.figure(figsize=(4, 3))
          for key, value in result.items():
              plt2.clf()
              sns.histplot(data=result[key])
              plt2.savefig(self.path / f"save_histogram_{method_name}_{str(key)}.png")
              plt2.clf()
              

        return result

def log_probability_original(pred_mahala, pred_knn, n):
    print("test_ood.py ==> log_probability()")
    if isinstance(pred_mahala, dict):
        result = {name: ((-n * (np.log(-(pred_knn[name])))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((-n * (np.log(-(pred_knn)))) - (pred_mahala ** 2))
    return result

def normalized_log_probability_original(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std, n):
    if isinstance(pred_mahala, dict):
        pred_mahala_result = {name: - (pred_mahala[name] ** 2) for name, df in pred_mahala.items()}
        pred_knn_result = {name: (-n * np.log(-pred_knn[name])) for name, df in pred_mahala.items()}

        result = {name: (((pred_mahala_result[name] - mahala_mean) / mahala_std) 
            + (pred_knn_result[name] - knn_mean) / knn_std ) for name, df in pred_mahala.items()}
    else:
        pred_knn_result = -n * np.log(-pred_knn)
        pred_mahala_result = - (pred_mahala **2)

        result = ((pred_mahala_result - mahala_mean) / mahala_std + (pred_knn_result - knn_mean) / knn_std)
    return result

def square_log_probability_original(pred_mahala, pred_knn, n):
    print("test_ood.py ==> square_log_probability()")
    if isinstance(pred_mahala, dict):
        result = {name: ((- math.sqrt(n) * (np.log(-(pred_knn[name])))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((- (math.sqrt(n)) * (np.log(-(pred_knn)))) - (pred_mahala ** 2))
    return result

def log_probability(pred_mahala, pred_knn, n):
    print("test_ood.py ==> log_probability()")
    if isinstance(pred_mahala, dict):
        result = {name: ((-n * (np.log(-(pred_knn[name]), where = -(pred_knn[name]) > 0.0))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((-n * (np.log(-(pred_knn), where = -(pred_knn) > 0.0))) - (pred_mahala ** 2))
    return result

def normalized_log_probability(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std, n):
    if isinstance(pred_mahala, dict):
        pred_mahala_result = {name: - (pred_mahala[name] ** 2) for name, df in pred_mahala.items()}
        pred_knn_result = {name: (-n * (np.log(-(pred_knn[name]), where = -(pred_knn[name]) > 0.0))) for name, df in pred_mahala.items()}
        result = {name: (((pred_mahala_result[name] - mahala_mean) / mahala_std) 
            + (pred_knn_result[name] - knn_mean) / knn_std ) for name, df in pred_mahala.items()}
    else:
        pred_knn_result = (-n * (np.log(-(pred_knn), where = -(pred_knn) > 0.0)))
        pred_mahala_result = - (pred_mahala **2)

        result = ((pred_mahala_result - mahala_mean) / mahala_std + (pred_knn_result - knn_mean) / knn_std)
        
    return result

def square_log_probability(pred_mahala, pred_knn, n):
    print("test_ood.py ==> square_log_probability()")
    if isinstance(pred_mahala, dict):
        result = {name: (((- math.sqrt(n) * (np.log(-(pred_knn[name]), where = -(pred_knn[name]) > 0.0)))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((- (math.sqrt(n)) * (np.log(-(pred_knn), where = -(pred_knn) > 0.0))) - (pred_mahala ** 2))
    return result

def weighted_geometric_mean(pred_mahala, pred_knn, alpha):
    print("test_ood.py ==> weighted_geometric_mean()")
    if isinstance(pred_mahala, dict):
        result = {name: -(((-df) ** alpha) * ((-pred_knn[name]) ** (
            1 - alpha))) for name, df in pred_mahala.items()}
    else:
        result = -(((-pred_mahala) ** alpha) * ((-pred_knn) ** (1-alpha)))
    return result

def weighted_arthmetic_mean(pred_mahala, pred_knn, mahala_mean, knn_mean, alpha):
    print("test_ood.py ==> weighted_arthmetic_mean()")
    #mean_mahala = {name: df.mean() for name, df in pred_mahala.items()}
    #mean_knn = {name: df.mean() for name, df in pred_knn.items()}
    #norm_pred_mahala = {name: df - mean_mahala[name] for name, df in pred_mahala.items()}
    #norm_pred_knn = {name: df - mean_knn[name] for name, df in pred_knn.items()}
    if isinstance(pred_mahala, dict):
        result = {name: -(alpha * (-df/mahala_mean) + (1-alpha) * (-pred_knn[name]/knn_mean)) for name, df in pred_mahala.items()}
    else:
        result = -((alpha) * (-pred_mahala/mahala_mean) + (1-alpha) * (-pred_knn/knn_mean))
    return result

def max_distance(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std):
    n = 512
    if isinstance(pred_mahala, dict):
        
        result = {name: np.maximum((((pred_mahala[name]) - mahala_mean) / mahala_std), 
            (((pred_knn[name]) - knn_mean) / knn_std)) for name, df in pred_mahala.items()}
    else:
        result = np.maximum((((pred_mahala) - mahala_mean) / mahala_std), 
            (((pred_knn) - knn_mean) / knn_std))
    return result

def max_distance2(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std):
    n = 512
    if isinstance(pred_mahala, dict):
        
        result = {name: np.maximum((np.abs(pred_mahala[name] - mahala_mean) / mahala_std), 
            (np.abs(pred_knn[name] - knn_mean) / knn_std)) for name, df in pred_mahala.items()}
    else:
        result = np.maximum((np.abs(pred_mahala - mahala_mean) / mahala_std), 
            (np.abs(pred_knn - knn_mean) / knn_std))
    return result

def hist_plot_mahala_knn(pred_mahala, pred_knn, method_name):
    if isinstance(pred_mahala, dict):
      mahala_df = {name: pd.DataFrame(-df, columns=["mahala_" + name]) for name, df in pred_mahala.items()}
      knn_df = {name: pd.DataFrame(-df, columns=["knn_" + name]) for name, df in pred_knn.items()}
      result = {name: pd.concat([mahala_df[name], knn_df[name]]) for name, df in pred_knn.items()}
      for key, value in result.items():
        sns.histplot(data=result[key])
        plt.savefig(f"save_histogram_{method_name}_{str(key)}.png")
    return result

def pearson_coefficient(a, b):
    return stats.pearsonr(a, b)


def test_ood(dataset, model, alpha):
    print("test_ood.py ==> test_ood()")
    print(
        f"\n\n================ Testing Features On {dataset} {model} ================", flush=True)
    pred_probs = []
    pred_clean_probs = []
    #ft.create_summary(ft.conf.predict_mahala, "x-ood-mahala")

    ## ft_mahala -> this will be in mahala
    ## FeatureTester__init__(self, dataset: str, model: str, feature_model, folder_name=""
    
    print("\n\n==> a) Calculating Mahala on Extreme values..")
    ft_mahala_xood = FeatureTester(dataset, model, "mahala", "knn", extreme=True, pen=False)
    pred_mahala_xood, pred_clean_mahala_xood = ft_mahala_xood.create_summary_combine(
        ft_mahala_xood.conf.predict_mahala, "x-ood-mahala")
    incorrect_indices_table_mahala_xtreme=ft_mahala_xood.taylor_table(pred_mahala_xood, pred_clean_mahala_xood,
                            "x-ood-mahala-extreme-" + str(alpha), "mahala")


    print("\n\n==> b) Calculating KNN on Penultimate layer values..")
    ft_knn_pen = FeatureTester(dataset, model, "knn", "knn", extreme=False, pen=True)
    ft_knn_pen.fit_knn(test=False)
    pred_knn_pen, pred_clean_knn_pen = ft_knn_pen.create_summary_combine(
        ft_knn_pen.conf.predict_knn_faiss, "open-ood-knn")
    incorrect_indices_table_knn_pen = ft_knn_pen.taylor_table(pred_knn_pen, pred_clean_knn_pen, "knn-penultimate-features-" + str(alpha), "knn")

    # ================  %%  ====================
    #     uncomment the following line to get the instances of other two methods
    # ================= %%  ====================
    
    
    # print("\n\n==> c) Calculating Mahala on Penultimate layer values..")
    # ft_mahala_pen = FeatureTester(dataset, model, "mahala", "knn", extreme=False, pen=True)
    # pred_mahala_pen, pred_clean_mahala_pen = ft_mahala_pen.create_summary_combine(
    #     ft_mahala_pen.conf.predict_mahala, "x-ood-mahala")
    # ft_mahala_pen.taylor_table(pred_mahala_pen, pred_clean_mahala_pen,
    #                         "mahala-penultimate-" + str(alpha), "mahala")


    # print("\n\n==> d) Calculating KNN on Extreme values..")
    # ft_knn_xood = FeatureTester(dataset, model, "knn", "knn", extreme=True, pen=False)
    # ft_knn_xood.fit_knn(test=False)
    # pred_knn_xood, pred_clean_knn_xood = ft_knn_xood.create_summary_combine(
    #     ft_knn_xood.conf.predict_knn_faiss, "open-ood-knn")
    # ft_knn_xood.taylor_table(pred_knn_xood, pred_clean_knn_xood, "knn-extreme-features-" + str(alpha), "knn")
    
    # ================= %%  ====================
    
    # # if (np.isnan(pred_knn)== True):
    # # if (pd.isna(pred_knn)== True):
        
    # #     ft_knn.taylor_table(pred_knn, pred_clean_knn, "knn-penultimate-features-" + str(alpha), "knn")
    # # else:
    # #     print("flag It is failing ")

    # # hist_plot_mahala_knn(pred_mahala,pred_knn,"mahala_knn")
    
    # ==========================================
    #     
    # ==========================================
    
    # # weighted_arthmetic_mean
    # pred_arth = weighted_arthmetic_mean(pred_mahala, pred_knn, ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, alpha)
    # print("pred_arth Keys:", pred_arth.keys())
    # pred_clean_arth = weighted_arthmetic_mean(pred_clean_mahala, pred_clean_knn, ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, alpha)
    # ft_knn.taylor_table(pred_arth, pred_clean_arth, "x-ood-mahala-knn-arth-" + str(alpha),"arthmetic_mean")

    # # weighted_geometric_mean
    # pred_geo = weighted_geometric_mean(pred_mahala, pred_knn, alpha)
    # pred_clean_geo = weighted_geometric_mean(
    #     pred_clean_mahala, pred_clean_knn, alpha)
    # ft_knn.taylor_table(pred_geo, pred_clean_geo, "x-ood-mahala-knn-geo-" + str(alpha),"geometric_mean")
   
 
    # # print(f" pred_mahala 2.51 : {pred_mahala}")
    # # print(f" pred_knn 2.51: {pred_knn}")
    
    # # print(f" pred_clean_mahala 2.52: {pred_clean_mahala}")
    # # print(f" pred_clean_knn 2.52: {pred_clean_knn}")
    
    
    # ==============  =  ================
    #  save pickles    
    # ==============  =  ================
    # with open('pred_mahala_xood_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_mahala_xood, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('pred_knn_pen_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_knn_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # with open('pred_clean_mahala_xood_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_clean_mahala_xood, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('pred_clean_knn_pen_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_clean_knn_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # print(f" pred_mahala 2.51 : {pred_mahala}")
    # # print(f" pred_knn 2.51: {pred_knn}")
    
    # # print(f" pred_clean_mahala 2.52: {pred_clean_mahala}")
    # # print(f" pred_clean_knn 2.52: {pred_clean_knn}")
    
    # with open('pred_mahala_pen_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_mahala_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('pred_knn_xood_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_knn_xood, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # with open('pred_clean_mahala_pen_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_clean_mahala_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # with open('pred_clean_knn_xood_'+str(dataset)+'_.pickle', 'wb') as handle:
    #     pickle.dump(pred_clean_knn_xood, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # ==============  =  ================
    #  save pickles done   
    # ==============  =  ================            


    # ===============================  ##  ==============================================
    # combination of knn and mahala with extreme - penultimate features
    # uncomment the following lines to get results for different combinations 
    # ===============================  ##  ==============================================
    
    # Mahala xood + KNN pen log probabilty
    # pred_log_m_xood_knn_pen = log_probability(pred_mahala_xood, pred_knn_pen, ft_knn_pen.conf.knn_n)
    # pred_clean_log_m_xood_knn_pen = log_probability(pred_clean_mahala_xood, pred_clean_knn_pen,ft_knn_pen.conf.knn_n)
    # ft_knn_pen.taylor_table(pred_log_m_xood_knn_pen, pred_clean_log_m_xood_knn_pen, "xood-mahala-pen-knn-log", "log_probability" )

    # # Mahala Pen + KNN pen log probabilty
    # pred_log_m_pen_knn_pen = log_probability(pred_mahala_pen, pred_knn_pen, ft_knn_pen.conf.knn_n)
    # pred_clean_log_m_pen_knn_pen = log_probability(pred_clean_mahala_pen, pred_clean_knn_pen,ft_knn_pen.conf.knn_n)
    # ft_knn_pen.taylor_table(pred_log_m_pen_knn_pen, pred_clean_log_m_pen_knn_pen, "pen-mahala-pen-knn-log", "log_probability" )

    # # Mahala xood + KNN Xood log probabilty
    # pred_log_m_xood_knn_xood = log_probability(pred_mahala_xood, pred_knn_xood, ft_knn_xood.conf.knn_n)
    # pred_clean_log_m_xood_knn_xood = log_probability(pred_clean_mahala_xood, pred_clean_knn_xood,ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_log_m_xood_knn_xood, pred_clean_log_m_xood_knn_xood, "xood-mahala-xood-knn-log", "log_probability" )

    # # Mahala pen + KNN xood log probabilty
    # pred_log_m_pen_knn_xood = log_probability(pred_mahala_pen, pred_knn_xood, ft_knn_xood.conf.knn_n)
    # pred_clean_log_m_pen_knn_xood = log_probability(pred_clean_mahala_pen, pred_clean_knn_xood,ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_log_m_pen_knn_xood, pred_clean_log_m_pen_knn_xood, "pen-mahala-xood-knn-log", "log_probability" )
    
    ### == ###
    # Mahala xood + KNN pen square log probabilty  
    pred_sq_log_m_xood_knn_pen = square_log_probability(pred_mahala_xood, pred_knn_pen, ft_knn_pen.conf.knn_n)
    t3 = time.time()
    pred_clean_sq_log_m_xood_knn_pen = square_log_probability(pred_clean_mahala_xood, pred_clean_knn_pen, ft_knn_pen.conf.knn_n)
    t4 = time.time()
    elapsed_time2 = t4 -t3
    print("flag 2.007 Elapsed time:", elapsed_time2, "seconds")
    incorrect_indices_table_xood_mahala_pen_knn_log_sq = ft_knn_pen.taylor_table(pred_sq_log_m_xood_knn_pen, pred_clean_sq_log_m_xood_knn_pen, "xood-mahala-pen-knn-log-sq", "square_log_probability")
    ### == ###
    
    # # Mahala Pen + KNN pen square log probabilty  
    # pred_sq_log_m_pen_knn_pen = square_log_probability(pred_mahala_pen, pred_knn_pen, ft_knn_pen.conf.knn_n)
    # pred_clean_sq_log_m_pen_knn_pen = square_log_probability(pred_clean_mahala_pen, pred_clean_knn_pen, ft_knn_pen.conf.knn_n)
    # ft_knn_pen.taylor_table(pred_sq_log_m_pen_knn_pen, pred_clean_sq_log_m_pen_knn_pen, "pen-mahala-pen-knn-log-sq", "square_log_probability")

    # # Mahala xood + KNN Xood square log probabilty  
    # pred_sq_log_m_xood_knn_xood = square_log_probability(pred_mahala_xood, pred_knn_xood, ft_knn_xood.conf.knn_n)
    # pred_clean_sq_log_m_xood_knn_xood = square_log_probability(pred_clean_mahala_xood, pred_clean_knn_xood, ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_sq_log_m_xood_knn_xood, pred_clean_sq_log_m_xood_knn_xood, "xood-mahala-xood-knn-log-sq", "square_log_probability")

    # # Mahala pen + KNN xood square log probabilty  
    # pred_sq_log_m_pen_knn_xood = square_log_probability(pred_mahala_pen, pred_knn_xood, ft_knn_xood.conf.knn_n)
    # pred_clean_sq_log_m_pen_knn_xood = square_log_probability(pred_clean_mahala_pen, pred_clean_knn_xood, ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_sq_log_m_pen_knn_xood, pred_clean_sq_log_m_pen_knn_xood, "pen-mahala-xood-knn-log-sq", "square_log_probability")


    # Mahala xood + KNN pen normalized_log_probability
    # pred_n_log_m_xood_knn_pen = normalized_log_probability(pred_mahala_xood, pred_knn_pen,
    #         ft_mahala_xood.conf.mahala_mean, ft_knn_pen.conf.knn_mean, ft_mahala_xood.conf.mahala_std, ft_knn_pen.conf.knn_std, ft_knn_pen.conf.knn_n)
    # pred_n_clean_log_m_xood_knn_pen = normalized_log_probability(pred_clean_mahala_xood, pred_clean_knn_pen, 
    #         ft_mahala_xood.conf.mahala_mean, ft_knn_pen.conf.knn_mean, ft_mahala_xood.conf.mahala_std, ft_knn_pen.conf.knn_std, ft_knn_pen.conf.knn_n)
    # ft_knn_pen.taylor_table(pred_n_log_m_xood_knn_pen, pred_n_clean_log_m_xood_knn_pen, "xood-mahala-pen-knn-n-log","normalized_log_probability")
    
    # # Mahala Pen + KNN pen normalized_log_probability
    # pred_n_log_m_pen_knn_pen = normalized_log_probability(pred_mahala_pen, pred_knn_pen,
    #         ft_mahala_pen.conf.mahala_mean, ft_knn_pen.conf.knn_mean, ft_mahala_pen.conf.mahala_std, ft_knn_pen.conf.knn_std, ft_knn_pen.conf.knn_n)
    # pred_n_clean_log_m_pen_knn_pen = normalized_log_probability(pred_clean_mahala_pen, pred_clean_knn_pen, 
    #         ft_mahala_pen.conf.mahala_mean, ft_knn_pen.conf.knn_mean, ft_mahala_pen.conf.mahala_std, ft_knn_pen.conf.knn_std, ft_knn_pen.conf.knn_n)
    # ft_knn_pen.taylor_table(pred_n_log_m_pen_knn_pen, pred_n_clean_log_m_pen_knn_pen, "pen-mahala-pen-knn-n-log","normalized_log_probability")
    
    # # Mahala xood + KNN Xood normalized_log_probability
    # pred_n_log_m_xood_knn_xood = normalized_log_probability(pred_mahala_xood, pred_knn_xood,
    #         ft_mahala_xood.conf.mahala_mean, ft_knn_xood.conf.knn_mean, ft_mahala_xood.conf.mahala_std, ft_knn_xood.conf.knn_std, ft_knn_xood.conf.knn_n)
    # pred_n_clean_log_m_xood_knn_xood = normalized_log_probability(pred_clean_mahala_xood, pred_clean_knn_xood, 
    #         ft_mahala_xood.conf.mahala_mean, ft_knn_xood.conf.knn_mean, ft_mahala_xood.conf.mahala_std, ft_knn_xood.conf.knn_std, ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_n_log_m_xood_knn_xood, pred_n_clean_log_m_xood_knn_xood, "xood-mahala-xood-knn-n-log","normalized_log_probability")
    
    # # Mahala Pen + KNN xood normalized_log_probability
    # pred_n_log_m_pen_knn_xood = normalized_log_probability(pred_mahala_pen, pred_knn_xood,
    #         ft_mahala_pen.conf.mahala_mean, ft_knn_xood.conf.knn_mean, ft_mahala_pen.conf.mahala_std, ft_knn_xood.conf.knn_std, ft_knn_xood.conf.knn_n)
    # pred_n_clean_log_m_pen_knn_xood = normalized_log_probability(pred_clean_mahala_pen, pred_clean_knn_xood, 
    #         ft_mahala_pen.conf.mahala_mean, ft_knn_xood.conf.knn_mean, ft_mahala_pen.conf.mahala_std, ft_knn_xood.conf.knn_std, ft_knn_xood.conf.knn_n)
    # ft_knn_xood.taylor_table(pred_n_log_m_pen_knn_xood, pred_n_clean_log_m_pen_knn_xood, "pen-mahala-xood-knn-n-log","normalized_log_probability")
    
    # ===============================  ##  ==============================================
    
    
    '''
    # =============================================================================
    #     saving incorrect_indices_table as pickle file
    # =============================================================================
    with open('incorrect_indices_table_knn_pen'+"_"+str(dataset)+'_.pickle', 'wb') as handle:
        pickle.dump(incorrect_indices_table_knn_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('incorrect_indices_table_mahala_xtreme'+"_"+str(dataset)+'_.pickle', 'wb') as handle:
        pickle.dump(incorrect_indices_table_mahala_xtreme, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('incorrect_indices_table_xood_mahala_pen_knn_log_sq'+"_"+str(dataset)+'_.pickle', 'wb') as handle:
        pickle.dump(incorrect_indices_table_xood_mahala_pen_knn_log_sq, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # =============================================================================
    #     save_missing_indices_ of cifar10 (OOD) images_in_folder (ID- MNIST)
    # =============================================================================
    
    if (dataset== 'mnist'):
        incorrect_indices_knn_cifar10 = incorrect_indices_table_knn_pen.loc['Cifar10', 'incorrect_indices']
        incorrect_indices_mahala_cifar10 = incorrect_indices_table_mahala_xtreme.loc['Cifar10', 'incorrect_indices']
        incorrect_indices_xood_mahala_pen_knn_log_sq_cifar10 = incorrect_indices_table_xood_mahala_pen_knn_log_sq.loc['Cifar10', 'incorrect_indices']
    
        print("len(incorrect_indices_knn_cifar10)  :",len(incorrect_indices_knn_cifar10))
        print("len(incorrect_indices_mahala_cifar10)  :",len(incorrect_indices_mahala_cifar10))
        print("len(incorrect_indices_xood_mahala_pen_knn_log_sq_cifar10)  :",len(incorrect_indices_xood_mahala_pen_knn_log_sq_cifar10))
     
        
        knn= set(incorrect_indices_knn_cifar10)    # len(list(knn)) = 581
        comb_sq_log = set(incorrect_indices_xood_mahala_pen_knn_log_sq_cifar10)  # len(list(knn)) = 80    581
        common_indices = knn.intersection(comb_sq_log)  # len(list(common_indices))=49
        
        # we are removing common indices which are wrongly classified by both knn and comb_log_sq
        knn_only = knn- common_indices  # len(list(knn_only)) =532
        comb_sq_log_only = comb_sq_log- common_indices # len(list(comb_sq_log_only)) =31
        
        missing_indices = knn_only - comb_sq_log_only
        save_missing_cifar10_indices_images_in_folder_for_mnist_id(list(missing_indices),"missing_indices_ood_cifar10__id_mnist", dataset )
        
    
    # =============================================================================
    #     save_missing_indices_ of document images_in_folder 
    # =============================================================================
    if (dataset== 'document'):
        
        incorrect_indices_knn_document = incorrect_indices_table_knn_pen.loc['Rvl_Cdip_O', 'incorrect_indices']
        incorrect_indices_mahala_document = incorrect_indices_table_mahala_xtreme.loc['Rvl_Cdip_O', 'incorrect_indices']
        incorrect_indices_xood_mahala_pen_knn_log_sq_document = incorrect_indices_table_xood_mahala_pen_knn_log_sq.loc['Rvl_Cdip_O', 'incorrect_indices']
    
        print("flag 1.32 len(incorrect_indices_knn_document)  :",len(incorrect_indices_knn_document))
        print("flag 1.32 len(incorrect_indices_mahala_document)  :",len(incorrect_indices_mahala_document))
        print("flag 1.32 len(incorrect_indices_xood_mahala_pen_knn_log_sq_document)  :",len(incorrect_indices_xood_mahala_pen_knn_log_sq_document))
     
        knn= set(incorrect_indices_knn_document)    # len(list(knn)) = 
        comb_sq_log = set(incorrect_indices_xood_mahala_pen_knn_log_sq_document)  # len(list(knn)) = 
        common_indices = knn.intersection(comb_sq_log)  # len(list(common_indices))=
        
        # we are removing common indices which are wrongly classified by both knn and comb_log_sq
        knn_only = knn- common_indices  # len(list(knn_only)) =79
        comb_sq_log_only = comb_sq_log- common_indices # len(list(comb_sq_log_only)) = 39
        
        print("flag 1.32 len(knn_only)  :",len(knn_only))  
        print("flag 1.32 len(comb_sq_log_only)  :",len(comb_sq_log_only))
        
        # indices_failed_by_knn =A = {1, 3, 7, 8, 13, 35}
        # indices_failed_by_comb_sq_log =B = {8, 13, 17}
        # indices_corrected_by_comb_sq_log  = A - B
        # print(result)
        
        # missing indices -> indices corrected by com_sq_log method  but failed by knn
        missing_indices = knn_only - comb_sq_log_only    
        save_missing_document_indices_images_in_folder(list(missing_indices),"missing_indices_ood_rvl_cdip_o__id_document_fulldataset" )
        
            # ====================================================================================
            #     saving y_pred values of incorrect_indices of document for each of the FeatureTester instance   
            # ====================================================================================
            
        incorrect_indices_table_knn_pen_rvl_cdip_o =incorrect_indices_table_knn_pen.loc['Rvl_Cdip_O']
        y_pred_list_knn_pen_rvl_cdip_o = list(incorrect_indices_table_knn_pen_rvl_cdip_o.loc['y_pred_ood'])
            
        incorrect_indices_table_xood_mahala_pen_knn_log_sq_rvl_cdip_o =incorrect_indices_table_xood_mahala_pen_knn_log_sq.loc['Rvl_Cdip_O']
        y_pred_list_xood_mahala_pen_knn_log_sq_rvl_cdip_o = list(incorrect_indices_table_xood_mahala_pen_knn_log_sq_rvl_cdip_o.loc['y_pred_ood'])
        
        incorrect_indices_table_mahala_xtreme_rvl_cdip_o =incorrect_indices_table_mahala_xtreme.loc['Rvl_Cdip_O']
        y_pred_list_mahala_xtreme_rvl_cdip_o = list(incorrect_indices_table_mahala_xtreme_rvl_cdip_o.loc['y_pred_ood'])
        
        index_list= list(missing_indices)
        
        y_pred_knn_pen_rvl_cdip_o = [y_pred_list_knn_pen_rvl_cdip_o[i] for i in index_list]
        y_pred_xood_mahala_pen_knn_log_sq_rvl_cdip_o = [y_pred_list_xood_mahala_pen_knn_log_sq_rvl_cdip_o[i] for i in index_list]
        y_pred_mahala_xtreme_rvl_cdip_o = [y_pred_list_mahala_xtreme_rvl_cdip_o[i] for i in index_list]
        
        data = {
            'index_list': index_list,
            'y_pred_knn_pen_rvl_cdip_o': y_pred_knn_pen_rvl_cdip_o,
            'y_pred_xood_mahala_pen_knn_log_sq_rvl_cdip_o': y_pred_xood_mahala_pen_knn_log_sq_rvl_cdip_o,
            'y_pred_mahala_xtreme_rvl_cdip_o': y_pred_mahala_xtreme_rvl_cdip_o
        }
        df = pd.DataFrame(data)
        df.to_csv('y_pred_values_of_missing_indices_ood_rvl_cdip_o_id_document.csv', index=True)
        
    
    
    # =============================================================================
    #     save_missing_indices_ of imagenet images_in_folder 
    # =============================================================================
    if (dataset== 'imagenet'):
        
        incorrect_indices_knn_imagenet = incorrect_indices_table_knn_pen.loc['Species', 'incorrect_indices']
        incorrect_indices_mahala_imagenet = incorrect_indices_table_mahala_xtreme.loc['Species', 'incorrect_indices']
        incorrect_indices_xood_mahala_pen_knn_log_sq_imagenet = incorrect_indices_table_xood_mahala_pen_knn_log_sq.loc['Species', 'incorrect_indices']
    
        print("len(incorrect_indices_knn_imagenet)  :",len(incorrect_indices_knn_imagenet))
        print("len(incorrect_indices_mahala_imagenet)  :",len(incorrect_indices_mahala_imagenet))
        print("len(incorrect_indices_xood_mahala_pen_knn_log_sq_imagenet)  :",len(incorrect_indices_xood_mahala_pen_knn_log_sq_imagenet))
     
        knn= set(incorrect_indices_knn_imagenet)    # len(list(knn)) = 
        comb_sq_log = set(incorrect_indices_xood_mahala_pen_knn_log_sq_imagenet)  # len(list(knn)) = 
        common_indices = knn.intersection(comb_sq_log)  # len(list(common_indices))=
        
        # we are removing common indices which are wrongly classified by both knn and comb_log_sq
        knn_only = knn- common_indices  # len(list(knn_only)) =79
        comb_sq_log_only = comb_sq_log- common_indices # len(list(comb_sq_log_only)) = 39
        
        print("flag 1.32 len(knn_only)  :",len(knn_only))  
        print("flag 1.32 len(comb_sq_log_only)  :",len(comb_sq_log_only))
        
        missing_indices = knn_only - comb_sq_log_only
        print("flag 1.33 missing_indices_list  :",list(missing_indices))
        save_missing_indices_images_in_folder(list(missing_indices),"missing_indices_ood_species__id_imagenet", dataset )
        
            # ====================================================================================
            #     saving y_pred values of incorrect_indices of document for each of the FeatureTester instance   
            # ====================================================================================
            
        incorrect_indices_table_knn_pen_imagenet =incorrect_indices_table_knn_pen.loc['Species']
        y_pred_list_knn_pen_imagenet = list(incorrect_indices_table_knn_pen_imagenet.loc['y_pred_ood'])
            
        incorrect_indices_table_xood_mahala_pen_knn_log_sq_imagenet =incorrect_indices_table_xood_mahala_pen_knn_log_sq.loc['Species']
        y_pred_list_xood_mahala_pen_knn_log_sq_imagenet = list(incorrect_indices_table_xood_mahala_pen_knn_log_sq_imagenet.loc['y_pred_ood'])
        
        incorrect_indices_table_mahala_xtreme_imagenet =incorrect_indices_table_mahala_xtreme.loc['Species']
        y_pred_list_mahala_xtreme_imagenet = list(incorrect_indices_table_mahala_xtreme_imagenet.loc['y_pred_ood'])
        
        index_list= list(missing_indices)
        
        y_pred_knn_pen_imagenet = [y_pred_list_knn_pen_imagenet[i] for i in index_list]
        y_pred_xood_mahala_pen_knn_log_sq_imagenet = [y_pred_list_xood_mahala_pen_knn_log_sq_imagenet[i] for i in index_list]
        y_pred_mahala_xtreme_imagenet = [y_pred_list_mahala_xtreme_imagenet[i] for i in index_list]
        
        data = {
            'index_list': index_list,
            'y_pred_knn_pen_imagenet': y_pred_knn_pen_imagenet,
            'y_pred_xood_mahala_pen_knn_log_sq_imagenet': y_pred_xood_mahala_pen_knn_log_sq_imagenet,
            'y_pred_mahala_xtreme_imagenet': y_pred_mahala_xtreme_imagenet
        }
        df = pd.DataFrame(data)
        df.to_csv('y_pred_values_of_missing_indices_ood_species__id_imagenet.csv', index=True)
        
    '''
    
    # # =============================================================================
    # #   get_mcnemar_test for_all_ood_data
    # # =============================================================================
    # with open('incorrect_indices_table_knn_pen.pickle', 'wb') as handle:
    #     pickle.dump(incorrect_indices_table_knn_pen, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('incorrect_indices_table_xood_mahala_pen_knn_log_sq.pickle', 'wb') as handle:
    #     pickle.dump(incorrect_indices_table_xood_mahala_pen_knn_log_sq, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # get_mcnemar_for_all_ood_data(dataset,incorrect_indices_table_knn_pen, incorrect_indices_table_xood_mahala_pen_knn_log_sq)
    # print("flag 1.3 finished")

    # print("1.32 saving done")

    
    # =============================================================================
    #     Pearson Coefficient
    # =============================================================================
    # if isinstance(pred_mahala_xood, dict):
    #     for key in pred_mahala_xood.keys():
    #         print("Dataset for Mahala Extreme Values: ", key)
    #         print("Mahala Xood + Knn Pen : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + KNN pen : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_log_m_pen_knn_pen[key]))
    #         print("Mahala xood + KNN Xood : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + KNN xood : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_log_m_pen_knn_xood[key]))
    #         print("Mahala Xood + Knn Pen : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_sq_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + Knn Pen : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_sq_log_m_pen_knn_pen[key]))
    #         print("Mahala Xood + Knn Xood : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_sq_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + Knn Xood : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_sq_log_m_pen_knn_xood[key]))
    #         print("Mahala Xood + Knn Pen : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_n_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + Knn Pen : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_n_log_m_pen_knn_pen[key]))
    #         print("Mahala Xood + Knn Xood : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_n_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + Knn Xood : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_mahala_xood[key], pred_n_log_m_pen_knn_xood[key]))
    #         print("<<<<<<<<==================================================================================>>>>>>>")
    

    # if isinstance(pred_knn_pen, dict):
    #     for key in pred_knn_pen.keys():
    #         print("Dataset for KNN Penultimate Layer: ", key)
    #         print("Mahala Xood + Knn Pen : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + KNN pen : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_log_m_pen_knn_pen[key]))
    #         print("Mahala xood + KNN Xood : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + KNN xood : Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_log_m_pen_knn_xood[key]))
    #         print("Mahala Xood + Knn Pen : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_sq_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + Knn Pen : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_sq_log_m_pen_knn_pen[key]))
    #         print("Mahala Xood + Knn Xood : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_sq_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + Knn Xood : Sq Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_sq_log_m_pen_knn_xood[key]))
    #         print("Mahala Xood + Knn Pen : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_n_log_m_xood_knn_pen[key]))
    #         print("Mahala Pen + Knn Pen : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_n_log_m_pen_knn_pen[key]))
    #         print("Mahala Xood + Knn Xood : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_n_log_m_xood_knn_xood[key]))
    #         print("Mahala Pen + Knn Xood : N - Log Probability - Pearson Coefficient value: ", pearson_coefficient(pred_knn_pen[key], pred_n_log_m_pen_knn_xood[key]))
    #         print("<<<<<<<<==================================================================================>>>>>>>")


    # ft_mahala.create_summary_combine(ft_mahala.conf.softmax, "baseline")
    # ft.create_summary(ft.conf.energy, "energy")
    # ft.create_summary(ft.conf.react_energy, "react_energy")
    # for i in range(10):
    # ft.fit(new_cal_set=True)
    #ft.create_summary(ft.conf.predict_proba, f"x-ood-lr")
    # ft.fit_knn(test=False)
    #ft.create_summary(ft.conf.predict_knn, f"x-ood-knn")
    # ft.fit_knn_faiss()
    # Add KNN Faiss algorithm to this
    #ft.create_summary(ft.conf.predict_knn_faiss, f"knn-open-ood")
    # ft.test_distorted()
    # ft.test_ood()
    
    


if __name__ == "__main__":
    
    start_time = time.time()
    
    # sys.stdout = open("console_output_imagenet.txt", "w")
    # test_ood("mnist", "lenet", 0.5)
    test_ood("cifar10", "resnet", 0.5)

    # test_ood("cifar100", "resnet", 0.5)
    # test_ood("document", "resnet50_docu", 0.5)

    # test_ood("imagenet", "resnet50", 0.5)
    # for i in [0.7]:
    #   test_ood("imagenet", "resnet34", i)
    #   test_ood("cifar10", "resnet", i)
    #  test_ood("cifar100", "resnet", i)
    #test_ood("cifar100", "resnet", 0.7)
    #test_ood("cifar100", "resnet50")
    #test_ood("cifar100", "resnet101")
    # for m in "resnet", "densenet":
    # for m in "densenet":
    #   for d in "svhn", "cifar10", "cifar100":
    #      test_ood(d, m)
    # for m in "resnet18", "resnet34", "resnet50", "resnet101":
    #     test_ood("imagenet", m)
    
    print("\nExecution Complete") 
    time_taken = convert_seconds((time.time() - start_time))
    print("--- time taken :  %s ---" % time_taken)
    