
import numpy as np
import pandas as pd
from torch.linalg import vector_norm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from data import get_images_and_labels
from typing import Callable, Dict
from utils import get_torch_device
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.nn import functional as F
import faiss
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sys
import time


def normalizer(x): return x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class LpNorm:
    print("confidenciator.py  ==> LpNorm()")
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x):
        return {f"L{self.p}-norm": vector_norm(x, self.p, dim=tuple(range(1, len(x.shape))))}


class DynamicRange:
    print("confidenciator.py  ==> DynamicRange()")
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        x = x.view(x.shape[0], -1)
        return {"DynamicRange": torch.amax(torch.abs(x), dim=1) / (1e-15 + vector_norm(x, self.p, dim=1))}


class SplitDynamicRange:
    print("confidenciator.py  ==> SplitDynamicRange()")
    def __init__(self, p=1):
        self.p = p

    def __call__(self, x):
        def dr(y):
            y = y.view(x.shape[0], -1)
            return torch.amax(y, dim=1) / vector_norm(y, self.p, dim=1)

        return {"NegDynamicRange": dr(torch.relu(-x)),
                "PosDynamicRange": dr(torch.relu(x))}


class SplitLpNorm:
    print("confidenciator.py  ==> SplitLpNorm()")
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x):
        return {"NegLpNorm": vector_norm(torch.relu(-x), self.p, dim=tuple(range(1, len(x.shape)))),
                "PosLpNorm": vector_norm(torch.relu(x), self.p, dim=tuple(range(1, len(x.shape))))}


class Positivity:
    def __call__(self, x):
        print("confidenciator.py  ==> Positivity()")
        return {"Positivity": torch.mean((x > 0).float(), dim=tuple(range(1, len(x.shape))))}


class Sum:
    print("confidenciator.py  ==> Sum()")
    def __call__(self, x):
        return {"Sum": torch.sum(x, dim=tuple(range(1, len(x.shape))))}


class MinMax:
    print("confidenciator.py  ==> MinMax()")
    def __call__(self, x: torch.Tensor):
        amin, amax = torch.aminmax(x.view(x.shape[0], -1), dim=1)
        return {"Min": -amin, "Max": amax}


class Min:
    def __call__(self, x: torch.Tensor):
        amin, amax = torch.aminmax(x.view(x.shape[0], -1), dim=1)
        return {"Min": -amin}


class Max:
    def __call__(self, x: torch.Tensor):
        amin, amax = torch.aminmax(x.view(x.shape[0], -1), dim=1)
        return {"Max": amax}


# Based on https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, transform, features):
        super().__init__()
        print("\nconfidenciator.py  ==> FeatureExtractor()")
        self.model = model
        self.feat_fns = features
        self._features = {}
        self.device = get_torch_device()
        self.transform = transform
        self.knn_features = []
        i = 0
        # print(model)
        supported_activations = nn.ReLU, nn.GELU, nn.LeakyReLU
        print("\nExtracting Activation Layers:\n")
        for layer in model.modules():
            if isinstance(layer, supported_activations):
                print("layer")
                layer.register_forward_hook(
                    self.save_features_hook(f"relu_{i}"))
                print(f"{layer} Is Relu {i}")   # ReLU6(inplace=True) Is Relu 1
                i += 1
                # print(f"{layer} Is Relu {i}")

    def save_features_hook(self, layer_id: str) -> Callable:
        print("confidenciator.py  ==> FeatureExtractor.save_features_hook()")
        def fn(_, input_, __):
            for f in self.feat_fns:
                for name, output in f(input_[0]).items():
                    self._features[f"{name}_{layer_id}"] = output
        return fn

    def forward(self, x):
        # print("confidenciator.py  ==> FeatureExtractor.forward()")
        # print("forward is called")
        output = self.model(x)
        return output, self._features
    
    def predict(self, images):   # predict_openood
        # replace the predict with predict_openood when running with document dataset
        """ 
        receives an array of (5000,3,32,32) size (for cifar10)
        returns output_np and features
        output_np.shape = (50000,10)  # predictions of corresponding images
        features = a dictionary containing the Max_relu_n, Min_relu_n for all the images
        this function adds activation layer features Max_relu_n, Min_relu_n
        """
        print("confidenciator.py  ==> FeatureExtractor.predict()")
        print("predict() openood version called")

        #print("incoming shape to FeatureExtractor.predict()  : ", images.shape)
        # images = torch.tensor(images, dtype=torch.float)
        # images = self.transform(images)
        # images = TensorDataset(images)
        # images = DataLoader(images, batch_size=128)
        output = []
        labels = []
        features = {}
        with torch.no_grad():
            for i, data in enumerate(images):
                
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print(" 
                # print("data shape: ", data.keys())
                label = data["label"]
                data = data["data"]
                # print("data 2.3 shape: ", data.shape)
                
                                                                         # ", end="\r")
                #data = data[0].to(self.device)
                data = data.to(self.device)
                # print("data 2.4 shape: ", data.shape)
                #data = torch.moveaxis(data, 1, 3)
                # print("size of data: ", data.shape)
                #data = torch.reshape(data, (-1,))
                #print("size of data: ", data.shape)
                
                # sys.exit()
                out, feat = self(data)
                #print("passing this line")
                output.append(out)
                labels.append(label)
                
                # print("Self Features in forward funtion is : ",
                #       self._features.keys())
                if len(features) == 0:
                    features = {key: [] for key in self._features.keys()}
                for k in features.keys():
                    features[k].append(feat[k])
                
                # # Stop after a certain number of batches (e.g., 10)
                # if i== 5:
                #     break
                    
            for k in features.keys():
                features[k] = torch.cat(features[k]).cpu().detach().numpy()
        output_np = torch.cat(output).cpu().detach().numpy()
        labels_np = torch.cat(labels).cpu().detach().numpy()
        return labels_np, output_np, features

    def predict_knn(self, images):  # predict_knn_openood
        # replace the predict_knn with predict_knn_openood when running with document dataset
    
        print("confidenciator.py  ==> FeatureExtractor.predict_knn()")
        print("predict_knn() openood version called")

        #images = torch.tensor(images, dtype=torch.float32)
        #images = self.transform(images)
        #images = TensorDataset(images)
        # images = DataLoader(images, batch_size=128)
        # changing this batch size for imagenet from 128 to 256
        #images = DataLoader(images, batch_size=256)

        output = []
        pen_features = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print("i Value: ", i)
                #print("Self Features in forward funtion is : ", self._features.keys())
                #print("data[0].shape :",data[0].shape)
                #batch_size = data[0].shape
                #print("batch_size[0] :", batch_size[0])
                
                #data = data[0].to(self.device)
                data = data["data"]
                data = data.to(self.device)
                out, features = self.model.forward_knn2(data, return_feature_list = True)
                # print("feature.shape :",feature.shape)
                # print("length of features :", len(features))
                # print("shape of final layer :", features[-1].shape)
                feature = features[-1]

                # if len(features) == 0:
                #   feat = {key: [] for key in self._features.keys()}
                # for k in feat.keys():
                #   print("Feature k: ", k)
                #  print("feature data: " , feature.shape)
                # features.append(normalizer(feature.data.cpu().numpy()))
                 
                dim = feature.shape[1]
                # normalizer1 = normalizer(feature.data.cpu().numpy().reshape(int(batch_size[0]), dim, -1))
                # pen_features.append(np.squeeze(normalizer1))
                # activation_log.append(np.squeeze(normalizer1))
                
                pen_features.append(normalizer(feature.data.cpu().numpy().reshape(int(data.shape[0]),dim , -1).mean(2)))
                
                if out is not None:
                    output.append(out)
                    
                # # Stop after a certain number of batches (e.g., 10)
                # if i== 5:
                #     break
                
            self.knn_features = np.concatenate(pen_features, axis=0)
        return output if len(output) == 0 else torch.cat(output), self.knn_features

   
    def predict_docu(self, images):   # predict_docu
        # rename this function to  predict()  when running with document datasets
        # keep the name predict_docu() if you are not using document dataset, rather using openood datsets
        # predict_docu(self, images):
        """ 
        receives an array of (5000,3,32,32) size (for cifar10)
        returns output_np and features
        output_np.shape = (50000,10)  # predictions of corresponding images
        features = a dictionary containing the Max_relu_n, Min_relu_n for all the images
        this function adds activation layer features Max_relu_n, Min_relu_n
        """
        print("confidenciator.py  ==> FeatureExtractor.predict()")
        print("predict() document version called")

        #print("incoming shape to FeatureExtractor.predict()  : ", images.shape)
        # images = torch.tensor(images, dtype=torch.float)
        # images = self.transform(images)
        # images = TensorDataset(images)
        # images = DataLoader(images, batch_size=128)
        output = []
        labels = []
        features = {}
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print(" 
                # print("data shape: ", data.keys())
                
                # commenting for document
                # data = data["data"]
                # label = data["label"]

                # print("before size of data[0].shape: ", data[0].shape)
                
                if isinstance(data, list):
                    data = data[0].to(self.device)

                else:
                    data = data.to(self.device)


                # data = data[0].to(self.device)
                # data = data.to(self.device)
                #data = torch.moveaxis(data, 1, 3)
                # print("size of data: ", data.shape)
                #data = torch.reshape(data, (-1,))
                # print("size of data: ", data.shape)
                
                #sys.exit()
                out, feat = self(data)
                #print("passing this line")
                output.append(out)
                # labels.append(label)
                
                # print("Self Features in forward funtion is : ",
                #       self._features.keys())
                if len(features) == 0:
                    features = {key: [] for key in self._features.keys()}
                for k in features.keys():
                    features[k].append(feat[k])
            for k in features.keys():
                features[k] = torch.cat(features[k]).cpu().detach().numpy()
        output_np = torch.cat(output).cpu().detach().numpy()
        # labels_np = torch.cat(labels).cpu().detach().numpy()
        labels_np = []
        return labels_np, output_np, features
    
    

    def predict_knn_docu(self, images):  # predict_knn_docu 
    # rename this function to  predict_knn()  when running with document datasets
    # keep the name to predict_knn_docu() if you are running openood dataset
        print("confidenciator.py  ==> FeatureExtractor.predict_knn()")
        print("predict_knn() document version called")

        #images = torch.tensor(images, dtype=torch.float32)
        #images = self.transform(images)
        #images = TensorDataset(images)
        # images = DataLoader(images, batch_size=128)
        # changing this batch size for imagenet from 128 to 256
        #images = DataLoader(images, batch_size=256)

        output = []
        pen_features = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print("i Value: ", i)
                #print("Self Features in forward funtion is : ", self._features.keys())
                #print("data[0].shape :",data[0].shape)
                #batch_size = data[0].shape
                #print("batch_size[0] :", batch_size[0])
                
                # data = data[0].to(self.device)
                
                if isinstance(data, list):
                    data = data[0].to(self.device)

                else:
                    data = data.to(self.device)
                
                # commenting out for document data
                # data = data["data"]
                # data = data.to(self.device)
                
                out, features = self.model.forward_knn2(data, return_feature_list = True)
                # print("feature.shape :",feature.shape)
                # print("length of features :", len(features))
                # print("shape of final layer :", features[-1].shape)
                feature = features[-1]

                # if len(features) == 0:
                #   feat = {key: [] for key in self._features.keys()}
                # for k in feat.keys():
                #   print("Feature k: ", k)
                #  print("feature data: " , feature.shape)
                # features.append(normalizer(feature.data.cpu().numpy()))
                 
                dim = feature.shape[1]
                # normalizer1 = normalizer(feature.data.cpu().numpy().reshape(int(batch_size[0]), dim, -1))
                # pen_features.append(np.squeeze(normalizer1))
                # activation_log.append(np.squeeze(normalizer1))
                
                pen_features.append(normalizer(feature.data.cpu().numpy().reshape(int(data.shape[0]),dim , -1).mean(2)))
                
                if out is not None:
                    output.append(out)
            self.knn_features = np.concatenate(pen_features, axis=0)
        return output if len(output) == 0 else torch.cat(output), self.knn_features

    def predict_react(self, images, f=torch.logsumexp):
        print("confidenciator.py  ==> FeatureExtractor.predict_react()")
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print("                                                          ", end="\r")
                data = data[0].to(self.device)
                logits = self.model.forward_threshold(data)
                # out = torch.nn.functional.softmax(logits, dim=1)
                out = f(logits.data, dim=1)
                output.append(out)
        return torch.cat(output).cpu().detach().numpy()

    def predict_f(self, images, f):
        print("confidenciator.py  ==> FeatureExtractor.predict_f()")
        images = torch.tensor(images, dtype=torch.float)
        images = self.transform(images)
        images = TensorDataset(images)
        images = DataLoader(images, batch_size=128)
        output = []
        with torch.no_grad():
            for i, data in enumerate(images):
                # print(f"Computing predictions: {i + 1}/{len(images)}             ", end="\r")
                # print("                                                          ", end="\r")
                data = data[0].to(self.device)
                logits = self.model.forward(data)
                out = f(logits, dim=1) if f else logits
                output.append(out)
        return torch.cat(output).cpu().detach().numpy()


class Confidenciator:

    def __init__(self, model: nn.Module, transform, train_set,mahala_xood,knn_pen , features=(MinMax(),),reg= 1):
        print("\n\n ##  Creating Confidenciator  ##")
        print("confidenciator.py  ==> __init__()")
        self.model = FeatureExtractor(model, transform, features)
        self.feat_cols = []
        #print("train set shape Before add_prediction_and_features : ", train_set.shape)
        ##
        if not mahala_xood:
            train_set_mahala = self.add_prediction_and_penultimate_features_dl_to_mahala(train_set)
        else:
            train_set_mahala = self.add_prediction_and_features_dl(train_set)
            
        # if knn_pen:
        #     train_set_knn = self.add_prediction_and_features_knn(train_set)
        # else:
        #     train_set_knn = self.add_prediction_and_extreme_features_dl_to_knn(train_set)
            
        
        ##
        # train_set_mahala = self.add_prediction_and_features_dl(train_set)
        train_set_knn = self.add_prediction_and_features_knn(train_set)
        # train_set_final_mahala = self.add_prediction_and_penultimate_features_dl_to_mahala(train_set)
        # print("train set shape After add_prediction_and_features: ", train_set.shape)
        # combine_train_set_mahala_knn = pd.concat([train_set_mahala, train_set_knn], ignore_index=True, axis=1)
        self.index = None
        self.K = 50

        #train_set = train_set[train_set["is_correct"]]
        # print("combine_train_set_mahala_knn of shape:  ", combine_train_set_mahala_knn.shape)
        self.lr = None
        self.coeff = None
        self.concatenated_vectors = None
        self.pt_combine = PowerTransformer()
        self.pt_final_mahala = PowerTransformer()
        self.scaler_final_mahala = StandardScaler()
        self.scaler_combine = StandardScaler()
        self.pt = PowerTransformer()
        self.pt_knn = PowerTransformer()
        self.scaler = StandardScaler()
        self.scaler_knn = StandardScaler()
        print("[self.feat_cols]:\n", [self.feat_cols])
        x = self.pt.fit_transform(
            self.scaler.fit_transform(train_set_mahala))
        # x = self.pt.fit_transform(
        #     self.scaler.fit_transform(train_set_mahala[self.feat_cols]))
        # x_combine = self.pt_combine.fit_transform(self.scaler_combine.fit_transform(combine_train_set_mahala_knn))
        # x_final_mahala = self.pt_final_mahala.fit_transform(self.scaler_final_mahala.fit_transform(train_set_final_mahala))
        
        # if reg < np.inf:
        #     cov = np.cov(x, rowvar=False)
        #     # cov_comb = np.cov(x_combine, rowvar=False)
        #     # cov_final_mahala = np.cov(x_final_mahala, rowvar=False)
        #     self.inv_cov = np.linalg.inv(
        #         cov + reg * np.identity(len(self.feat_cols)))
        #     # self.combine_inv_cov = np.linalg.inv(
        #     #     cov_comb + reg * np.identity(combine_train_set_mahala_knn.shape[1]))
        #     # self.final_mahala_inv_cov = np.linalg.inv(
        #         # cov_final_mahala + reg * np.identity(train_set_final_mahala.shape[1]))
        # else:
        #     self.inv_cov = np.identity(len(self.feat_cols))
            
            ##
        if reg < np.inf:
            cov = np.cov(x, rowvar=False)
                    # cov_comb = np.cov(x_combine, rowvar=False)
                    # cov_final_mahala = np.cov(x_final_mahala, rowvar=False)
            if not mahala_xood:
                self.inv_cov = np.linalg.inv(cov + reg * np.identity(train_set_mahala.shape[1]))
            else:
                self.inv_cov = np.linalg.inv(cov + reg * np.identity(len(self.feat_cols)))
                    # self.combine_inv_cov = np.linalg.inv(
                    #     cov_comb + reg * np.identity(combine_train_set_mahala_knn.shape[1]))
                    # self.final_mahala_inv_cov = np.linalg.inv(
                        # cov_final_mahala + reg * np.identity(train_set_final_mahala.shape[1]))
        else:
            self.inv_cov = np.identity(len(self.feat_cols))
            
            ##
            
        # self.mean = np.zeros(len(self.feat_cols))
        if not mahala_xood:
            self.mean = np.zeros(train_set_mahala.shape[1])
        else:
            self.mean = np.zeros(len(self.feat_cols))
        # self.comb_mean = np.zeros(combine_train_set_mahala_knn.shape[1])
        # self.mean_final_mahala = np.zeros(train_set_final_mahala.shape[1])
        self.reg = reg
        
        # calculating mahala for trainset
        self.mahala_train = -np.apply_along_axis(lambda row: mahalanobis(row, self.mean, self.inv_cov), 1, x)
        
        mahala_sq = -(self.mahala_train ** 2)
        self.mahala_mean = np.abs(mahala_sq.mean())
        self.mahala_std = np.abs(mahala_sq.std())
        self.mahala_max_mean = self.mahala_train.mean()
        self.mahala_max_std = self.mahala_train.std()
        print("**********************")
        print("Mahala Mean: ", self.mahala_mean)
        print("Mahala Std: ", self.mahala_std)
        print("**********************")
        print("**********************")
        print("Mahala Max Mean: ", self.mahala_max_mean)
        print("Mahala Max Std: ", self.mahala_max_std)
        print("**********************")

    def add_prediction_and_features(self, df: pd.DataFrame):
        print("\nconfidenciator.py  ==> Confidenciator.add_prediction_and_features()")
        
        pred, features = self.model.predict(
            get_images_and_labels(df, labels=False, chw=True))
        
        if len(self.feat_cols) == 0:
            self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())
            
        df["pred"] = np.argmax(pred, axis=-1)
        df["is_correct"] = df["pred"] == df["label"].to_numpy()
        df["Max_out"] = np.max(pred, axis=-1)
        df["Min_out"] = -np.min(pred, axis=-1)
        df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
        # self.extreme_value_vector = df[self.feat_cols]
        print("returning add_prediction_and_features() with shape:", df.shape)
        return df
    
    def add_prediction_and_features_dl(self, dataloader):
        print("\nconfidenciator.py  ==> Confidenciator.add_prediction_and_features_dl()")
        
        labels, pred, features = self.model.predict(dataloader)
        
        if len(self.feat_cols) == 0:
            self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())
            
        df = pd.DataFrame(features)
        # df["pred"] = np.argmax(pred, axis=-1)
        # df["is_correct"] = df["pred"] == labels
        df["Max_out"] = np.max(pred, axis=-1)
        df["Min_out"] = -np.min(pred, axis=-1)
        # self.extreme_value_vector = df[self.feat_cols]
        print("returning add_prediction_and_features_dl() with shape:", df.shape)
        return df
    
    def add_prediction_and_penultimate_features_dl_to_mahala(self, dataloader):
        print("\nconfidenciator.py  ==> Confidenciator.add_prediction_and_penultimate_features_dl_to_mahala()")
        
        # pred, features = self.model.predict_knn(dataloader)
        pred, features = self.model.predict_knn(dataloader)
        
        # if len(self.feat_cols) == 0:
        #     self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())

        #pred, features = self.model.predict(get_images_and_labelsd(df, labels=False, chw=True))
        df = pd.DataFrame(features)
        print("Mahala dataset shape: ", df.shape)
        #return df
        return df

    def add_prediction_and_extreme_features_dl_to_knn(self, dataloader):
        print("\nconfidenciator.py  ==> Confidenciator.add_prediction_and_extreme_features_dl_to_knn()")
        
        labels, pred, features = self.model.predict(dataloader)
        
        if len(self.feat_cols) == 0:
            self.feat_cols = ["Max_out", "Min_out"] + list(features.keys())
            
        df = pd.DataFrame(features)
        #df["pred"] = np.argmax(pred, axis=-1)
        #df["is_correct"] = df["pred"] == labels
        df["Max_out"] = np.max(pred, axis=-1)
        df["Min_out"] = -np.min(pred, axis=-1)
        # self.extreme_value_vector = df[self.feat_cols]
        print("returning add_prediction_and_extreme_features_dl_to_knn() with shape:", df.shape)
        return df

    def add_prediction_and_features_knn(self, dataloader):
        print("\nconfidenciator.py  ==> Confidenciator.add_prediction_and_features_knn()")
        pred, features = self.model.predict_knn(dataloader)
        #pred, features = self.model.predict(get_images_and_labelsd(df, labels=False, chw=True))
        df = pd.DataFrame(features)
        return df

    def fit(self, cal: Dict[str, pd.DataFrame], c=None):
        print("confidenciator.py  ==> Confidenciator.fit()")
        nbr_folds = len(cal)
        cal = pd.concat(list(cal.values()), ignore_index=True)
        cal = self.add_prediction_and_features(cal)
        features = split_features(self.pt.transform(
            self.scaler.transform(cal[self.feat_cols])))
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2",
             solver="liblinear", class_weight="balanced"))
        ])
        if c is None:
            params = {"lr__C": list(np.logspace(-8, 0, 17)), }
            grid = GridSearchCV(
                self.lr, params, scoring='roc_auc', n_jobs=30, cv=nbr_folds)
            grid.fit(X=features, y=cal["is_correct"].to_numpy())
            self.lr = grid.best_estimator_
            print(pd.DataFrame(grid.cv_results_)[
                  ["mean_test_score", "std_test_score", "rank_test_score", "params"]])
        else:
            self.lr.fit(features, cal["is_correct"])
        self.coeff = pd.Series(self.lr["lr"].coef_[0],
                               [c + "+" for c in self.feat_cols] + [c + "-" for c in self.feat_cols])
        print(self.coeff)

    def fit_knn_faiss(self, df: pd.DataFrame, c=None):
        print("confidenciator.py  ==> Confidenciator.fit_knn_faiss()")
        ##
        # save pickle file here
        self.index = faiss.IndexFlatL2(df.shape[1])

        #x = self.pt.transform(self.scaler.transform(df[self.feat_cols]))
        # x = self.pt_knn.fit_transform(self.scaler_knn.fit_transform(df))
        # print("X Shape before adding to index: ", x.shape)
        x = df.to_numpy()
        self.knn_n = x.shape[1]
        self.index.add((np.ascontiguousarray(x.astype(np.float32))))
        train_D, _ = self.index.search((np.ascontiguousarray(x.astype(np.float32))), self.K)
        kth_train_dist = -train_D[:, -1]
        
        print("**********************")
        knn_log = - df.shape[1] * np.log(-kth_train_dist, where = -kth_train_dist > 0.0)
        self.knn_std = np.abs(knn_log.std())
        self.knn_mean =  (np.abs(knn_log).mean())
        self.knn_max_mean = np.abs(kth_train_dist.mean())
        self.knn_max_std= np.abs(kth_train_dist.std())
        print("**")
        print("**")
        print("KNN Mean: ", self.knn_mean)
        print("KNN Std: ", self.knn_std)
        print("**")
        print("**")
        print("KNN Max Mean: ", self.knn_max_mean)
        print("KNN Max Std: ", self.knn_max_std)
        print("**")

        print("Training Data fit KNN Faiss completed..")


    def fit_knn(self, cal: Dict[str, pd.DataFrame], c=None):
        print("confidenciator.py  ==> Confidenciator.fit_knn()")
        self.knn = KNeighborsClassifier(n_neighbors=3)
        cal = pd.concat(list(cal.values()), ignore_index=True)
        cal = self.add_prediction_and_features(cal)
        features = features = split_features(self.pt.transform(
            self.scaler.transform(cal[self.feat_cols])))
        print("Shape of the Features fitting KNN: ", features.shape)
        self.knn.fit(features, cal["is_correct"])

    def predict_knn_faiss(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==> Confidenciator.predict_knn_faiss()")
        print("flag 2.12a TESTING DATASET: ", dataset.shape)
        #dataset = dataset[self.feat_cols]
        # if not isInTest:
        #   output, feature_normed = self.model.predict_knn(get_images_and_labels(dataset, labels=False, chw=True), isOOD=True)
        # else:

        # extreme values of KNN
        # feature_normed = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        # feature_normed = self.pt_knn.transform(self.scaler_knn.transform(dataset))
        
        ### this block is for saving pickles for saiful start
        # if dataset.shape == (10000, 2048):
        #     with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/dataset_inaturalist_testdata_from_xood.pickle', 'wb') as handle:
        #         pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # elif dataset.shape == (45000, 2048):
        #     with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/dataset_imagenet_testdata_from_xood.pickle', 'wb') as handle:
        #         pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        ### this block is for saving pickles for saiful end
        
        feature_normed = dataset.to_numpy()  # this is for KNN
        #feature_normed = (np.ascontiguousarray(feature_normed.astype(np.float32)))
        print("Test Feature Normed : ", feature_normed.shape)
        print("Test Output shape : ", feature_normed.shape)
        
        t1 = time.time()
        
        D, _ = self.index.search((np.ascontiguousarray(
            feature_normed.astype(np.float32))), self.K)
        kth_dist = -D[:, -1]
        t2 = time.time()
        
        elapsed_time = t2 -t1
        print("flag 2.12b Elapsed time knn:", elapsed_time, "seconds")

        return kth_dist

    def predict_mahala(self, dataset: pd.DataFrame):  # this was buggy, now its working properly
        print("confidenciator.py  ==> Confidenciator.predict_mahala()")
        print("flag 2.11a TESTING DATASET: ", dataset.shape)
        # if not all(col in dataset.columns for col in self.feat_cols):
        #     dataset = self.add_prediction_and_features(dataset)
         
        t1 = time.time()
        # x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        x = self.pt.transform(self.scaler.transform(dataset))
        
        if self.reg < np.inf:
            pred_m = -np.apply_along_axis(lambda row: mahalanobis(row, self.mean, self.inv_cov), 1, x)
            t2 = time.time()
            elapsed_time = t2 -t1
            print("flag 2.11b Elapsed time mahala:", elapsed_time, "seconds")
            return pred_m
        
        pred_m = -np.apply_along_axis(lambda row: np.linalg.norm(row - self.mean, ord=2), 1, x)
        t2 = time.time()
        elapsed_time = t2 -t1
        print("flag 2.11b Elapsed time mahala:", elapsed_time, "seconds")

        return pred_m
    
    def predict_mahala_ok(self, dataset: pd.DataFrame):  # this is from the older version
        print("confidenciator.py  ==> Confidenciator.predict_mahala()")
        print("dataset.shape initial", dataset.shape)
        if not all(col in dataset.columns for col in self.feat_cols):
            print("inside if cond of predict_mahala")
            dataset = self.add_prediction_and_features(dataset)
        print("dataset.shape second:", dataset.shape)
            
        x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        print("x.shape:", x.shape)
        if self.reg < np.inf:
            return -np.apply_along_axis(lambda row: mahalanobis(row, self.mean, self.inv_cov), 1, x)
        return -np.apply_along_axis(lambda row: np.linalg.norm(row - self.mean, ord=2), 1, x)

    def predict_comb_mahala(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==> Confidenciator.predict_comb_mahala()")
        print("dataset.shape initial", dataset.shape)
        # if not all(col in dataset.columns for col in self.feat_cols):
        #     dataset = self.add_prediction_and_features(dataset)
            
        # x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        x = self.pt_combine.transform(self.scaler_combine.transform(dataset))
        
        if self.reg < np.inf:
            return -np.apply_along_axis(lambda row: mahalanobis(row, self.comb_mean, self.combine_inv_cov), 1, x)
        return -np.apply_along_axis(lambda row: np.linalg.norm(row - self.mean, ord=2), 1, x)
    
    def predict_final_mahala(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==> Confidenciator.predict_final_mahala()")
        print("dataset.shape initial", dataset.shape)
        # if not all(col in dataset.columns for col in self.feat_cols):
        #     dataset = self.add_prediction_and_features(dataset)
            
        # x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        x = self.pt_final_mahala.transform(self.pt_final_mahala.transform(dataset))
        
        if self.reg < np.inf:
            return -np.apply_along_axis(lambda row: mahalanobis(row, self.mean_final_mahala, self.final_mahala_inv_cov), 1, x)
        return -np.apply_along_axis(lambda row: np.linalg.norm(row - self.mean, ord=2), 1, x)

    def predict_proba(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==> Confidenciator.predict_proba()")
        if not all(col in dataset.columns for col in self.feat_cols):
            dataset = self.add_prediction_and_features(dataset)
        x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        return self.lr.predict_proba(split_features(x))[:, 1]

    def predict_knn(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  Confidenciator.predict_knn() ")
        if not all(col in dataset.columns for col in self.feat_cols):
            dataset = self.add_prediction_and_features(dataset)
        x = self.pt.transform(self.scaler.transform(dataset[self.feat_cols]))
        return self.knn.predict_proba(split_features(x))[:, 1]

    def react_energy(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  react_energy() ")
        pred = self.model.predict_react(
            get_images_and_labels(dataset, labels=False, chw=True))
        return pred

    def react_max(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  react_max() ")
        pred = self.model.predict_react(get_images_and_labels(
            dataset, labels=False, chw=True), torch.amax)
        return pred

    def react_softmax(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  react_softmax() ")
        pred = self.model.predict_react(get_images_and_labels(
            dataset, labels=False, chw=True), F.softmax)
        return np.max(pred, axis=-1)

    def energy(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  energy() ")
        pred = self.model.predict_f(get_images_and_labels(
            dataset, labels=False, chw=True), torch.logsumexp)
        return pred

    def softmax(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  softmax() ")
        pred = self.model.predict_f(get_images_and_labels(
            dataset, labels=False, chw=True), F.softmax)
        return np.max(pred, axis=-1)

    def max(self, dataset: pd.DataFrame):
        print("confidenciator.py  ==>  max() ")
        pred = self.model.predict_f(get_images_and_labels(
            dataset, labels=False, chw=True), None)
        return np.max(pred, axis=-1)


def split_features(features: np.ndarray):
    print("confidenciator.py  ==>  split_features() ")
    return np.concatenate([- np.clip(features, 0, None), - np.clip(-features, 0, None)], axis=1)


# to run the code for document dataset you need to rename the function of FeatureExtractor.predict() and FeatureExtractor.predict_knn()
# rename the function predict_openood() to predict()
# and predict_knn_openood() to predict_knn()