from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import sys

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.pickle_data = []
        
        self.activation_log2= None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        activation_log = []
        activation_log2 = []

        net.eval()
        print("****************")
        # print("id_loader_dict['val'] :", id_loader_dict['val'])
        # val_loader = id_loader_dict["val"]
        # print("len(val_loader.dataset) :", len(val_loader.dataset))
        # val_features_dict = next(iter(val_loader))
        # print("val_features_dict.keys()", val_features_dict.keys())
        # val_features = val_features_dict["data"]
        # val_labels = val_features_dict["label"]
        # print("len(val_features):", len(val_features))
        # print("len(val_labels):", len(val_labels))
        
        print("id_loader_dict['train'] :", id_loader_dict['train'])
        train_loader = id_loader_dict["train"]
        print("len(train_loader.dataset) :", len(train_loader.dataset))
        print("train_loader :", train_loader)
    


        print("****************")
        count= 0
        
        with torch.no_grad():
            
            for batch in tqdm(id_loader_dict['train'],
                              desc='Eval: ',
                              position=0,
                              leave=True):
                print("count :", count)
                if count == 16:
                    break
                data = batch['data'].cuda()
                data = data.float()

                batch_size = data.shape[0]

                _, features = net(data, return_feature_list=True)
                print("saiful 2 len(features) :",len(features))


                feature = features[-1]
                print("saiful 2 feature.shape :",feature.shape)
                dim = feature.shape[1]
                activation_log.append(
                    normalizer(feature.data.cpu().numpy().reshape(
                        batch_size, dim, -1).mean(2)))
                
                # normalizer1 = normalizer(feature.data.cpu().numpy().reshape(batch_size, dim, -1))
                # activation_log.append(np.squeeze(normalizer1))

                
                # activation_log2.append(
                #     normalizer(feature.data.cpu().numpy().reshape(
                #         batch_size, dim, -1)))
                # print("number of running batch :", batch)
                count= count +1

        self.activation_log = np.concatenate(activation_log, axis=0)
        if self.activation_log.shape[0]== 4096:
            print("saiful total number of images  ")
        else:
            print("saiful total number of images :",self.activation_log.shape)
            # sys.exit()
        
        # print("activation_log2 shape", activation_log2[0].shape)
        #
        # print("saiful activation_log.shape : ", activation_log.shape)
        print("saiful Activation Log shape : ", self.activation_log.shape)

        # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/activation_log_openood.pickle', 'wb') as handle:
        #     pickle.dump(self.activation_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/feature_imagenet_valdata_from_openood.pickle', 'wb') as handle:
        #     pickle.dump(feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        # print("knn_postprocess.py  ==> postprocess()")
        # print("saiful_feature_normed.shape: ",feature_normed.shape)
        
        ## saifuls testing purpose
        
        # self.pickle_data.append(feature_normed)
    
            
        # if feature_normed.shape[0] == 200:
        #     pickle_file = np.concatenate(self.pickle_data, axis=0)
        #     print("length of pickle dataset for imagnet testdata:", len(pickle_file))

        #     self.pickle_data.clear()
            
        #     with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/imagenet_testdata_from_openood.pickle', 'wb') as handle:
        #         pickle.dump(pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # elif feature_normed.shape[0] == 16:
        #     pickle_file_inat = np.concatenate(self.pickle_data, axis=0)
        #     print("length of pickle dataset for inaturalist testdata:", len(pickle_file_inat))

        #     self.pickle_data.clear()

        #     with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/Inatiraluist_testdata_from_openood.pickle', 'wb') as handle:
        #         pickle.dump(pickle_file_inat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # elif feature_normed.shape[0] == 3:
        #     print("length of pickle automatic parameter:", len(self.pickle_data))
        #     self.pickle_data.clear()
        # else:
        #    pass
        


        
        # ##
        # activation_log2 = []
        # print("****************")
        # for batch in tqdm(data):
        #     output, feature = net(data, return_feature=True)
        #     feature_normed = normalizer(feature.data.cpu().numpy())

        #     batch_size = data.shape[0]

        #     _, features = net(data, return_feature_list=True)

        #     feature = features[-1]
        #     dim = feature.shape[1]
        #     activation_log2.append(feature_normed)
        # self.activation_log2 = np.concatenate(activation_log2, axis=0)
        # print("saiful Activation Log2 shape : ", self.activation_log2.shape)
        # print("****************")

        ##
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
