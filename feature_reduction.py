from sklearn.manifold import TSNE
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def viz_plotly(data):
    fig = px.scatter(data, x='x', y='y', color="is_correct")
    fig.write_image("feature_tras.png")

def tnse_reduction(df):
    transform = TSNE
    trans = transform(n_components=2)
    emb_transformed_tsne = pd.DataFrame(trans.fit_transform(df))
    return emb_transformed_tsne


if __name__ == "__main__":
    data_conf_cifar10 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar10transformer_conf.pkl")
    data_conf_cifar100 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar100transformer_conf.pkl")
    data_Train_cifar10 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar10transformer_Train.pkl")
    data_Train_cifar100 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar100transformer_Train.pkl")
    data_Val_cifar10 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar10transformer_Val.pkl")
    data_Val_cifar100 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar100transformer_Val.pkl")
    data_Test_cifar10 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar10transformer_Test.pkl")
    data_Test_cifar100 = pd.read_pickle(f"out_of_distribution/feature_transfomers_cifar100transformer_Test.pkl")
    #data_cifar10 = data_cifar10.dropna(axis=1)
    #data_cifar100 = data_cifar10.dropna(axis=1)
    #df_class = data["is_correct"]
    #data_cifar10 = data_cifar10.drop(columns=["is_correct"], axis=1)
    #data_cifar100 = data_cifar100.drop(columns=["is_correct"], axis=1)
    #red_data_cifar10 = tnse_reduction(data_cifar10)
    #red_data_cifar100 = tnse_reduction(data_cifar100)
    #final = pd.concat([red_data, df_class], axis=1)
    #final = final.rename({0:'x', 1:'y'}, axis=1)
    #viz_plotly(final)
    print("Shapes of Cifar 10 Conf: {0} \n, Train: {1} \n, Val: {2} \n, Test: {3} : "
    .format(data_conf_cifar10.shape, data_Train_cifar10.shape, data_Val_cifar10.shape, data_Test_cifar10.shape))
    print("Shapes of Cifar 100 Conf: {0} \n, Train: {1} \n, Val: {2} \n, Test: {3} : "
    .format(data_conf_cifar100.shape, data_Train_cifar100.shape, data_Val_cifar100.shape, data_Test_cifar100.shape))


    
