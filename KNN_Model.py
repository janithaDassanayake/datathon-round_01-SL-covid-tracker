import pandas as pd
import numpy as np
from sklearn  import preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df=pd.read_csv("geographycal_file/geograohycal.csv")

# Normalize total_bedrooms column
x_array = np.array(df['Status'])
normalized_X = preprocessing.normalize([x_array])
df["Status"]=normalized_X[0]

df_matrix=df.pivot_table(index="Detected Prefecture",columns="Date Announced",values="Status").fillna(0)
from scipy.sparse import csr_matrix

dis_fetures=csr_matrix(df_matrix)

from sklearn.neighbors import NearestNeighbors

model_NN=NearestNeighbors(metric="cosine",algorithm="brute")
model_NN.fit(dis_fetures)
dis_list=list(df_matrix.index)


def getNearstDis(district):
    global dis_list
    global df_matrix
    global model_NN
    query_index = dis_list.index(district)
    dis, ind = model_NN.kneighbors(df_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=5)

    res = {}

    for i in range(0, len(dis.flatten())):
        if i == 0:
            res[df_matrix.index[query_index]] = {}
        else:
            res[df_matrix.index[query_index]][df_matrix.index[ind.flatten()[i]]] = dis.flatten()[i]
    return res
