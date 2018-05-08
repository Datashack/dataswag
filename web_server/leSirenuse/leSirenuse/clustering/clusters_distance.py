import numpy as np
import pandas as pd
from leSirenuse.clustering import Predicting

def get_result():
    obj_pred = Predicting.Main('', False)
    model = obj_pred.load_model("leSirenuse/clustering/model_users.plk")
    #Filter Data
    data = pd.DataFrame(model.means_)

    #Calculate distance to each cluster
    final = np.zeros((len(data),len(model.means_)+1))
    for i in range(0,len(data)):
        temp_dist = []
        for j in range(0, len(model.means_)):
            dist_vec = np.linalg.norm(data.iloc[i,:].values.reshape(1,-1)-model.means_[j].reshape(1,-1))
            temp_dist.append(dist_vec)
        final[i,:] = temp_dist + [np.argmin(temp_dist)]
    columns = ['Dist_'+str(i) for i in range(len(model.means_))]+['Prediction']
    df_final = pd.DataFrame(final, columns=columns)
    print(df_final)

get_result()
