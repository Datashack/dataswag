from leSirenuse.clustering import Predicting
from leSirenuse.clustering import Image_CNN
from sklearn.externals import joblib
from scipy import stats
import pandas as pd


class Backend:
    def __init__(self, target_path):
        self.df_target_presence = None
        print('importing Predicting')
        self.obj_pred = Predicting.Main(target_path, False)
        print('importing images model')
        self.model_images = self.obj_pred.load_model("leSirenuse/clustering/model_images.plk")
        print('importing users model')
        self.model_users = self.obj_pred.load_model("leSirenuse/clustering/model_users.plk")

    def compute_pics_presence(self):
        #CNN initialization has to be done by the same thread calling predict_image
        print('importing CNN')
        self.obj_cnn = Image_CNN.Main('leSirenuse/clustering/KRM_weights-260-0.69.hdf5')
        df_target = self.obj_pred.get_target_posts()

        #Import Model and Embedding - create image cnn obj
        prediction = self.obj_cnn.predict_image(df_target['Image'])
        df_target = self.obj_cnn.combine_image(df_target, prediction, "CNN_Feature_")
        #Convert to Features
        extra_cols = ['File', 'Image']
        self.df_target_presence = self.obj_pred.get_cluster_presence(df_target, extra_cols, self.model_images)

    def get_scores(self, target):
        probabilities = self.df_target_presence[['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']]
        files = []
        KL= []
        community = []
        for i in range(probabilities.shape[0]):
            pic_dist = list(probabilities.iloc[i])
            for v in enumerate(self.model_users.means_):
                clustercenter = (v[1])
                KLdiv = stats.entropy(pk=clustercenter, qk=pic_dist)
                if (KLdiv == float('inf')):
                    KLdiv = 563.2
                KL.append(KLdiv)
                files.append(str(self.df_target_presence.File[i]))
                community.append((v[0]))
        KLdivergencedf = (pd.DataFrame({"KL_score": KL,"picture_uploaded": files, 'community': community }))
        scores = KLdivergencedf.sort_values('KL_score').reset_index()
        return scores

    def get_competitors_distance(self):
        extra_cols = ['File','Prediction']
        company_final_df = self.obj_pred.get_dist2comp(self.df_target_presence, extra_cols)
        return company_final_df
