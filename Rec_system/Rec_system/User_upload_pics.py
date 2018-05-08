

#Target path is where the pics for the user are stored
## Moreno- Idk what to do about Loading 'obj-cnn' and loading model_images and model_users

import Image_CNN
import  Clustering
import Predicting

import cv2 ## issues 
import skimage
from skimage import io
import pickle
#define File-Path to Users Folder
#import scikit-image
import skimage
import os
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from sklearn.mixture import GaussianMixture
import glob
from skimage.viewer import ImageViewer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
%matplotlib inline
from sklearn.externals import joblib

def Final_pipeline(target_path, caption_flag = False):
    
    """How the user will upload the images.
    moreno note that I have two questions for you above"""
    
    #Create object
    obj_pred = Predicting.Main(target_path, caption_flag)

    df_target = obj_pred.get_target_posts()
    
    if caption_flag:
        #Get embedding for captions
        embedding_captions = obj_lstm.embedd_text(df_target['Caption'])

        #Get Predictions
        prediction = obj_lstm.predict_text(embedding_captions)

        #Combine Dataframe with Text Features
        df_target = obj_lstm.combine_text(df_target, prediction, "LSTM_Feature_")
        display(df_target.head())
    
    #Import Model and Embedding - create image cnn obj
    obj_cnn = Image_CNN.Main('/Users/kimia/Desktop/Capstone/hotel/Text + Image Analysis/Final/KRM_weights-260-0.69.hdf5')
    prediction = obj_cnn.predict_image(df_target['Image'])
    df_target = obj_cnn.combine_image(df_target, prediction, "CNN_Feature_")
    #Load Model
    model_images = obj_pred.load_model("/Users/kimia/Desktop/Capstone/hotel/Text + Image Analysis/Final/model_images.plk")
    model_users = obj_pred.load_model("//Users/kimia/Desktop/Capstone/hotel/Text + Image Analysis/Final/model_users.plk")
    
        #Convert to Features
    if caption_flag:
        extra_cols = ['Caption','File','Image']
    else:
        extra_cols = ['File','Image']

    df_target_presence = obj_pred.get_cluster_presence(df_target, extra_cols, model_images)
    
    extra_cols = ['File','Prediction']
    company_final_df = obj_pred.get_dist2comp(df_target_presence, extra_cols)
    
    