import Predicting

target_path = "/Users/kmotwani/Desktop/Me/Education/Courses/Capstone Project/Target/"
caption_flag = False

#Create object
obj_pred = Predicting.Main(target_path, caption_flag)
class Brands_Similarity:

    def __init__(self):
        #Define target path and create test object
        target_path = "/Users/kmotwani/Desktop/Me/Education/Courses/Capstone Project/Target/"
        caption_flag = True

        #Create object
        self.obj_pred = Predicting.Main(target_path, caption_flag)

        #Load Model
        self.model_images = self.obj_pred.load_model("/Users/kmotwani/Desktop/Me/Education/Courses/Capstone Project/model_images.plk")
        self.model_users = self.obj_pred.load_model("/Users/kmotwani/Desktop/Me/Education/Courses/Capstone Project/model_users.plk")


        #Get Target Images
        df_target = obj_pred.get_target_posts()

        display(df_target.head())

        if caption_flag:
            #Get embedding for captions
            embedding_captions = obj_lstm.embedd_text(df_target['Caption'])

            #Get Predictions
            prediction = obj_lstm.predict_text(embedding_captions)

            #Combine Dataframe with Text Features
            df_target = obj_lstm.combine_text(df_target, prediction, "LSTM_Feature_")
            display(df_target.head())


#Get embedding for captions
prediction = obj_cnn.predict_image(df_target['Image'])

#Combine Prediction
df_target = obj_cnn.combine_image(df_target, prediction, "CNN_Feature_")
display(df_target.head())

#Convert to Features
if caption_flag:
    extra_cols = ['Caption','File','Image']
else:
    extra_cols = ['File','Image']
df_target_presence = obj_pred.get_cluster_presence(df_target, extra_cols, model_images)
display(df_target_presence.head())


#Get Company Distances
extra_cols = ['File','Prediction']
company_final_df = obj_pred.get_dist2comp(df_target_presence, extra_cols)
display(company_final_df.head())
