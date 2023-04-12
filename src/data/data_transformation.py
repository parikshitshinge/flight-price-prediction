import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import getDayPhase, convertToMins, save_object

from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack

@dataclass
class DataTransformationConfig:
    preprocessed_data_file = os.path.join('./data/processed' , 'processed.csv')
    vectorizer_obj_file = os.path.join('models', 'vectorizer.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessed_data(self, raw_biz_data_path, raw_eco_data_path):
        """
        This function is responsible for preprocessing data
        """
        try:
           # Merge business and economy files
            raw_biz_df = pd.read_csv(raw_biz_data_path)
            raw_eco_df = pd.read_csv(raw_eco_data_path)
            logging.info('Read business and economy data completed')
            
            raw_biz_df['type'] = 'Business'
            raw_eco_df['type'] = 'Economy'

            raw_data = pd.concat([raw_biz_df, raw_eco_df], ignore_index=True)
            logging.info('Merging of business and economy data completed')
            
            logging.info('Data transformation started')
            # Convert 'date' to datetime type
            raw_data['date'] = pd.to_datetime(raw_data['date'], format="%d-%m-%Y")
            logging.info("Converted 'date' column to datetime")
            
            # Remove comma from 'price' column
            raw_data['price'] = pd.to_numeric(raw_data['price'].apply(lambda x: x.replace(",","")))
            logging.info("Removed comma from 'price' column")
            
            # Remove garbage characters from 'stop' column
            raw_data['stop'] = raw_data['stop'].apply(lambda x: '1-stop' if x.startswith('1-stop') else x )
            logging.info("Removed garbage characters from 'stop' column")
            
            # Convert time taken from HHh:MMm format to total minutes
            raw_data['time_taken'] = raw_data['time_taken'].apply(convertToMins)
            logging.info("Converted 'time_taken' column to minutes")
            
            # Compute 'days_left' based on static logic
            raw_data['days_left'] = np.where(raw_data['date'].dt.strftime('%m').astype(int) > 2, raw_data['date'].dt.strftime('%d').astype(int)+18, np.where(raw_data['date'].dt.strftime('%m').astype(int) == 2, raw_data['date'].dt.strftime('%d').astype(int)-10, raw_data['date'].dt.strftime('%d').astype(int)))
            logging.info("Computed 'days_left' column")
                    
            # Compute phase of the day for both 'dep_time' and 'arr_time'
            raw_data['dep_time_phase'] = raw_data['dep_time'].apply(getDayPhase)
            raw_data['arr_time_phase'] = raw_data['arr_time'].apply(getDayPhase)
            logging.info("Computed 'dep_time_phase' and 'arr_time_phase'")
            
            # Dropping 'date', 'dep_time' and 'arr_time' columns"
            raw_data.drop(['date', 'dep_time', 'arr_time'], axis=1, inplace=True)
            logging.info("Dropped 'date', 'dep_time' and 'arr_time' columns")
            logging.info("Preprocessing is completed")
            

            save_object(
                file_path = self.data_transformation_config.preprocessed_data_file,
                obj = raw_data
            )
                        
            logging.info("Preprocessed data is saved")
            
            return (
                raw_data
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
            
    def get_vectorizer_object(self):
        '''
        This function is responsible to return column vectorizer object
        '''
        try:
            numerical_features = ['days_left']
            categorical_features = ['airline', 'from', 'to', 'type', 'dep_time_phase', 'arr_time_phase']
            
            cat_pipeline = Pipeline(
                steps = [
                    ("ohe", OneHotEncoder())
                ]
                )
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                ]
                )
            
            logging.info("Categorical columns: {0}".format(categorical_features))
            logging.info("Numerical columns: {0}".format(numerical_features))
            
            vectorizer = ColumnTransformer([
                                              ("cat_piplines", cat_pipeline, categorical_features),
                                              ("num_piplines", num_pipeline, numerical_features)
                                              ])
            
            return vectorizer
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, raw_biz_data_path, raw_eco_data_path):
        try:
            logging.info('Obtaining preprocessed data')
            
            preprocessed_data = self.get_preprocessed_data(raw_biz_data_path, raw_eco_data_path) 
 
            logging.info('Obtained processed data')
            
            logging.info('Obtaining vectorizer object')

            vectorizer_obj = self.get_vectorizer_object()

            logging.info("Applying vectorizer object on processed data")
            
            numerical_features = ['days_left']
            categorical_features = ['airline', 'from', 'to', 'type', 'dep_time_phase', 'arr_time_phase']
            
            preprocessed_data_vectorized = vectorizer_obj.fit_transform(preprocessed_data)
          
            preprocessed_data_vectorized = hstack((preprocessed_data_vectorized, preprocessed_data['price'].values.reshape(-1,1) )).tocsr()
            
            preprocessed_data_vectorized_arr = preprocessed_data_vectorized.toarray()
            
            logging.info("Vectorization is completed")
            
            save_object(
                file_path = self.data_transformation_config.vectorizer_obj_file,
                obj = vectorizer_obj
            )
            
            logging.info("Saved vectorizer object")
                        
            return (
                preprocessed_data_vectorized_arr,
                self.data_transformation_config.vectorizer_obj_file
            )
                        
        except Exception as e:
            raise CustomException(e, sys)
        