import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_file = os.path.join('models', 'categorical_vectorizers_auto.pkl')
    
class DataTransformation:
    def __init__(self):
        sel.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_features = []
            categorical_features = ['airline', 'ch_code', 'from', 'stop', 'to', 'type', 'dep_time_phase', 'arr_time_phase']
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", LabelEncoder())
                ]
                )
            
            logging.info("Categorical columns: {0}".format(categorical_features))
            
            preprocessor = ColumnTransformer([
                                              ("cat_piplines", cat_pipeline, categorical_features)
                                              ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()
            
            target_col = "price"
            numerical_columns = [""]
            
            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]
            
            logging.info("Applying preprocessing object on train and test data")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_file,
                obj = preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_file
            )
                        
        except Exception as e:
            raise CustomException(e, sys)