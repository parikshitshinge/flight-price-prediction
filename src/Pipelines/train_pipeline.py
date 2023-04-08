import os 
import sys 
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object, evaluate_models

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

@dataclass 
class TrainModelConfig:
    trained_model_file_path = os.path.join("models", "model.pkl")
    
class TrainModel:
    def __init__(self):
        self.train_model_config = TrainModelConfig()
        
    def initiate_train_model(self, train_array, test_array):
        try:
            logging.info("Split train and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            
            models = {
                "Linear Regression": LinearRegression(),
                "Rdge" : Ridge(),
                "Lasso" : Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor()
            }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models )
            
            # Get the best model score & name from dict 
            best_mdoel_score = max(sorted(model_report.values()))            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_mdoel_score)]
            best_model = models[best_model_name]
            
            if best_mdoel_score<0.6:
                raise CustomException("No best model found")
            
            save_object(file_path = self.train_model_config.trained_model_file_path, obj=best_model)
            
            y_pred = best_model.predict(X_test)
            
            r2_scr = r2_score(y_test, y_pred)
            
            return r2_scr
            
        except Exception as e:
            raise CustomException(e, sys)