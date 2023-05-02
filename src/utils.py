import os 
import sys
from src.exception import CustomException
from src.logger import logging
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_true, y_pred)(y_train, y_train_pred)
            test_model_score = r2_score(y_true, y_pred)(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'wb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def convertToMins(time_taken):
    """
    Takes string time in format "HHh MMm" and converts to total minutes
    e.g. 02h 10m returns 130

    """    
    time_taken = str(time_taken)
    hour, minute = time_taken.split(" ")
    hour = int(float(hour[:-1].replace('',"0")))
    minute = int(float(minute[:-1].replace('',"0")))

    mins = (hour*60)+minute
    return mins
    
def getDayPhase(time):
    """
    Returns phase of the day ('morning', 'afternoon', 'evening' or 'night') based on 24-hour format time.
    E.g. for '22:10', this returns 'night' 
    E.g. for '07:10', this returns 'morning'
    """
    hour, _ = time.split(':')
    hour = int(hour)
    if hour < 6:
        return 'early_morning'
    elif hour < 12:
        return 'morning'
    elif hour < 16:
        return 'afternoon'
    elif hour < 19:
        return 'evening'
    else:
        return 'night'