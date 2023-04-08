import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "models\model.pkl"
            preprocessor_path = "models\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, airline:str, ch_code:str, from_:str, stop:str, to_:str, type_:str, dep_time_phase:str, arr_time_phase:str):
        self.airline = airline
        self.ch_code = ch_code
        self.from_ = from_ 
        self.stop = stop
        self.to_ = to_
        self.type_ = type_
        self.dep_time_phase = dep_time_phase
        self.arr_time_phase = arr_time_phase
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {'hi':100}
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e, sys)