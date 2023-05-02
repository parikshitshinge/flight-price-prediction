import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = "./models/model.pkl"            
            vectorizer_path = "./models/vectorizer.pkl"
            model = load_object(file_path=model_path)
            logging.info("model.pkl is loaded")
            vectorizer = load_object(file_path=vectorizer_path)
            logging.info("vectorizer.pkl is loaded")
            vectorized_data = vectorizer.transform(features)
            logging.info("Query point is vectorized")
            pred = model.predict(vectorized_data)
            logging.info("Successfully predicted")
            return pred
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, airline:str, from_:str, to_:str, type_:str, dep_time_phase:str, arr_time_phase:str):
        self.airline = airline
        self.from_ = from_ 
        self.to_ = to_
        self.type_ = type_
        self.dep_time_phase = dep_time_phase
        self.arr_time_phase = arr_time_phase
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                            "airline" : [self.airline],
                            "from_" : [self.from_],
                            "to_" : [self.to_],
                            "type_" : [self.type_], 
                            "dep_time_phase" : [self.dep_time_phase],
                            "arr_time_phase" : [self.arr_time_phase]
            }
            logging.info("Returning query point as DataFrame")
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e, sys)