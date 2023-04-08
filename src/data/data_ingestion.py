import os 
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # Configure connection from data source
    raw_biz_data_path: str = os.path.join('data/raw', 'business.csv')
    raw_eco_data_path: str = os.path.join('data/raw', 'economy.csv')
    
    # Define destination path to load source data
    biz_train_data_path: str = os.path.join('data/interim', 'biz_train.csv')
    eco_train_data_path: str = os.path.join('data/interim', 'eco_train.csv')

    biz_test_data_path: str = os.path.join('data/interim', 'biz_test.csv')
    eco_test_data_path: str = os.path.join('data/interim', 'eco_test.csv')

    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Read data from source database
            business = pd.read_csv("./data/raw/business.csv")
            logging.info("Read the business.csv dataset as dataframe")
            economy = pd.read_csv("./data/raw/economy.csv")
            logging.info("Read the economy.csv dataset as dataframe") 
            
            os.makedirs(os.path.dirname(self.ingestion_config.biz_train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.eco_train_data_path), exist_ok=True)
            
            # Ingest data into our raw folder as it is from source
            business.to_csv(self.ingestion_config.raw_biz_data_path, index=False, header=True)
            economy.to_csv(self.ingestion_config.raw_eco_data_path, index=False, header=True)
            
            # Split the data into train and test
            logging.info("Train test split initiated")
            biz_train_set, biz_test_set = train_test_split(business, test_size = 0.2, random_state = 15)
            eco_train_set, eco_test_set = train_test_split(economy, test_size = 0.2, random_state = 15)
            
            biz_train_set.to_csv(self.ingestion_config.biz_train_data_path, index=False, header=True)
            eco_train_set.to_csv(self.ingestion_config.eco_train_data_path, index=False, header=True)
            
            biz_test_set.to_csv(self.ingestion_config.biz_test_data_path, index=False, header=True)
            eco_test_set.to_csv(self.ingestion_config.eco_test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.biz_train_data_path,
                self.ingestion_config.eco_train_data_path,
                self.ingestion_config.biz_test_data_path,
                self.ingestion_config.eco_test_data_path
            )
                       
        except Exception as e:
            raise CustomException(e, sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()