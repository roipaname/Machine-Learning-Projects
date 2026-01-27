from database.connection import DatabaseConnection
from datetime import datetime,timedelta
import pandas as pd
from loguru import logger
from typing import Dict,List
from sqlalchemy import func, case


db = DatabaseConnection()

def extract_customer_features(customer_id: str, as_of_date: datetime = None) -> Dict:
    pass