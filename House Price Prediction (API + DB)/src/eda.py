import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

df = pd.read_csv(os.getenv("PATH_TO_DATASET"))
print("Dataset loaded successfully.")
print("DATATFRAME SHAPE:",df.shape)
print("\n"+"="*50)
print("Column Information:")
print(df.info())
print("\n"+"="*50)
print("First 5 Rows of the Dataset:")
print(df.head())