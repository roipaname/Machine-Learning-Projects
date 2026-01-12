from config.settings import KAGGLE_TRAIN_DATASET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv(KAGGLE_TRAIN_DATASET)
print(df.info())