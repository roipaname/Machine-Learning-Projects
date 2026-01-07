import pandas as pd
from config.settings import KAGGEL_DATASET
import logging
logging.basicConfig(level=logging.INFO)
from  typing import List
if not KAGGEL_DATASET.exists():
    raise FileNotFoundError(f"Dataset not found at {KAGGLE_DATASET}")


df=pd.read_csv(KAGGEL_DATASET)

print(df.dtypes)
print(df.info())
print(df.isna().sum())
df=df.drop_duplicates()
df=df.dropna(subset=['url','category'])
df['title']=df['title'].fillna("").astype(str)
df['body']=df['text'].fillna("").astype(str)
df['url']=df['url'].astype(str)
df['source']="kaggle"
df['category']=df['tag'].astype(List)
print(df['category'][0])
df=df.drop(columns=['text','authors','timestamp'])


