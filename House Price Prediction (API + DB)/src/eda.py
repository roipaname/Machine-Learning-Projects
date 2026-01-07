import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
sns.set_style('whitegrid')
df=pd.read_csv(os.getenv("PATH_TO_DATASET"))
#Basic Information
print('\n'+'='*50)
print("Dataset Shape:", df.shape)
print('\n'+'='*50)
print(df.info())
print('\n'+'='*50)
print("First 5 Rows of the Dataset:")
print(df.head())


#Checking for missing values
missing=df.isnull().sum()
missing_pct=(missing/len(df))*100
missing_df=pd.DataFrame({
    "Missing_Count":missing,
    "Missing_Percentage":missing_pct
}).sort_values(by="Missing_Count",ascending=False)
print('\n'+'='*50)
print("Missing Values in Each Column:")
print(missing_df[missing_df["Missing_Count"]>0])

#Checking Unique Values for Categorical Features
cat_cols=df.select_dtypes(include=['object']).columns
print('\n'+'='*50)
print("Unique Values in Categorical Columns:")
for col in cat_cols:
    unique_vals=df[col].nunique()
    print(f"Unique values in {col}: {unique_vals}")


#Price Distribution
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.hist(df['price'].dropna(),bins=30,color='blue',edgecolor='gold')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')

plt.subplot(1,3,2)
plt.hist(np.log1p(df['price'].dropna()),bins=30,color='green',edgecolor='gold')
plt.xlabel('Log(Price + 1)')
plt.ylabel('Frequency')
plt.title('Log-Transformed Price Distribution')

plt.subplot(1,3,3)
sns.boxplot(df['price'].dropna(),color='orange')
plt.xlabel('Price')
plt.title('Price Boxplot')

plt.tight_layout()
plt.show()


#Correlations
num_cols=df.select_dtypes(include=[np.number]).columns
corr_matrix=df[num_cols].corr()
print(corr_matrix['price'].sort_values(ascending=False))
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',center=0)
plt.title('Correlation Matrix')
plt.show()

#Distribution of properties by state
plt.figure(figsize=(12,6))
state_counts=df['state'].value_counts().head(20)
state_counts.plot(kind='bar',color='purple',edgecolor='black')
plt.xlabel('State')
plt.ylabel('Number of Properties')
plt.xticks(rotation=45)
plt.title('Top 20 States by Number of Properties')
plt.show()

plt.figure(figsize=(12,6))
df.groupby('bed')['price'].median().plot(kind='bar',color='teal',edgecolor='black')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Median Price')
plt.title('Median Price by Number of Bedrooms')
plt.show()
