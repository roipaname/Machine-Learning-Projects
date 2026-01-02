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

#Checking for missing values
missing=df.isnull().sum()
missing_pct=(missing/ len(df))*100

missing_df=pd.DataFrame(
    {
        'missing_count':missing,
        'missing_percentage':missing_pct
    }
).sort_values(by='missing_count', ascending=False)


print("\n"+"="*50)
#Basic stats
print("Numerical Statistics:")
print(df.describe())

#Checking for Categorical columns
categorical_cols=df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"Number of unique values in {col}: {df[col].nunique()}")

#Price Distribution
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.hist(df['price'].dropna(),bins=50,edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Distribution')

plt.subplot(1,3,2)
plt.hist(np.log1p(df['price'].dropna()),bins=50,edgecolor='black', color='green')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.title('Log Price Distribution')

plt.subplot(1,3,3)
plt.boxplot(df['price'].dropna())
plt.ylabel('Price')
plt.title('Price Boxplot')

plt.tight_layout()
plt.show()
#Correlation
numeric_cols=df.select_dtypes(include=['float','int']).columns
corr_matrix=df[numeric_cols].corr()
print("\n"+"="*50)
print("Correlation Matrix:")
print(corr_matrix)
print("\n"+"="*50)
print(corr_matrix['price'].sort_values(ascending=False))


plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",center=0)
plt.title('Correlation Heatmap')
plt.show()


#Dsitribution of properties by State
plt.figure(figsize=(12,6))
state_counts=df['state'].value_counts().head(20)
print(state_counts)
state_counts.plot(kind='bar')
plt.xlabel('State')
plt.ylabel('Number of Properties')
plt.title('Top 20 States by Number of Properties')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price  by number of bedrooms
plt.figure(figsize=(10,6))
df.groupby('bed')['price'].median().plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Median Price')
plt.title('Median Price by Number of Bedrooms')
plt.tight_layout()
plt.show()