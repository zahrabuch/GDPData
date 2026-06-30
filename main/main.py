import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import kagglehub
from kagglehub import KaggleDatasetAdapter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = 'country_data.csv'

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "arslaan5/global-data-gdp-life-expectancy-and-more",
  file_path,
)

print("First 5 records:", df.head())
print(df.describe())


names = ["name", "region", "capital", "currency", "iso2"]

df_num = df.drop(columns=names)
"""
corrmat = df_num.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
"""
data = df_num[['gdp', 'imports', 'exports', 'tourists', 'surface_area']]
print(data.isna().sum())
data = data.dropna(subset=['gdp'])

cols = ['imports', 'exports', 'tourists', 'surface_area']

for col in cols:
    data[col] = data[col].fillna(df[col].median())

print(data.isna().sum())
"""
corrmat = train.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
"""
y = np.log(data['gdp'])
x = data[['imports', 'exports', 'tourists', 'surface_area']]

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, predictions)
print(score)
