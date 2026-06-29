import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = 'country_data.csv'

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "arslaan5/global-data-gdp-life-expectancy-and-more",
  file_path,
)

print("First 5 records:", df.head())
print(df.describe())


names = ["name", "region", "capital", "currency", "iso2"]

df = df.drop(columns=names)
corrmat = df.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
