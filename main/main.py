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
