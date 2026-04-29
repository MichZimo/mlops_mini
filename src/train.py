from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from src.model import create_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from pathlib import Path
from config import BASE_DIR

# Paths
plot_path = BASE_DIR / "data" / "processed" / "cali_housing_pairplot.png"
model_path = BASE_DIR / "models" / "cali_housing_model.joblib"

d = fetch_california_housing(as_frame = True)
df = d.frame
df_pred = df[['MedInc', 'Longitude', 'Latitude', 'MedHouseVal']]
print(df.head())
print(df.dtypes)

# Plot distributions
sns.pairplot(df_pred)
plt.savefig(plot_path)
plt.show()

# How do income and neighborhood predict house prices?
X = df_pred[['MedInc', 'Longitude', 'Latitude']] # Features
y = df_pred['MedHouseVal'] # Targets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
    ) # Split into train and test set
reg = create_model()
reg = reg.fit(X_train, y_train) # Fit model

dump(reg, model_path) # Save model
print('saved')
