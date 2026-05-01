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
from src.evaluate import adjusted_r2, mse
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Multiple_Linear_Regression_Project")
mlflow.sklearn.autolog() # Keep for learning MLFlow; REMOVE for production
 
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
plt.close() # instead of plt.show() to free up memory and not block CI and pipelines

# How do income and neighborhood predict house prices?
X = df_pred[['MedInc', 'Longitude', 'Latitude']] # Features
y = df_pred['MedHouseVal'] # Targets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
    ) # Split into train and test set

with mlflow.start_run(run_name = "Baseline-multiple-linear-regression"):
    reg = create_model()
    reg = reg.fit(X_train, y_train) # Fit model

    # Make prediction
    y_pred = reg.predict(X_test)

    # Evaluate model
    r2_value = r2_score(y_test, y_pred)
    adjusted_r2_score = adjusted_r2(X_test, y_test, y_pred)
    mse_score = mse(y_test, y_pred)
    print(f"R2: {r2_value}, Adjusted R2: {adjusted_r2_score}, \n MSE: {mse_score}")

    # Log custom metrics 
    mlflow.log_metric("adjusted_r2", adjusted_r2_score)
    mlflow.log_metric("mse", mse_score)

    # Log artifact 
    mlflow.log_artifact(str(plot_path))

    dump(reg, model_path) # Save model
    print('saved')

