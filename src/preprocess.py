# Data only
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow


def load_data():
    d = fetch_california_housing(as_frame = True)
    df = d.frame
    return df[['MedInc', 'Longitude', 'Latitude', 'MedHouseVal']]

def prepare_features(df):
    X = df[['MedInc', 'Longitude', 'Latitude']] # Features
    y = df['MedHouseVal'] # Targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def create_plot(df, plotting=False, plot_path=None):  # only if needed
    if not plotting:
        return

    if plot_path is None:
        plot_path = Path("data/processed/cali_housing_pairplot.png")
    
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    
    sns.pairplot(df)
    plt.savefig(plot_path)
    plt.close()
    
    mlflow.log_artifact(str(plot_path))
    return
