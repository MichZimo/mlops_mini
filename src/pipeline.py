# Orchestration
from src.preprocess import load_data, prepare_features, create_plot
from src.train import train_and_log

def main():    
    # Preprocess data and prepare features
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_features(df)
    create_plot(df)
    
    # Train and log to Mlflow
    run_id = train_and_log(run_name = "Baseline-multiple-linear-regression")

    best_run = get_best_run(...)
    promote_if_better(...)