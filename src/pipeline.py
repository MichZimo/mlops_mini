# Orchestration
from mlflow.tracking import MlflowClient
from src.preprocess import load_data, prepare_features, create_plot
from src.train import log_to_mlflow
from src.registry import get_best_candidate, promote_best

def main():    
    model_name= "house-price-predictor"
    exp_name = "Multiple_Linear_Regression_Project" # Containername for runs
    run_name = "baseline-linear-regression"

    client = MlflowClient()
    
    # Preprocess data and prepare features
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_features(df)
    plot_path = create_plot(df)
        
    # Train and log to Mlflow
    run_id = log_to_mlflow(client, model_name, exp_name, run_name,
                                     X_train, X_test, y_train, y_test, plot_path)
    
    
    # Selection
    best_candidate_version, best_candidate_run = get_best_candidate(client, model_name, exp_name)
    promote_best(model_name, best_candidate_run, best_candidate_version, client)
