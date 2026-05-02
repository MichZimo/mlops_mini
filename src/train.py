# Responsibility: training + logging

import pandas as pd
import numpy as np
from src.model import create_model
from src.evaluate import adjusted_r2, mse
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import time

def train_model(X_train, y_train): 
    model = create_model()
    return model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Evaluate model
    r2_value = r2_score(y_test, y_pred)
    adjusted_r2_score = adjusted_r2(X_test, y_test, y_pred)
    mse_score = mse(y_test, y_pred)
    print(f"R2: {r2_value}, Adjusted R2: {adjusted_r2_score}, \n MSE: {mse_score}")
    return y_pred, r2_value, adjusted_r2_score, mse_score

def log_to_mlflow(run_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name) as run: 
        model = train_model(X_train, y_train)
        y_pred, r2_value, adjusted_r2_score, mse_score = evaluate_model(model, X_test, y_test)

        signature = infer_signature(X_test, y_pred) # Signature
        # Log custom metrics 
        mlflow.log_metric("r2", r2_value)
        mlflow.log_metric("adjusted_r2", adjusted_r2_score)
        mlflow.log_metric("mse", mse_score)

        # Log artifact 
        mlflow.log_artifact(str(plot_path))

        # Log the model artifact and register it under a global registry name
        model_info = mlflow.sklearn.log_model(
            sk_model=reg,
            name="sklearn-model",
            signature=signature,
            registered_model_name=model_name, 
        )

        # Retry loop to check whether MLFlow is consistent with version naming
        current_version = None
        for _ in range(5):
            model_versions = client.search_model_versions(f"name='{model_name}'")
            for mv in model_versions:
                if mv.run_id == run.info.run_id:
                    current_version = mv.version
                    break
            if current_version:
                break
            time.sleep(1)

        if current_version is None:
            raise RuntimeError("Model version not found after logging")
        
        mlflow.set_tag("model_version", current_version)
    return

def train_and_log(X_train, y_train, X_test, y_test, run_name):
    model = train_model(X_train, y_train)
    r2_value, adjusted_r2, mse_score = evaluate_model(model, X_test, y_test)
    run_id = log_to_mlflow(run_name)    
    return run_id
#########

def decide_production_run(model_name, exp_name, client):
    ## Logics
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        raise ValueError("Experiment not found")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
        )
    valid = runs[runs["metrics.adjusted_r2"] > 0]
    if valid.empty:
        raise ValueError("No valid runs found")
    best_candidate_run = valid.sort_values(by="metrics.mse", ascending=True).iloc[0]
    best_candidate_id = best_candidate_run["run_id"]

    if best_candidate_run.empty:
        raise ValueError("No valid runs found")
    
    ##

    # Get all versions of your registered model
    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    # Find the version created by best run
    best_candidate_version = None
    for mv in model_versions:
        if mv.run_id == best_candidate_id:
            best_candidate_version = mv.version
            break
    if best_candidate_version is None:
        raise ValueError("No model version found for best run")

    print(f"Best model candidate run: {best_candidate_version}")
    return best_candidate_run, best_candidate_version

# Compare best candidate against current production version
def promote_best(model_name, best_candidate_run, best_candidate_version, client):
    try:
        prod_model = client.get_latest_versions(model_name, stages=["Production"])[0]
        prod_run = client.get_run(prod_model.run_id)
        prod_adj_r2 = prod_run.data.metrics.get('adjusted_r2', -np.inf)
        prod_mse = prod_run.data.metrics.get('mse', np.inf)
        print(f"Current Production model: ({prod_model.version}) "
            f"adj_r2={prod_adj_r2}, mse={prod_mse}")

    except:
        best_version = best_candidate_version
        client.transition_model_version_stage(
            name=model_name, 
            version=best_version, 
            stage="Production", 
            archive_existing_versions=True
        )
        return
    if best_candidate_run['metrics.mse'] < prod_mse:
        best_version = best_candidate_version
        client.transition_model_version_stage(
            name=model_name, 
            version=best_version, 
            stage="Production", 
            archive_existing_versions=True
            )
        return
    else: 
        return


if __name__ == "__main__":
    model_name = "sk-learn-multiple-lin-reg-model" # Model name!
    exp_name = "Multiple_Linear_Regression_Project" # Containername for runs
    plot_path = BASE_DIR / "data" / "processed" / "cali_housing_pairplot.png"

    mlflow.set_experiment(exp_name) 
    client = MlflowClient()
    
     = train(df_pred)
    mlflow_exp(X_train, X_test, y_train, y_test, plot_path, client, model_name)
    best_candidate_run, best_candidate_version = decide_production_run(model_name, exp_name, client)
    promote_best(model_name, best_candidate_run, best_candidate_version, client)


## Necessary changes to MLFlow: promote only if new version better than current
## Next step: FastAPI consumes Production model
## Stretch goal: CI/ CD with Docker, then retrainon new data