# Responsibility: training + logging
from src.model import create_model
from src.evaluate import adjusted_r2, mse
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
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

def log_to_mlflow(client, model_name, exp_name, run_name, X_train, X_test, y_train, y_test, plot_path):
    mlflow.set_experiment(exp_name) 

    with mlflow.start_run(run_name = run_name) as run: 
        model = train_model(X_train, y_train)
        y_pred, r2_value, adjusted_r2_score, mse_score = evaluate_model(model, X_test, y_test)

        signature = infer_signature(X_test, y_pred) # Signature

        # Log parameters
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("features", ["MedInc", "Longitude", "Latitude"])
        
        # Log custom metrics 
        mlflow.log_metric("r2", r2_value)
        mlflow.log_metric("adjusted_r2", adjusted_r2_score)
        mlflow.log_metric("mse", mse_score)
        if plot_path is not None:
            mlflow.log_artifact(str(plot_path))        

        # Log the model artifact and register it under a global registry name
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
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

    return run.info.run_id, current_version




## Stretch goal: CI/ CD with Docker, then retrainon new data