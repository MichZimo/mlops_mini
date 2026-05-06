# Responsibility: model lifecycle (MLflow)
import mlflow
import numpy as np
from src.selection import get_best_run, get_best_version
    
def get_best_candidate(client, model_name, exp_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        raise ValueError("Experiment not found")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
        )
    best_candidate_id, best_candidate_run = get_best_run(runs)
    # Get all versions of your registered model
    model_versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    best_candidate_version = get_best_version(model_versions, best_candidate_id)
    return best_candidate_version, best_candidate_run

def promote_best(model_name, best_candidate_run, best_candidate_version, client):
    try:
        prod_model = client.get_latest_versions(model_name, stages=["Production"])[0]
        prod_run = client.get_run(prod_model.run_id)
        prod_adj_r2 = prod_run.data.metrics.get('adjusted_r2', -np.inf)
        prod_mse = prod_run.data.metrics.get('mse', np.inf)
        print(f"Current Production model: ({prod_model.version}) "
            f"adj_r2={prod_adj_r2}, mse={prod_mse}")

    except IndexError:
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
