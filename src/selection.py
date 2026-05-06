# Responsibility: choose best run (No mlflow transition here)

def get_best_run(runs):#(model_name, exp_name, client):
    ## Logics
    valid = runs[runs["metrics.adjusted_r2"] > 0]
    if valid.empty:
        raise ValueError("No valid runs found")
    best_candidate_run = valid.sort_values(by="metrics.mse", ascending=True).iloc[0]
    best_candidate_id = best_candidate_run["run_id"]

    if best_candidate_run.empty:
        raise ValueError("No valid runs found")
    
    return best_candidate_id, best_candidate_run

def get_best_version(model_versions, best_candidate_id):
     # Find the version created by best run
    best_candidate_version = None
    for mv in model_versions:
        if mv.run_id == best_candidate_id:
            best_candidate_version = mv.version
            break
    if best_candidate_version is None:
        raise ValueError("No model version found for best run")

    print(f"Best model candidate run: {best_candidate_version}")

    return  best_candidate_version