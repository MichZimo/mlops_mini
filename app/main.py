from fastapi import FastAPI
from pydantic import BaseModel # Data validation and parsing lib
import numpy as np
import pandas as pd
import mlflow
from mlflow.pyfunc import load_model
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient


model_name= "house-price-predictor"
stage = "Production"
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model once per startup
    client = MlflowClient()
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=["Production"]
    )

    if not latest_versions:
        raise RuntimeError(f"No Production model found for {model_name}")

    model_version = latest_versions[0].version
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    ml_models["model"] = model
    ml_models["version"] = model_version

    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()

 
app = FastAPI(title = 'House price prediction, California', lifespan=lifespan)


class HouseData(BaseModel):
    med_inc: float 
    longitude: float
    latitude: float

# Rootendpoint
@app.get("/")  
def read_root():
    # Return a simple JSON response
    return {"message": "Welcome to the California House Prediction API"}  

# Check, that API is alive
@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': ml_models["model"] is not None
    }

# Endpoint
@app.post('/predict')
def predict_price(data : HouseData):
    features = pd.DataFrame(
        [
            {
                "MedInc": data.med_inc,
                "Longitude": data.longitude,
                "Latitude": data.latitude
            }
        ]
    )
    price = ml_models["model"].predict(features)
    
    return {'predicted_price': float(price[0]), 'model_version': ml_models["version"]}

class MultipleHouses(BaseModel):
    houses: list[HouseData]

@app.post('/predict_many')
def predict_many_prices(data : MultipleHouses):
    features = pd.DataFrame(
        [
            {
                "MedInc": house.med_inc,
                "Longitude": house.longitude,
                "Latitude": house.latitude
            }
        for house in data.houses]
    )
    prices = ml_models["current_model"].predict(features)
    return {'predicted_prices': prices.tolist(), 'model_version': ml_models["version"]}


'''
Load model only on startup (not import time)
Add /reload-model endpoint
Add version logging in API response
Add CI pipeline trigger for retraining
'''
