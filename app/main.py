from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel # Data validation and parsing lib
import numpy as np
from config import MODEL_PATH
import pandas as pd
import mlflow

# load model
model = load(MODEL_PATH) 
app = FastAPI(title = 'House price prediction, California')

class HouseData(BaseModel):
    med_inc: float 
    longitude: float
    latitude: float

# Check, that API is alive
@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None
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
    price = model.predict(features)
    return {'predicted_price': float(price[0])}

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
    prices = model.predict(features)
    return {'predicted_prices': prices.tolist()}