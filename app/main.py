from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel # Data validation and parsing lib
import numpy as np
from config import MODEL_PATH

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
    features = np.array([[data.med_inc, data.longitude, data.latitude]])
    price = model.predict(features)
    return {'predicted_price': float(price[0])}

class MultipleHouses(BaseModel):
    houses: list[HouseData]

@app.post('/predict_many')
def predict_many_prices(data : MultipleHouses):
    features = np.array([[house.med_inc, house.longitude, house.latitude] for house in data.houses])
    prices = model.predict(features)
    return {'predicted_prices': prices.tolist()}