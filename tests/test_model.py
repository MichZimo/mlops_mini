#import requests
from config import BASE_URL
import pytest
from src.predict import predict_price

url = BASE_URL + '/predict' 
url_batch = BASE_URL + '/predict_many'

@pytest.fixture
def new_data():
    return {'med_inc': 20000, 'longitude': 118.2426, 'latitude': 34.0549} # LA

@pytest.fixture
def many_houses(new_data):
    return {
        'houses':[
            new_data,
            {'med_inc': 20000, 'longitude': 122.2730, 'latitude': 37.8715}, # Berkeley
            {'med_inc': 20000, 'longitude': 122.4194, 'latitude': 37.7749} # SF
            ]
        }

def test_single_pred(new_data):
    response = predict_price(url, new_data)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_batch_pred(many_houses):
    responses = predict_price(url_batch, many_houses)
    assert responses.status_code == 200
    assert isinstance(responses.json()["predicted_prices"], list)
