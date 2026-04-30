import pytest
import requests
from app.main import app
from fastapi.testclient import TestClient
import pandas as pd

client = TestClient(app)
endp = '/predict'  
endp_batch = '/predict_many' 
endp_h = '/health'


def predict_price(endpoint, data):
    return client.post(endpoint, json = data) 
    #instead of requests.post(url, json = data)

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

def test_health():
    response = client.get(endp_h)
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_single_pred(new_data):
    response = predict_price(endp, new_data)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float) 

def test_batch_pred(many_houses):
    responses = predict_price(endp_batch, many_houses)
    assert responses.status_code == 200
    assert isinstance(responses.json()["predicted_prices"], list)

def test_fail_single():
    response = predict_price(endp, {})
    assert response.status_code == 422

def test_fail_batch():
    responses = predict_price(endp_batch, {})
    assert responses.status_code == 422