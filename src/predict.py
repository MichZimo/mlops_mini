import requests

def predict_price(url, data):
    return requests.post(url, json = data)