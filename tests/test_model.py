import requests
from dotenv import load_dotenv 
import os 
load_dotenv() # This loads variables from .env into os.environ 

api_url = os.getenv("BASE_URL") 
debug = os.getenv("DEBUG", "False") # Disable debug mode



url = api_url + '/predict' 

new_data = {'med_inc': 20000, 'longitude': 118.2426, 'latitude': 34.0549} # LA

response = requests.post(url, json = new_data)

#print(response.json()['predicted_price'])

# Batch test

url_batch = api_url + '/predict_many'

many_houses = {
    'houses':[
        {'med_inc': 20000, 'longitude': 118.2426, 'latitude': 34.0549}, # LA
        {'med_inc': 20000, 'longitude': 122.2730, 'latitude': 37.8715}, # Berkeley
        {'med_inc': 20000, 'longitude': 122.4194, 'latitude': 37.7749} # SF
    ]
}


responses = requests.post(url_batch, json = many_houses)

#print(responses.json()['predicted_prices'])
# automated tests with pytest and assert