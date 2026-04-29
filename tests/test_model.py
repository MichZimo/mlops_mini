import requests
from config import BASE_URL

url = BASE_URL + '/predict' 
new_data = {'med_inc': 20000, 'longitude': 118.2426, 'latitude': 34.0549} # LA
response = requests.post(url, json = new_data)

#print(response.json()['predicted_price'])

# Batch test

url_batch = BASE_URL + '/predict_many'

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