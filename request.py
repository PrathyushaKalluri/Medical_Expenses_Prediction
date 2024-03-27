import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json= {'age' : 40,
        'sex' : 1,
        'bmi' : 45.50,
        'children' : 4,
        'smoker' : 1,
        'region' : 3})

print(r.json())