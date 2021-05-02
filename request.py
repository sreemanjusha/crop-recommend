import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={'N':90, 'P':49, 'K':21, 'temperature':24.84, 'humidity':68.35, 'ph':6.47, 'rainfall':74.05})

print(r.json())

