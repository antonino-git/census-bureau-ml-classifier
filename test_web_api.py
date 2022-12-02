import requests

input_data = {
    "age": 76,
    "workclass": "Private",
    "fnlgt": 124191,
    "education": "Master",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

response = requests.post(
    "https://mlops-api-test.herokuapp.com/prediction",
    json=input_data)
print(f'Response status code: {response.status_code}')
print(f'Response body: {response.json()}')
