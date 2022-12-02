from fastapi.testclient import TestClient

# Import our app from main.py
from main import app

# Create an instance of TestClient for our app.
client = TestClient(app)


def test_api_get_root():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['message'] == 'Welcome from census bureau ML classifier'


def test_api_post_prediction_low():
    input_data = {
        "age": 59,
        "workclass": "Private",
        "fnlgt": 109015,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    expected_result = '<=50K'

    r = client.post('/prediction', json=input_data)
    assert r.status_code == 200
    assert r.json()['prediction'] == expected_result


def test_api_post_prediction_high():
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

    expected_result = '>50K'

    r = client.post('/prediction', json=input_data)
    assert r.status_code == 200
    assert r.json()['prediction'] == expected_result


def test_api_wrong_get():

    r = client.get('/test')

    assert r.status_code == 404
