from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_training.ml.model import load_model, load_process_data_cfg, inference
from model_training.ml.data import process_data

import pandas as pd


MODEL_PATH = 'model/model.pkl'
ENCODER_PATH = 'model/encoder.pkl'
LB_PATH = 'model/lb.pkl'
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model = load_model(MODEL_PATH)
encoder, lb = load_process_data_cfg(ENCODER_PATH, LB_PATH)


class CensusModelInput(BaseModel):
    age: int = Field(..., example=22)
    workclass: str = Field(..., example='Private')
    fnlgt: int = Field(..., example=201490)
    education: str = Field(..., example='HS-grad')
    education_num: int = Field(..., example=9, alias='education-num')
    marital_status: str = Field(...,
                                example='Never-married',
                                alias='marital-status')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Own-child')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., example=0, alias='capital-gain')
    capital_loss: int = Field(..., example=0, alias='capital-loss')
    hours_per_week: int = Field(..., example=20, alias='hours-per-week')
    native_country: str = Field(...,
                                example='United-states',
                                alias='native-country')


app = FastAPI(title="API for the census bureau ML classifier", version="0.1.0")


@app.get("/")
async def welcome_msg():
    return {'message': 'Welcome from census bureau ML classifier'}


@app.post("/prediction")
async def post_prediction(input_data: CensusModelInput):

    input_data_dict = input_data.dict(by_alias=True)
    input_data_pd = pd.DataFrame.from_dict([input_data_dict])

    X, _, _, _ = process_data(X=input_data_pd, categorical_features=CAT_FEATURES,
                              label=None, training=False, encoder=encoder, lb=lb)

    y_pred = inference(model, X)

    if y_pred[0] == 0:
        return {'prediction': '<=50K'}
    elif y_pred[0] == 1:
        return {'prediction': '>50K'}
    else:
        raise HTTPException(status_code=500, detail="Internal error")
