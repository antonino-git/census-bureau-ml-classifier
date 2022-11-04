from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd

class CensusModelInput(BaseModel):
    age: int = Field(..., example=22)
    workclass: str = Field(..., example='Private')
    fnlgt: int = Field(..., example=201490)
    education: str = Field(..., example='HS-grad')
    education_num:int = Field(..., example=9, alias='education-num')
    marital_status: str = Field(..., example='Never-married', alias='marital-status')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Own-child')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., example=0, alias='capital-gain')
    capital_loss: int = Field(..., example=0, alias='capital-loss')
    hours_per_week:int = Field(..., example=20, alias='hours-per-week')
    native_country: str = Field(..., example='United-states', alias='native-country')

app = FastAPI(title="API for the census bureau ML classifier", version="0.1.0")

@app.get("/")
async def welcome_msg():
    return{'message':'Welcome from census bureau ML classifier'}

@app.post("/prediction")
async def post_prediction(input_data: CensusModelInput):

    input_data_dict = input_data.dict(by_alias=True)
    input_data_pd = pd.DataFrame.from_dict([input_data_dict])

    print(input_data_pd)

    return{'prediction':0}