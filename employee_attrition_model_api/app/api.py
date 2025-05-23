import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from employee_attrition_model import __version__ as model_version
from employee_attrition_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {   
            'age': '45',
            'attrition': 'Yes',
            'businesstravel': 'Travel_Rarely',
            'dailyrate': '1100',
            'department': 'Sales',
            'distancefromhome': '2',
            'education': '1',
            'educationfield': 'Life Sciences',
            'employeecount': '1',
            'employeenumber': '49',
            'environmentsatisfaction': '4',
            'gender': 'Male',
            'hourlyrate': '90',
            'jobinvolvement': '4',
            'joblevel': '3',
            'jobrole': 'Research Scientist',
            'jobsatisfaction': '2',
            'maritalstatus': 'Married',
            'monthlyincome': '6000',
            'monthlyrate': '24000',
            'numcompaniesworked': '4',
            'over18':'Y',
            'overtime':'No',
            'percentsalaryhike':'5',
            'performancerating':'8',
            'relationshipsatisfaction':'4',
            'standardhours':'8',
            'stockoptionlevel':'2',
            'totalworkingyears':'10',
            'trainingtimeslastyear':'3',
            'worklifebalance':'5',
            'yearsatcompany':'2',
            'yearsincurrentrole':'2',
            'yearssincelastpromotion':'2',
            'yearswithcurrmanager':'2'
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    attrition predictions with the employee_attrition_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

