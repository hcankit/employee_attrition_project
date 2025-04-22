import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from employee_attrition_model import __version__ as _version
from employee_attrition_model.config.core import config
from employee_attrition_model.pipeline import attrition_pipe
from employee_attrition_model.processing.data_manager import load_pipeline
from employee_attrition_model.processing.data_manager import pre_pipeline_preparation
from employee_attrition_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
attrition_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = attrition_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = attrition_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":

    data_in={
            'age': ['45'],
            'attrition': ['Yes'],
            'businesstravel': ['Travel_Rarely'],
            'dailyrate': ['1100'],
            'department': ['Sales'],
            'distancefromhome': ['2'],
            'education': ['1'],
            'educationfield': ['Life Sciences'],
            'employeecount': ['1'],
            'employeenumber':['49'],
            'environmentsatisfaction': ['4'],
            'gender': ['Male'],
            'hourlyrate': ['90'],
            'jobinvolvement': ['4'],
            'joblevel': ['3'],
            'jobrole': ['Research Scientist'],
            'jobsatisfaction': ['2'],
            'maritalstatus': ['Married'],
            'monthlyincome': ['6000'],
            'monthlyrate': ['24000'],
            'numcompaniesworked': ['4'],
            'over18':['Y'],
            'overtime':['No'],
            'percentsalaryhike':['5'],
            'performancerating':['8'],
            'relationshipsatisfaction':['4'],
            'standardhours':['8'],
            'stockoptionlevel':['2'],
            'totalworkingyears':['10'],
            'trainingtimeslastyear':['3'],
            'worklifebalance':['5'],
            'yearsatcompany':['2'],
            'yearsincurrentrole':['2'],
            'yearssincelastpromotion':['2'],
            'yearswithcurrmanager':['2']
            }
    make_prediction(input_data=data_in)