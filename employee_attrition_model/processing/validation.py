import sys
from datetime import datetime
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from employee_attrition_model.config.core import config
from employee_attrition_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(df=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    age: Optional[int]
    attrition: Optional[str]
    businesstravel: Optional[str]
    dailyrate: Optional[str]
    department: Optional[str]
    distancefromhome: Optional[str]
    education: Optional[str]
    educationfield: Optional[float]
    employeecount: Optional[float]
    employeenumber: Optional[float]
    environmentsatisfaction: Optional[float]
    gender: Optional[int]
    hourlyrate: Optional[int]
    jobinvolvement: Optional[str]
    joblevel: Optional[str]
    jobrole: Optional[float]
    jobsatisfaction: Optional[float]
    maritalstatus: Optional[float]
    monthlyincome: Optional[float]
    monthlyrate: Optional[int]
    numcompaniesworked: Optional[int]
    over18:Optional[int]
    overtime:Optional[int]
    percentsalaryhike:Optional[int]
    performancerating:Optional[int]
    relationshipsatisfaction:Optional[int]
    standardhours:Optional[int]
    stockoptionlevel:Optional[int]
    totalworkingyears:Optional[int]
    trainingtimeslastyear:Optional[int]
    worklifebalance:Optional[int]
    yearsatcompany:Optional[int]
    yearsincurrentrole:Optional[int]
    yearssincelastpromotion:Optional[int]
    yearswithcurrmanager:Optional[int]
    
    # cnt: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]