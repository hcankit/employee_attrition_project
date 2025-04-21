from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]

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

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

