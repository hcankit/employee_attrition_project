import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from employee_attrition_model.config.core import config
from employee_attrition_model.processing.features import OutlierHandler
from employee_attrition_model.processing.features import ColumnDropper
from employee_attrition_model.processing.features import businesstravel_mapping
from employee_attrition_model.processing.features import department_mapping
from employee_attrition_model.processing.features import educationfield_mapping
from employee_attrition_model.processing.features import gender_mapping
from employee_attrition_model.processing.features import jobrole_mapping
from employee_attrition_model.processing.features import maritalstatus_mapping
from employee_attrition_model.processing.features import overtime_mapping
from employee_attrition_model.processing.data_manager import outlier_features
from employee_attrition_model.processing.data_manager import unused_colms


attrition_pipe=Pipeline([
    ('yr_mapper', businesstravel_mapping),
    ('mnth_mapper', department_mapping),
    ('season_mapper', educationfield_mapping),
    ('weather_mapper', gender_mapping),
    ('holiday_mapper', jobrole_mapping),
    ('workingday_mapper', maritalstatus_mapping),
    ('hour_mapper', overtime_mapping),
    ('outlier_handler', OutlierHandler(outlier_features)),
    ('column_dropper', ColumnDropper(unused_colms)),
    ('regressor', RandomForestClassifier())
])


