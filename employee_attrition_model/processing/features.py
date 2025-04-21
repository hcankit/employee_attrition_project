from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# class WeekdayImputer(BaseEstimator, TransformerMixin):
#     """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

#     def __init__(self, variables: str):
#         if not isinstance(variables, str):
#             raise ValueError("variables should be a list")

#         self.variables = variables

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         self.fill_value=X[self.variables].mode()[0]
#         return self

#     def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:

#         df = dataframe.copy()
#         wkday_null_idx = df[df[self.variables].isnull() == True].index
#         # print(len(wkday_null_idx))
#         # df['dteday'] = pd.to_datetime(df['dteday'])
#         df.loc[wkday_null_idx, self.variables] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

#         return df

# class WeathersitImputer(BaseEstimator, TransformerMixin):
#     """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

#     def __init__(self, variables: str):
#         self.variables = variables

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         # X[self.variables].fillna('Clear', inplace=True)
#         # 'df.method({col: value}, inplace=True)'
#         X.fillna({self.variables: 'Clear'}, inplace=True)
#         return X    


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        #for feature in self.variables:
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: str):
        if isinstance(variables, str):
           self.variables = [variables] # if it's a single variable transform to list
        elif not isinstance(variables, list):
            raise ValueError("variables should be a list or string")
        else:
            self.variables=variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:  # loop over all numerical variables
            q1 = X.describe()[var].loc['25%']
            q3 = X.describe()[var].loc['75%']
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            for i in X.index:
                if X.loc[i,var] > upper_bound:
                    X.loc[i,var]= upper_bound
                if X.loc[i,var] < lower_bound:
                    X.loc[i,var]= lower_bound

        return X


# class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
#     """ One-hot encode weekday column """

#     def __init__(self, variables: str):
#         self.variables = variables
#         self.encoder = OneHotEncoder(sparse_output=False)

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         self.encoder.fit(X[[self.variables]])
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         encoded_weekday = self.encoder.transform(X[['weekday']])
#         enc_wkday_features = self.encoder.get_feature_names_out(['weekday'])
#         X[enc_wkday_features] = encoded_weekday
#         # Drop original weekday column after one-hot encoding
#         X.drop(columns=[self.variables], inplace=True)
#         return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """ Drop specified columns """
    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.drop(labels = self.variables, axis = 1, inplace = True)
        return X


# # Instantiate mapper for all ordinal categorical features

businesstravel_dict = {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2 }
businesstravel_mapping = Mapper('businesstravel', businesstravel_dict)

department_dict = {'Sales':0, 'Research & Development':1, 'Human Resources':2}
department_mapping = Mapper('department', department_dict)

educationfield_dict = {'Life Sciences':0, 'Other':1, 'Medical':2, 'Marketing':3,
        'Technical Degree':4, 'Human Resources':5}
educationfield_mapping = Mapper('educationfield', educationfield_dict )

gender_dict = {'Female':0, 'Male':1}
gender_mapping = Mapper('gender', gender_dict )

jobrole_dict = {'Sales Executive':0, 'Research Scientist':1, 'Laboratory Technician':2,
        'Manufacturing Director':3, 'Healthcare Representative':4, 'Manager':5,
        'Sales Representative':6, 'Research Director':7, 'Human Resources':8}
jobrole_mapping = Mapper('jobrole', jobrole_dict)

maritalstatus_dict = {'Single':0, 'Married':1, 'Divorced':2}
maritalstatus_mapping = Mapper('maritalstatus', maritalstatus_dict)

overtime_dict = {'Yes':1, 'No':0}
overtime_mapping = Mapper('overtime', overtime_dict)
