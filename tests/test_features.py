
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from employee_attrition_model.config.core import config
from employee_attrition_model.processing.features import Mapper


def test_businesstravel_mapper(sample_input_data):
    print(sample_input_data[0].head())
    # Given
    transformer = Mapper(
        variables='businesstravel',
        mappings ={'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}
    )
    # assert np.isnan(sample_input_data[0].loc[12830,'season']) 

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1041,'businesstravel'] == 0