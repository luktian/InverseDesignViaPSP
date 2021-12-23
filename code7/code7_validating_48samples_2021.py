# fast-machine-learning==0.0.2.1
import joblib, pandas as pd
from fml.data import read_data
from fml.formulars import SplitFormular
from fml.data import DataObject
from fml.utils.base_func import metrics_
from fml.descriptors import HOIP

# Load voting model
voting_model = joblib.load("vr.joblib")
# Load 2021 samples for external validating
validate_samples_2021 = read_data("../code1/validating_42samples_year2021.xlsx")
# Split sample formulas
splitted_formular_2021, _ = SplitFormular().split_formulars(validate_samples_2021.PL.values, output=dict)

# Generate descriptors
hoip = HOIP()
descriptors = []
for formular in splitted_formular_2021:
    descriptors.append(hoip.describe_formular(formular, onehot=True))
descriptors = pd.concat(descriptors, axis=1).T
validating_obj = DataObject(X=descriptors.values, Y=validate_samples_2021.PLBandGap.values, Xnames=descriptors.columns, Yname="bg", indexes=descriptors.index.values.tolist())

# Predict bandgaps of validating samples
validating_preds = voting_model.predict(validating_obj)
# Evaluate results
result = metrics_(validating_obj.Y, validating_preds, task="reg")
print(f"External validating set: \n\
R2 score: {round(result['r2_score'], 2)}, MAE: {round(result['mae'], 3)}\n\
RMSE: {round(result['rmse'], 3)}, MSE: {round(result['mse'], 3)}")
# External validating set: 
# R2 score: 0.84, MAE: 0.04
# RMSE: 0.057, MSE: 0.003