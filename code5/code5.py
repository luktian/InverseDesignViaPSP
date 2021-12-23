# fast-machine-learning==0.0.2.1
from fml.pipelines import SHAPModelling
from fml.feature_selection import Shap
import joblib
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 600

from xgboost import XGBRegressor as XGB
from catboost import CatBoostRegressor as CAT
from lightgbm import LGBMRegressor as LGB
from sklearn.ensemble import GradientBoostingRegressor as GBT

# Load the training and test sets from the step Code3
train = joblib.load("..//code3//train_1805_1959.joblib")
test = joblib.load("..//code3//test_1805_1959.joblib")

# Define the feature numbers resulted from Code4
feature_numbers = {
    CAT: 13, 
    XGB: 13,
    LGB: 12,
    GBT: 10,
    }
# Define the model abbreviations
abbrs = {
    CAT: "CAT", 
    XGB: "XGB",
    LGB: "LGB",
    GBT: "GBT",
    }

# Iterate 4 models
for algo in [CAT, XGB, LGB, GBT]:
    # Transform the original training and test sets with the pre-defined optimal feature numbers
    train_, test_ = SHAPModelling().fit(algo, train.copy(), test.copy()).transform(feature_numbers[algo])
    # Use fml.Shap module to perform SHAP calculations automatically
    explainer = Shap().fit(algo, train_)
    # Plot the SHAP importance
    explainer.bar(max_display=20, xlabel=f"{abbrs[algo]} SHAP Values")