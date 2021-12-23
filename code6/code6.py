# fast-machine-learning==0.0.2.1
from fml.pipelines import SHAPModelling
from fml.parameter_opt import GridSearch
from fml.configs.auto_config import gridsearch_parameters_reg
import joblib

from xgboost import XGBRegressor as XGB
from catboost import CatBoostRegressor as CAT
from lightgbm import LGBMRegressor as LGB
from sklearn.ensemble import GradientBoostingRegressor as GBT

algorithm = XGB

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

# Transform the original training and test sets 
# with the pre-defined optimal feature numbers
train_, test_ = SHAPModelling().fit(
    algorithm, train, test
    ).transform(feature_numbers[algorithm])

# Perform grid search (GS) approach
grid_p = gridsearch_parameters_reg[algorithm]
gs = GridSearch(n_jobs=10).fit(
    algorithm, train_, test_, cv=True, **grid_p
    )

# GS results are exhibited as a table format
results = gs.results
# Obtain the best parameters and the best result
best_p = gs.best_p
best_result = gs.best_result
# Output the GS results into a table
results.to_excel(f"{algorithm}_gs_results.xlsx")

import joblib
joblib.dump(gs, f"gs_{algorithm}.joblib")

