# fast-machine-learning==0.0.2.1
from fml.pipelines import SHAPModelling
from fml.ensemble import VotingRegressor
import joblib, glob

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

# Obtain the best parameters from code 6
best_parameters = { i.split("_")[1].split(".")[0]:joblib.load(i).best_p \
                   for i in glob.glob("..//code6//*.joblib") }
best_parameters = { i:best_parameters[j] for i, j in abbrs.items() }

# Organize the algorithms, trains, tests, model parameters
trains, tests, model_ps, algos = [], [], [], []
for algorithm in [CAT, XGB, LGB, GBT]:
    train_, test_ = SHAPModelling().fit(
        algorithm, train.copy(), test.copy()
        ).transform(feature_numbers[algorithm])
    trains.append(train_)
    tests.append(test_)
    model_ps.append(best_parameters[algorithm])
    algos.append(algorithm)

# Fit the VotingRegressor
vr = VotingRegressor(verbose=True, rounds=100)
vr.fit(algos, trains, tests, model_ps)
# Get the results from vr object
results = vr.results
# Best weights
best_w = vr.best_weights
# Save the vr object
# joblib.dump(vr, "vr.joblib")


# algos = [CAT, XGB, LGB, GBT]
# trains = [train_cat, train_xgb, train_lgb, train_gbt]
# tests = [test_cat, test_xgb, test_lgb, test_gbt]
# model_ps = [
#     {'learning_rate': 0.03,
#      'n_estimators': 800,
#      'max_depth': 8,
#      'verbose': False,
#      'thread_count': 1},
#     {'n_estimators': 50, 'learning_rate': 0.3, 'max_depth': 5, 'n_jobs': 1},
#     {'learning_rate': 0.15000000000000002, 'n_estimators': 350, 'max_depth': 5},
#     {'learning_rate': 0.25, 'n_estimators': 80, 'max_depth': 4}])