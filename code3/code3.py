# fast-machine-learning==0.0.2.1
from fml.data import read_data
from fml.preprocessing import Preprocessing
from fml.sampling import HpSplit, random_split

# Load the generated data set composed of 437 samples × 129 features
dataset = read_data("..//code2//HOIP_dataset_437.xlsx", df=False)
print(f"Feature numbers: {dataset.X.shape[1]}")
# Prune the variables that have missing values or low standard deviations 
# nearly to zero, bringing about 102 remnant features left
preprocessed_dataset = Preprocessing(sd=True, corr=False).fit_transform(dataset)
print(f"Feature numbers: {preprocessed_dataset.X.shape[1]}")

# Use module HpSplit to optimize random state and test size in pursuit a suitable 
# sample distribution of training and test sets
# "rounds" is set as 200 to perform 200 iterations
# verbose is set as True to print the log information
# Pass dataset varibale as the first argument of "fit" function
# cv is set as True which means leaving one out cross validation (LOOCV)
# cv could also be set as an int number which refers to number-fold CV
# The root mean squared error (RMSE) of LOO is used as the optimizing fitness
hs = HpSplit(rounds=200, verbose=True).fit(dataset, cv=True)
# the parameters are saved in "best_params", and summary stores the log information
best_params, summary = hs.best_params, hs.summary
print(f"Random state: {best_params['random_state']}")
print(f"test size: {best_params['test_size']}")

# Use the best parameters from the HpSplit result to split the data
train, test = random_split(dataset, best_params['test_size'],
                           best_params['random_state'])

# Our optimized result:
# train, test = random_split(dataset, 0.18052525045302406, 1959)

# Ouput the train and test
import joblib
joblib.dump(train, "train.joblib")
joblib.dump(test, "test.joblib")

# If you want see the data, use "to_df()" to transform to dataframe
train = train.to_df()
test = test.to_df()
