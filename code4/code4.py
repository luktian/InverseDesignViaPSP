# fast-machine-learning==0.0.2.1
from fml.pipelines import RFA
import joblib

# Load the training and test sets from the step Code3
train = joblib.load("..//code3//train_1805_1959.joblib")
test = joblib.load("..//code3//test_1805_1959.joblib")

# Use our inhouse module RFA to perform feature selection
# Pass the training and test sets
# min_f and max_f refer to minimum and maximum feature numbers
rfa = RFA().fit_all(train, test, min_f=3, max_f=21)
# The results could be accessed in a dataframe "summary"
summary = rfa.summary_all
# Ouput the dataframe to an excel file if needed
summary.to_excel("RFA_result.xlsx", index=None)