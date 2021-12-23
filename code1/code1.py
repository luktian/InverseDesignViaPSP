# fast-machine-learning==0.0.2.1
from fml.data import read_data
from fml.formulars import SplitFormular
from fml.targets import DealMultipleTarget
import numpy as np

# Load the 1159 raw samples pre-extracted from 9509 publications
data = read_data("extracted_samples_1159.xlsx") # data is a dataframe
raw_formulars = data.PL
data.index = raw_formulars
raw_bg = data.PLBandGap

# Some unregular formular raios are also considered
unregular_formular_ratios = [
    [1.05, 1, 3],
    [1, 1.1, 3.3],
    [0.97, 1, 2.77],
    [0.966, 1, 2.90],
    [0.9575, 1, 2.9575],
    [1, 1.05, 3],
    [1.02, 1, 3],
    [0.95, 1, 3],
    [1.05, 1, 3.05],
    [0.99, 1, 3],
    ]

# Use the module fml.formulars.SplitFormular to split the formualrs into string format
splitted_formular, unsplitted_formular = SplitFormular(unregular_formular_ratios).\
    split_formulars(raw_formulars.values, output=str)

# Use the module fml.targets.DealMultipleTarget to deal with 
# the multiple target values of one singular sample
# 
# The deal method is selected as "prone_mean" that considers 
# the standard distribution of the targets. The deal alternatives 
# could also be "mean", "max", "min"
# 
# the first column of dealed_data is the formulars
# the second column of dealed_data is the processed band gaps
# the counters contain the statistical information of the multiple target values
dealed_data, counters = DealMultipleTarget(deal="prone_mean").deal_with_multiple_data(
    np.concatenate([np.array(splitted_formular).reshape(-1, 1), raw_bg.values.reshape(-1, 1)], axis=1)
    )

# The dealed_data could be outputed in an excel by using dataframe
import pandas as pd
output_data = pd.DataFrame(dealed_data, columns=["PL", "bg"])
output_data.iloc[:, 1] = output_data.iloc[:, 1].astype(float)
output_data.to_excel("pruned_samples_437.xlsx", index=None)
