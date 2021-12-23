# fast-machine-learning==0.0.2.1
from fml.data import read_data
from fml.formulars import SplitFormular
from fml.descriptors import HOIP
import pandas as pd

# Load the 437 pruned samples in dataframe format
# Index refers to the formular strings, and the first column is the pruned band gaps
bandgap_data = read_data("..//code1//pruned_samples_437.xlsx")
formulars = bandgap_data.index.values
bandgaps = bandgap_data.iloc[:, 0].values

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
# Split the pruned formular strings into dict format that will be passed into 
# the class module HOIP to generate descriptors
splitted_formulars, _ = SplitFormular(unregular_formular_ratios).split_formulars(formulars)

# The class HOIP is used to generate descriptors
# Firstly instantiate an HOIP object with the bool options as followed
# The base refers to use the base descriptors
# The str_bool refers to use the string descriptors
# The other referes to 66 other descriptors
# Ionization and ionic radii are calcualted separately
hoip = HOIP(base=True, str_bool=True, other=False, ionization=True, ionic_radii=True)

# Use the class function "describe_formular" of hoip object to calculate descriptors
# Pass a singular formular, which is formatted as:
# formular = [{atom:ratio, atom:ratio, ...},
#             {atom:ratio, atom:ratio, ...},
#             {atom:ratio, atom:ratio, ...}]
# The first dict refers to A site, the second and third refer to B and C sites
# Switch the onehot to True. The string descriptors will be generated in numeric data
descriptors = []
for formular, bg in zip(splitted_formulars, bandgaps):
    descriptors.append(hoip.describe_formular(formular, onehot=True))
# Collecting descriptors into variable "descriptors" and concatenate into a dataframe
descriptors = pd.concat(descriptors, axis=1).T

# Concatenate the band gap and descriptors into one sheet
dataset = pd.concat([bandgap_data, descriptors], axis=1)
dataset.to_excel("HOIP_dataset_437.xlsx")
