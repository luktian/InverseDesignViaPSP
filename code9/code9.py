# fast-machine-learning==0.0.2.1
from fml.formulars import SplitFormular
from fml.descriptors import base_descriptor
from fml.searching import HOIPHyperSearchingWithVotingRegressor
import joblib, numpy as np, pandas as pd, os, time, random

# Load WVR model
voting_model = joblib.load("..//code7//vr.joblib")

# Load the feature names from WVR model
columns_set = []
for trainobj in voting_model.trainobjects:
    columns_set += trainobj.Xnames.tolist()
columns_set = list(set(columns_set))

# Set the organic fragments and elements for each site
formular_info = [
    base_descriptor.index[118:].tolist() + ["K", "Cs", "Rb"], # atoms in A site
    [
     # "Pb", 
     "Sn", "Ge", "Pd", "Bi", "Sr", "Ca", "Cr", "La"], # atoms in B site
    ["Cl", "Br", "I"] # atoms in C site
]

# formular_info = [
#     ["FA", "MA", "Cs"], # atoms in A site
#     ["Sn", "Ge", "Pd", "Sr", "Ca", "Cr"], # atoms in B site
#     # ["Sn"],
#     ["Br", "I"] # atoms in C site
# ]

# Set the criterion value
criterion = 0.05
# The object usd to split formuala
splitformular = SplitFormular()

start_time = time.time()

targeted_value = 1.75

site_counts = np.array([3,3,3])

# Set the output file
output_file = f"outputs{''.join(site_counts.astype(str))}_{str(targeted_value)}.csv"

# Write information into output file
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        f.writelines(f"formular, Prediction, {', '.join(columns_set)}\n")
    existed_formulars = []
else:
    existed_data = pd.read_csv(output_file, index_col=0)
    existed_formulars = existed_data.index.tolist()
    if len(existed_formulars) > 0:
        existed_formulars = splitformular.split_formulars(existed_formulars)[0]
    columns_set = existed_data.columns[1:]
    columns_set = [ i[1:] for i in columns_set ]

# Run the proactive searching progress (PSP) for multiple times
for i in range(100000):
    
    # Set multiple expectation values to exert different searching
    # targeted_value = random.choice([1.35, 1.40, 1.45])
    # Or set a single value
    # targeted_value = 1.75
    
    # Choose one doping form
    # site_counts = np.array(random.choice([
    #     [1,1,1], [2,1,1], [1,2,1], [1,1,2], 
    #     [1,2,2], [2,1,2], [2,2,1], [2,2,2], 
    #     [3,3,3],
    #     ]))
    # Or set a single form
    
    
    # Choose an iteration step in PSP
    iteration = random.choice(np.linspace(100, 2000, 11).astype(int))
    
    # Perform the PSP
    hhwvr = HOIPHyperSearchingWithVotingRegressor(iteration, True, True).\
        fit_with_full_range(voting_model, formular_info, targeted_value, 
                            criterion, site_counts=site_counts)
    # Write the PSP results into output file
    if len(hhwvr.trials) > 0:
        trials = np.array(hhwvr.trials)
        trials_descriptors = pd.DataFrame(trials[:, 3:], 
                                          index=trials[:, 0], 
                                          columns=hhwvr.descriptor_names.tolist()
                                          )[columns_set].astype(float)
        predictions = trials[:, 1].astype(float)
        
        with open(output_file, "a") as f:
            for prediction, (formular, descripters) in \
                zip(predictions, trials_descriptors.iterrows()):
                try:
                    if splitformular.split_formular(formular) not in existed_formulars:
                        f.writelines(f"{formular}, {prediction}, {', '.join(descripters.astype(str).tolist())}\n")
                        existed_formulars.append(splitformular.split_formular(formular))
                except:
                    pass
end_time = time.time()
print(f"{end_time - start_time} s")