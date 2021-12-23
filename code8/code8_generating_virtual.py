# fast-machine-learning==0.0.2.1
import joblib, pandas as pd
from fml.data import DataObject
from fml.searching import HOIPHTSWithVotingRegressor

# load voting regressor model
voting_model = joblib.load("..//code7//vr.joblib")
# define site elements
site_elements = {'A': ['MA', 'FA', 'Cs', 'GA', 'EA', 'ED'],
                 'B': ['Pb', 'Sn', 'Ge', 'Cd', 'Pd'],
                 'C': ['I', 'Cl', 'Br']}
# high-throughput calculating
hoip_hts = HOIPHTSWithVotingRegressor(True).\
    fit_with_full_range(voting_model, 
                        list(site_elements.values()), [2, 2, 2], 
                        steps=[.25, .25, 1.0], starts=[0, 0, 0], 
                        ends=[1, 1, 3])
# generating the virtual data set
descriptors = pd.concat(hoip_hts.descriptors, axis=1).T
HTS_shap_data = DataObject(X=descriptors.values, Y=hoip_hts.predictions.values, 
                            Xnames=descriptors.columns, indexes=hoip_hts.predictions.index,
                            Yname=hoip_hts.predictions.name)
# save to local
joblib.dump(HTS_shap_data.to_df(), "dataset_virtual.joblib")