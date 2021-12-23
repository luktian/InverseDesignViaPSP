# fast-machine-learning==0.0.2.1
import joblib, numpy as np
from fml.ensemble import VotingShap
from shap import plots
from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300

# read voting model
voting_model = joblib.load("..//code7//vr.joblib")
# read virtual data set
dataset = joblib.load("dataset_virtual.joblib")
dataset = dataset.astype(float)
# fit voting shap object or the user could choose the supplied fitted object "votingshap_virtual.joblib""
vs = VotingShap(voting_model, dataset).fit() # vs = joblib.load("votingshap_virtual.joblib")
# read the feature names
columns = vs.columns_set[1:]
# read X and Y
X = vs.dataobject.to_df().iloc[:, 1:]
predicted_Y = voting_model.predict(dataset)
# read shap values
shap_values = vs.shap_values

# feature importance bar plot
plots.bar(deepcopy(shap_values), max_display=11)

# instance distribution plot
instance_order = np.array(sorted(zip(np.arange(len(X)), predicted_Y), key=lambda x: x[1], reverse=True))[:, 0].astype(int)
plots.heatmap(deepcopy(shap_values), max_display=11, instance_order=instance_order)

# decision plot
plots.decision(shap_values.base_values[0], deepcopy(shap_values.values), feature_names=columns)

# beeswarm plot
plots.beeswarm(deepcopy(shap_values), order=vs.feature_order, max_display=11)

# violin plot
plots.violin(deepcopy(shap_values.values), X, feature_names=columns, max_display=11)

# scatter plots for each feature
for findex in vs.feature_order[:10]:
    fname = X.columns[findex]
    plots.scatter(deepcopy(shap_values[:, findex]), color=predicted_Y)

# ICE plots for each feature
for findex in vs.feature_order[:10]:
    fname = X.columns[findex]
    plots.partial_dependence(findex, vs.predict, X, feature_names=columns)

# PD plots for each feature
for findex in vs.feature_order[:10]:
    fname = X.columns[findex]
    plots.partial_dependence(findex, vs.predict, X, feature_names=columns, ice=False)

