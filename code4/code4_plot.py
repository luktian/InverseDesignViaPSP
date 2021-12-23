# fast-machine-learning==0.0.2.1
from fml.data import read_data
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, MaxNLocator
from matplotlib import cm
plt.rcParams["figure.dpi"] = 600
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

rfa_results = read_data("RFA_result.xlsx")

abbrs = {
    "CatBoostRegressor": "CAT",
    "XGBRegressor": "XGB",
    "LGBMRegressor": "LGB",
    "GradientBoostingRegressor": "GBT",
    "SVR": "SVM",
    "DecisionTreeRegressor": "DTR",
    "LinearRegression": "MLR",
    }

colors = {
    "CatBoostRegressor": "tab:blue",
    "XGBRegressor": "tab:orange",
    "LGBMRegressor": "tab:green",
    "GradientBoostingRegressor": "tab:red",
    "SVR": "tab:purple",
    "DecisionTreeRegressor": "tab:brown",
    "LinearRegression": "tab:pink",
    }

plot_sequences = [
    ["CatBoostRegressor", "XGBRegressor", "LGBMRegressor", "GradientBoostingRegressor"],
    ["SVR", "DecisionTreeRegressor", "LinearRegression"]
    ]

def format_axis(tick_val, tick_pos):
    if int(tick_val) in range(21) and tick_val % 2 == 0:
        return int(tick_val)
    else:
        return ''

metrics = ["LOO RMSE", "LOO ${R^2}$", "test RMSE", "test ${R^2}$"]
metrics_titles = ["loo_rmse", "loo_r2_score", "test_rmse", "test_r2_score"]
# for metric, metric_t in zip(metrics, metrics_titles):
#     fig, axes = plt.subplots(1, 2, figsize=(23, 8))
#     for plot_sequence, ax, a_b in zip(plot_sequences, axes, ["a)", "b)"]):
#         for plot_algo in plot_sequence:
#             plot_xy = rfa_results[rfa_results.index==plot_algo]
#             plot_x, plot_y = plot_xy.feature_num, plot_xy.loc[:, metric_t]
#             ax.plot(plot_x, plot_y, label=abbrs[plot_algo], marker='o', 
#                     markersize=12, color=colors[plot_algo])
#         ax.legend()
#         ax.set_ylabel(f"{metric}", fontsize=22)
#         ax.set_xlabel("Feature Numbers", fontsize=22)
#         ax.xaxis.set_major_formatter(format_axis)
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         ax.set_title(f"RFA {metric}", fontsize=24)
#         ax.text(-0.13, 1, a_b, fontsize=24, transform=ax.transAxes)

for metric, metric_t in zip(metrics[:1], metrics_titles[:1]):
    fig, ax = plt.subplots(1, figsize=(11.5, 8))
    for plot_algo in plot_sequences[0]:
        plot_xy = rfa_results[rfa_results.index==plot_algo]
        plot_x, plot_y = plot_xy.feature_num, plot_xy.loc[:, metric_t]
        ax.plot(plot_x, plot_y, label=abbrs[plot_algo], marker='o', 
                markersize=12, color=colors[plot_algo])
    ax.legend()
    ax.set_ylabel(f"{metric}", fontsize=22)
    ax.set_xlabel("Feature Numbers", fontsize=22)
    ax.xaxis.set_major_formatter(format_axis)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"RFA {metric}", fontsize=24)
    ax.scatter(
        [13, 13, 12, 10], 
        [0.0852249424633167, 0.0967982375740553, 0.10789942645925, 0.0923759208813403], 
        marker="o", s=500, c="",
        edgecolor=["#000000"], linewidths=3,
        # edgecolor=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        )
