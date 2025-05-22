# -*- coding: utf-8 -*-
# Author: Junfei Zhou <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np

# methods = ['MPDA', 'BackAlign', 'CodeFilling', 'STAMP', 'GenComm']
# methods = ['MPDA', 'CodeFilling', 'STAMP', 'GenComm']
methods = [ 'STAMP','CodeFilling',  'GenComm']
agent_num = [1, 2, 3, 4, 5, 6, 7, 8]

# pose_error_performance_50 = {
#     "BackAlign(log2)": [6.57, 27.82, 28.71, 36.85, 43.42, 64.67, 65.56, 73.7],
#     "CodeFilling": [0.27, 0.54, 0.81, 1.08, 1.35, 1.62, 1.89, 2.16],
#     "STAMP": [0.547, 1.094, 1.641, 2.188, 2.735, 3.282, 3.829, 4.376],
#     "GenComm": [0.103, 0.206, 0.31, 0.403, 0.506, 0.61, 0.713, 0.816],
# }

# color = {
#     'GenComm': '#C05046',
#     'BackAlign(log2)': '#BFBFBF',
#     'CodeFilling': '#EFC085',
#     'STAMP': '#4BACC6',  
# }


pose_error_performance_50 = {
    "STAMP": [0.547, 1.094, 1.641, 2.188, 2.735, 3.282, 3.829, 4.376],
    "CodeFilling": [0.27, 0.54, 0.81, 1.08, 1.35, 1.62, 1.89, 2.16],
    "GenComm": [0.103, 0.206, 0.31, 0.403, 0.506, 0.61, 0.713, 0.816],
}

color = {
    'GenComm': '#C05046',
    'CodeFilling': '#EFC085',
    'STAMP': '#4BACC6',  
}

fig, ax = plt.subplots(figsize=(3.7, 4))

for method in methods:
    log_y = [np.log2(y) if method == 'BackAlign(log2)' or method == 'MPDA' else y for y in pose_error_performance_50[method]]
    if method == "BackAlign(log2)":
        log_y = [y-1 for y in log_y ]
    ax.plot(agent_num, log_y, '-s', label=method, markersize=4, color=color[method])

ax.set_xticks(agent_num)
ax.set_xlabel('Agent number', size=18)
ax.set_ylabel('#Params (M)', size=18)
ax.grid(False)
ax.tick_params(axis='both', colors='black')
ax.legend(loc='upper left', prop={'size': 10}, frameon=False)

plt.tight_layout()
file_path = "vis_result/scalability_log_transformed_with_stamp.png"
plt.savefig(file_path, dpi=500)
plt.close()
print(f"Saving to {file_path}")
