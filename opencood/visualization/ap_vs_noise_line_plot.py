# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import matplotlib.pyplot as plt
import numpy as np

# 移除 ggplot 风格以获得白色背景
# plt.style.use('ggplot')

methods = ['MPDA', 'BackAlign', 'CodeFilling','STAMP', 'GenComm']

xaxis_names = ['Pose Noise Std (m & deg)', 'Pose Noise Std (m & deg)', "Time Delay (ms)", "Time Delay (ms)"]
yaxis_names = ['Performance AP50', 'Performance AP70','Performance AP50','Performance AP70']

# noise_level = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# pose_error_performance_50 = {
#     "MPDA":  [0.687, 0.683, 0.67, 0.648, 0.631, 0.616],
#     "BackAlign": [0.715, 0.708, 0.692, 0.663, 0.64, 0.617],
#     "CodeFilling": [0.56, 0.556, 0.538, 0.517, 0.495, 0.482],
#     "GenComm": [0.761, 0.753, 0.728, 0.69, 0.66, 0.626],
# }
# pose_error_performance_70 = {
#     "MPDA":  [0.502, 0.498, 0.489, 0.474, 0.462, 0.453],
#     "BackAlign": [0.533, 0.524, 0.506, 0.481, 0.468, 0.451],
#     "CodeFilling": [0.416, 0.411, 0.391, 0.375, 0.361, 0.354],
#     "GenComm":   [0.575, 0.558, 0.524, 0.492, 0.467, 0.449],
# }

# time_delay = [0, 100, 200, 300, 400, 500,]
# time_delay_performance_50 = {
#     "MPDA":  [0.687, 0.673, 0.636, 0.588, 0.555, 0.536],
#     "BackAlign": [0.715, 0.694, 0.624, 0.576, 0.547, 0.532],
#     "CodeFilling": [0.56, 0.542, 0.487, 0.445, 0.42, 0.408],
#     "GenComm":   [0.761, 0.73, 0.643, 0.578, 0.544, 0.53],
# }
# time_delay_performance_70 = {
#     "MPDA":  [0.502, 0.485, 0.46, 0.433, 0.413, 0.401],
#     "BackAlign": [0.533, 0.486, 0.442, 0.421, 0.411, 0.404],
#     "CodeFilling": [0.416, 0.381, 0.34, 0.322, 0.315, 0.31],
#     "GenComm":   [0.575, 0.512, 0.447, 0.422, 0.405, 0.397],
# }

noise_level = [0, 0.2, 0.4, 0.6, 0.8,]
pose_error_performance_50 = {
    "MPDA":  [0.687, 0.683, 0.67, 0.648, 0.631, ],
    "BackAlign": [0.715, 0.708, 0.692, 0.663, 0.64, ],
    "CodeFilling": [0.56, 0.556, 0.538, 0.517, 0.495, ],
    'STAMP': [0.751, 0.736, 0.707, 0.666, 0.626], 
    "GenComm": [0.761, 0.753, 0.728, 0.69, 0.66, ],
}
pose_error_performance_70 = {
    "MPDA":  [0.502, 0.498, 0.489, 0.474, 0.462, ],
    "BackAlign": [0.533, 0.524, 0.506, 0.481, 0.468, ],
    "CodeFilling": [0.416, 0.411, 0.391, 0.375, 0.361, ],
    'STAMP': [0.544, 0.529, 0.5, 0.472, 0.444], 
    "GenComm":   [0.575, 0.558, 0.524, 0.492, 0.467, ],
}

time_delay = [0, 100, 200, 300, 400, 500]
time_delay_performance_50 = {
    "MPDA":  [0.687, 0.673, 0.654, 0.632, 0.612, 0.592],
    "BackAlign": [0.715, 0.694, 0.659, 0.63, 0.61,0.59],
    "CodeFilling": [0.56, 0.543, 0.514, 0.493, 0.475,0.46],
    'STAMP': [0.751, 0.710, 0.659, 0.620, 0.592, 0.574], 
    "GenComm":   [0.761, 0.73, 0.684, 0.648, 0.622, 0.601],
}
time_delay_performance_70 = {
    "MPDA":  [0.502, 0.485, 0.473, 0.458, 0.446, 0.435],
    "BackAlign": [0.533, 0.486, 0.463, 0.45, 0.443, 0.433],
    "CodeFilling": [0.416, 0.38, 0.36, 0.351, 0.342, 0.322],
    'STAMP': [0.544, 0.464, 0.44, 0.422, 0.416, 0.411], 
    "GenComm":   [0.575, 0.512, 0.479, 0.458, 0.446, 0.435],
}



draw_list = [
    {'xaxis': noise_level, 'yaxis': pose_error_performance_50, 'ylim': [0.2, 0.8]},
    {'xaxis': noise_level, 'yaxis': pose_error_performance_70, 'ylim': [0.2, 0.6]},
    {'xaxis': time_delay, 'yaxis': time_delay_performance_50, 'ylim': [0.2, 0.8]},
    {'xaxis': time_delay, 'yaxis': time_delay_performance_70, 'ylim': [0.2, 0.6]},
]

color = {
    'GenComm': 'r',
    'MPDA': 'skyblue',
    'BackAlign': 'slategrey',
    'CodeFilling': 'mediumpurple',
}

color = {
    'GenComm': '#E54C5E',                  # 保留红色
    'MPDA': '#4874CB',              # 蓝色 (标准 Matplotlib 蓝)
    'BackAlign': '#75BD42',         # 绿色 (标准 Matplotlib 绿)
    'STAMP': '#30C0B4',          # 紫色 (标准 Matplotlib 紫)
    'CodeFilling': '#EE822F',       # 橙色 (标准 Matplotlib 橙)
}

markers = {
    'GenComm': 'o',
    'MPDA': 's',
    'BackAlign': '^',
    'STAMP': 'D',
    'CodeFilling': 'D',
}

fig, axs = plt.subplots(1, 4, figsize=(12, 2.85))

for idx, ax in enumerate(axs):
    data = draw_list[idx]
    for method in methods:
        ax.plot(data['xaxis'], data['yaxis'][method],'-s', label=method, markersize=2, color=color[method])
    ax.set_xticks(data['xaxis'])
    if 'Pose' in xaxis_names[idx]:
        ax.set_xticks(data['xaxis'][::2])
    ax.set_xlabel(xaxis_names[idx])
    ax.set_ylabel(yaxis_names[idx])
    ax.grid(False)
    ax.set_ylim(data['ylim'])
    ax.tick_params(axis='both', colors='black')
    if 'xlabel' in data:
        ax.set_xticklabels(data['xlabel'])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower left', prop={'size': 6.8})

# plt.tight_layout()
# file_path = "vis_result/AP_PoseError/line_plot_of_AP_vs_PoseError.png"
# plt.savefig(file_path, dpi=500)
# plt.close()
# print(f"Saving to {file_path}")

# # 提取图例元素，仅一次
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03), frameon=False)

# 给图例预留空间
plt.tight_layout(rect=[0, 0.05, 1, 1])

file_path = "vis_result/AP_PoseError/line_plot_of_AP_vs_PoseError.pdf"
plt.savefig(file_path, dpi=800)
plt.close()
print(f"Saving to {file_path}")