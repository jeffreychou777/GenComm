# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据
# channels = [1, 2, 3, 4, 5, 6]
# performance = [0.8001, 0.8354, 0.8367, 0.8272, 0.8376, 0.8343]
# inference_time = [20.7, 20.98, 24.89, 24.9, 28.27, 31.45]  # ms
# comm_cost = [0.5, 1, 2, 6, 14, 28]  # MB

# # 气泡大小表示通信成本
# bubble_size = [s * 20 for s in comm_cost]

# # 创建图形和主轴
# fig, ax1 = plt.subplots(figsize=(4.9, 3))

# # 左轴：性能
# color_perf = 'r'
# line1, = ax1.plot(channels, performance, '-', color=color_perf, label='Performance (AP50)', zorder=1)

# # 气泡图：统一颜色，仅大小代表通信成本
# sc1 = ax1.scatter(
#     channels, performance,
#     s=bubble_size, color=color_perf,
#     alpha=0.6, edgecolors='k', linewidth=0.5, zorder=2
# )

# ax1.set_xlabel('Channel Size')
# ax1.set_ylabel('Performance (AP)', color=color_perf)
# ax1.tick_params(axis='y', labelcolor=color_perf)
# ax1.set_xticks(channels)
# ax1.grid(False)
# ax1.set_ylim(min(performance) - 0.02, max(performance) + 0.05)

# # 右轴：推理时间
# ax2 = ax1.twinx()
# color_time = '#1f77b4'
# line2, = ax2.plot(channels, inference_time, '-', color=color_time, label='Inference Time (ms)', zorder=0)
# ax2.set_ylabel('Inference Time (ms)', color=color_time)
# ax2.tick_params(axis='y', labelcolor=color_time)

# # 图例
# lines = [line1, line2]
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels, loc='upper left', prop={'size': 8})

# # 保存
# plt.tight_layout()
# plt.savefig('vis_result/channel_vs_perf_time_comm_bubble_only.png', dpi=500)
# plt.close()
# print("Saved to vis_result/channel_vs_perf_time_comm_bubble_only.png")


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# 示例数据
index = [1, 2, 3, 4, 5, 6]
channel_labels = [1, 2, 4, 16, 64, 128]
performance = [0.8001, 0.8354, 0.8367, 0.8272, 0.8376, 0.8343]
inference_time = [20.7, 20.98, 24.89, 24.9, 28.27, 31.45]  # ms
comm_cost = [0.5, 1, 2, 6, 14, 28]  # MB

# 气泡大小
bubble_size = [s * 20 for s in comm_cost]

# 创建图形和主轴
fig, ax1 = plt.subplots(figsize=(4.5, 2.8))

# 左轴：性能
color_perf = 'r'
line1, = ax1.plot(index, performance, '-', color=color_perf, label='Performance (AP50)', zorder=1)

# 气泡图：通信量
sc1 = ax1.scatter(
    index, performance,
    s=bubble_size, color=color_perf,
    alpha=0.6, edgecolors='k', linewidth=0.5, zorder=2
)

ax1.set_xlabel('Channel Size')
ax1.set_ylabel('Performance (AP)', color=color_perf)
ax1.tick_params(axis='y', labelcolor=color_perf)
ax1.set_xticks(index)
ax1.set_xticklabels(channel_labels)
ax1.grid(False)
ax1.set_ylim(min(performance) - 0.02, max(performance) + 0.05)

# 右轴：推理时间
ax2 = ax1.twinx()
color_time = '#1f77b4'
line2, = ax2.plot(index, inference_time, '-', color=color_time, label='Inference Time (ms)', zorder=0)
ax2.set_ylabel('Inference Time (ms)', color=color_time)
ax2.tick_params(axis='y', labelcolor=color_time)

# 自定义图例项：通信量气泡
bubble_legend = mlines.Line2D([], [], color='r', marker='o', linestyle='None',
                               markersize=10, markerfacecolor='r', alpha=0.5,
                               markeredgecolor='k', label='Communication Volume')


# 图例合并
lines = [line1, line2, bubble_legend]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', prop={'size': 9})

# 保存
plt.tight_layout()
plt.savefig('vis_result/vis_ablation_ch.png', dpi=500)
plt.close()
print("Saved to vis_result/vis_ablation_ch.png")

