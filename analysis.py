import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# # 原始模型的指标值
# original_model = [[0.165880, 0.854315], [95.17241, 11.29032]]
#
# # 优化后模型的指标值
# optimized_model = [[0.045992, 0.623829],[ 97.24137, 8.60215]]
#
# # 指标名称
# indicators = [['Jitter variance', 'Vehicle body shake variance'], ['Warning success rate', 'False alarm rate']]
#
# # 指标单位
# units = [['m²', 'm²'], ['%', '%']]
#
# # 创建子图
# fig, axs = plt.subplots(2, 2, figsize=(5, 5))
#
# # 设置柱状图的宽度
# width = 0.4
# x = np.arange(len(indicators))
#
# for i in range(2):
#     for j in range(2):
#         ax = axs[i][j]
#
#         if i == 1 and j == 0:
#             ax.set_ylim(80, 100)
#
#         ax.bar([0, 1], [original_model[i][j], optimized_model[i][j]], width,
#                color=['#00008B', 'g'], tick_label=['Raw data', 'Calibrated data'])
#         ax.set_title(f'{indicators[i][j]}')
#         ax.set_ylabel(units[i][j])
#
#         # # 绘制原始模型的柱状图
#         # ax.bar(x - width/2, original_model[i], width, label='原始模型')
#         # # 绘制优化后模型的柱状图
#         # ax.bar(x + width/2, optimized_model[i], width, label='优化后模型')
#         # # 添加标签、标题和图例
#         # ax.set_xlabel('指标')
#         # ax.set_ylabel(units[i])
#         # ax.legend()
#
# # 调整子图布局
# plt.tight_layout()
#
# # 显示子图
# plt.show()

'''
{'address 3': '34:8a:12:7d:7a:31', 'address 2': '34:8a:12:7d:7a:31', 'address 1': 'ff:ff:ff:ff:ff:ff', 'seq': 83, 
'ssid': 'TJ-DORM-WIFI', 'subtype': 'Beacon', 'type': 'management', 'duration': 0, 'encoding': 0, 
'snr': 8.76877248665261, 'frame_bytes': 299, 'd_frame_start': 162, 'frame_start_time': 34.26997305, 
'device': 'litux-Yoga-Pro-14s-IAH7'}
'''
#
# data = b'\t\x07\x02\x00\taddress 3\x02\x00\x1134:8a:12:7d:7a:31\t\x07\x02\x00\taddress 2\x02\x00\x1134:8a:12:7d:7a:31\t\x07\x02\x00\taddress 1\x02\x00\x11ff:ff:ff:ff:ff:ff\t\x07\x02\x00\x03seq\x0b\x00\x00\x00\x00\x00\x00\x00S\t\x07\x02\x00\x04ssid\x02\x00\x0cTJ-DORM-WIFI\t\x07\x02\x00\x07subtype\x02\x00\x06Beacon\t\x07\x02\x00\x04type\x02\x00\nmanagement\t\x07\x02\x00\x08duration\x03\x00\x00\x00\x00\t\x07\x02\x00\x08encoding\x0b\x00\x00\x00\x00\x00\x00\x00\x00\t\x07\x02\x00\x03snr\x04@!\x89\x9c\x8c y\xa1\t\x07\x02\x00\x0bframe_bytes\x0b\x00\x00\x00\x00\x00\x00\x01+\t\x07\x02\x00\rd_frame_start\x0b\x00\x00\x00\x00\x00\x00\x00\xa2\t\x07\x02\x00\x10frame_start_time\x04@A"\x8ez\x16F\x93\t\x07\x02\x00\x06device\x02\x00\x17litux-Yoga-Pro-14s-IAH7\x06'
# data2 = b'\t\x07\x02\x00\t'
#
# print(data)
# data2 = bytes(data2)
#
# for i in range(len(data)):
#     print(data[i])

fpr1 = [0, 0.02310924, 0.02415966, 0.02556022, 0.02591036,  0.02731092, 1]
tpr1 = [0, 0.94138373, 0.94298527, 0.94330557, 0.94202434, 0.94202434, 1]
roc_auc1 = 0.96

fpr2 = [0, 0.04131653, 1]
tpr2 = [0, 0.92184497, 1]
roc_auc2 = 0.94

fpr3 = [0, 0.06757703, 1]
tpr3 = [0, 0.91672005, 1]
roc_auc3 = 0.92

fpr4 = [0, 0.03081232, 1]
tpr4 = [0, 0.93914158, 1]
roc_auc4 = 0.95

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, lw=2, label='ROC curve (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, lw=2, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, lw=2, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, lw=2, label='ROC curve (area = %0.2f)' % roc_auc4)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线（随机猜测的性能）
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()