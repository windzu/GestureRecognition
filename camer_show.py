# import matplotlib.pyplot as plt
# import numpy as np


# # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
# plt.figure(figsize=(30, 6), dpi=80)
# plt.subplot(1, 1, 1)
# N = 27
# values = [0]*27
# index = np.arange(N)
# # 柱子的宽度
# width = 0.35

# # 设置横轴标签
# plt.xlabel('label')
# # 设置纵轴标签
# plt.ylabel('times')

# # 添加标题
# plt.title('gesture')
# # 添加纵横轴的刻度
# plt.xticks(index, ('Swiping Left',
#                    'Swiping Right',
#                    'Swiping Down',
#                    'Swiping Up',
#                    'Pushing Hand Away',
#                    'Pulling Hand In',
#                    'Sliding Two Fingers Left',
#                    'Sliding Two Fingers Right',
#                    'Sliding Two Fingers Down',
#                    'Sliding Two Fingers Up',
#                    'Pushing Two Fingers Away',
#                    'Pulling Two Fingers In',
#                    'Rolling Hand Forward',
#                    'Rolling Hand Backward',
#                    'Turning Hand Clockwise',
#                    'Turning Hand Counterclockwise',
#                    'Zooming In With Full Hand',
#                    'Zooming Out With Full Hand',
#                    'Zooming In With Two Fingers',
#                    'Zooming Out With Two Fingers',
#                    'Thumb Up',
#                    'Thumb Down',
#                    'Shaking Hand',
#                    'Stop Sign',
#                    'Drumming Fingers',
#                    'No gesture',
#                    'Doing other things',))
# plt.yticks(np.arange(0, 20, 1))
# # 绘制柱状图, 每根柱子的颜色为紫罗兰色
# p2 = plt.bar(index, values, width, label="labels", color="#87CEFA")
# plt.show()
# values = [10]*27
# p2 = plt.bar(index, values, width, label="labels", color="#87CEFA")
# plt.show()

# values = [5]*27
# p2 = plt.bar(index, values, width, label="labels", color="#87CEFA")
# plt.show()
