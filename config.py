# -*- coding: utf-8 -*-
# 光速
C = 299702547

# 每个时间戳的间隔时间
Per_Stamp = 17.2074 / (2 ** 40)  # 秒

# 车辆长宽信息
CAR_WIDTH = 1.5
CAR_LENGTH = 3.0

# 基站数量
RX_NUM = 3

# 标签数量（车辆标签与人员标签之和）
TX_NUM = 4
# 车辆标签起始编号
TX_STR_NUM = 5
# 是否有人员标签
HAVE_HUM = False
# 人员数量
HUM_NUM = 1

# 是否采集数据（True为采集数据，False为使用数据集测试）
SAVE_DATA_FLAG = False
# 数据表名称（写与读）
W_DATA_FILE_NAME = "data/data9_21_test1.xls"
R_DATA_FILE_NAME = "data/data_demo1.xls"

# NLoS采集数据
NLOS_DATA_NAME = "data/nlos_test1.xlsx"

# 是否启用构建与校正矩形模块
BUILD_RECT_FLAG = True

# 安全阈值距离
THRES_DIS = 2.0

# 是否启用预警模块（启动蜂鸣器）
WARN_MOD_FLAG = False
# 是否开启滤波
FILTER_FLAG = True
# 是否开启转向平滑矫正
TOWARDS_COR_FLAG = True
# 是否开启车辆标签点渲染
CAR_TX_RENDER_FLAG = False

# 当前启用的定位算法（True为TDOA算法，False为TWR算法）
TDOA_FLAG = False

############### 设置基站坐标 ####################

# 体育馆
# rx2 = [0, 0, 0]
# rx3 = [52.6, 0, 0]
# rx1 = [26.3, 32.8, 0]

# 高尔夫场
# rx1 = [0, 0, 0]
# rx3 = [42, 0, 0]
# rx2 = [21, 16, 0]

# 5楼空地
# rx2 = [0, 0, 0]
# rx3 = [8.3, 0, 0]
# rx1 = [4.15, 7.2, 0]

# 实验室
# rx1 = [0, 0, 0]
# rx3 = [2, 0, 0]
# rx2 = [1, 1, 0]

# 东门附近空地
rx3 = [0, 0, 0]
rx2 = [75, 0, 0]
rx1 = [37.5, 38, 0]

# 房间
# rx1 = [1.0, 2.75, 0]
# rx2 = [0, 0, 0]
# rx3 = [2, 0, 0]

rx4 = [0, 0, 0]
