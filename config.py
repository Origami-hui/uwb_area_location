# -*- coding: utf-8 -*-
# 光速
import json

C = 299702547

# 每个时间戳的间隔时间
Per_Stamp = 17.2074 / (2 ** 40)  # 秒

# 主基站所在串口
RX_COM = 'COM4'

# 每个标签的定位频率
LOCATION_FREQ = 2

# 车辆长宽信息
CAR_WIDTH = 1.5
CAR_LENGTH = 3.0

# 基站数量
RX_NUM = 3

# 标签数量（车辆标签与人员标签之和）
TX_NUM = 1
# 车辆标签起始编号
TX_STR_NUM = 5
# 是否有人员标签
HAVE_HUM = True
# 人员数量
HUM_NUM = 1

# 是否采集数据（True为采集数据，False为使用数据集测试）
SAVE_DATA_FLAG = False
# 数据表名称（写与读）
W_DATA_FILE_NAME = "data/data_nlos_imu_1204-7.xls"
R_DATA_FILE_NAME = "data/data_nlos_imu_1204-5.xls"

# NLoS采集数据
NLOS_DATA_NAME = "nlos dataset/nlos case1204-2.csv"
# 是否处于NLoS
IN_NLOS_FLAG = True

# NLoS识别训练模型名称
NLOS_MODEL_NAME = "random_forest_model_test.starry"

# 是否启用构建与校正矩形模块
BUILD_RECT_FLAG = True

# 是否对NLoS场景进行误差补偿
NLOS_FIX_FLAG = False

# 安全阈值距离
THRES_DIS = 2.0

# 是否启用预警模块（启动蜂鸣器）
WARN_MOD_FLAG = False
# 是否开启滤波
FILTER_FLAG = True
# 是否开启转向平滑矫正
TOWARDS_COR_FLAG = True
# 是否开启车辆标签点渲染
CAR_TX_RENDER_FLAG = True

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
# rx3 = [0, 0, 0]
# rx2 = [75, 0, 0]
# rx1 = [37.5, 38, 0]

# 房间
# rx1 = [1.0, 2.75, 0]
# rx2 = [0, 0, 0]
# rx3 = [2, 0, 0]

# 1楼亭子
rx1 = [7.4, 17.9, 0]
rx2 = [0, 0, 0]
rx3 = [15, 0, 0]

rx4 = [0, 0, 0]


def set_config(config):
    globals().update(config)
    # print(globals())


def get_config():
    variables = globals()
    serializable_variables = {
        key: value for key, value in variables.items()
        if isinstance(value, (int, float, str, list, dict)) and key != '__file__' and key != 'variables'
           and key != 'serializable_variables' and key != '__name__' and key != '__annotations__'
    }
    return serializable_variables


# if __name__ == '__main__':
    # variables = globals()

    # 过滤掉不能被 JSON 序列化的对象
    # 比如函数、类和自定义对象等
    # serializable_variables = {
    #     key: value for key, value in variables.items()
    #     if isinstance(value, (int, float, str, list, dict)) and key != '__file__' and key != 'variables'
    #        and key != 'serializable_variables' and key != '__name__' and key != '__annotations__'
    # }
    # print(serializable_variables)
    # # 保存到 JSON 文件
    # with open('config.json', 'w') as f:
    #     json.dump(serializable_variables, f)
    # with open('config.json', 'r') as f:
    #     variables = json.load(f)
    #     set_config(variables)
