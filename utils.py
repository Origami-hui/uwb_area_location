import math

import joblib
import pandas as pd

import GJK
import numpy as np
import xlrd, xlwt
from numpy.linalg import inv, det

from main import *
from TDOAInterface import *
from config import *

preAngle = []  # 矩形角度缓存池
correctAngle = 0

clf_loaded = joblib.load(NLOS_MODEL_NAME)


def cul_tx_location(tag_id, dis_list):
    # 二维场景下计算标签坐标(x, y)
    # 默认以rx1为原点，rx1到rx2的向量为x轴建立坐标系

    if len(dis_list) == 3:

        [x0, y0] = two_point_location(dis_list[0], dis_list[1], dis_list[2], rx1, rx2, rx3)
        [x1, y1] = two_point_location(dis_list[1], dis_list[2], dis_list[0], rx2, rx3, rx1)
        [x2, y2] = two_point_location(dis_list[2], dis_list[0], dis_list[1], rx3, rx1, rx2)

        # 取三角形中心作为定位坐标
        x = (x0 + x1 + x2) / 3
        y = (y0 + y1 + y2) / 3

        x = round(x, 6)
        y = round(y, 6)

        return [x, y]

    elif len(dis_list) == 4:

        [x0, y0] = two_point_location(dis_list[0], dis_list[1], dis_list[2], rx1, rx2, rx3)
        [x1, y1] = two_point_location(dis_list[1], dis_list[2], dis_list[3], rx2, rx3, rx4)
        [x2, y2] = two_point_location(dis_list[2], dis_list[3], dis_list[0], rx3, rx4, rx1)
        [x3, y3] = two_point_location(dis_list[3], dis_list[0], dis_list[1], rx4, rx1, rx2)

        # 取中心作为定位坐标
        x = (x0 + x1 + x2 + x3) / 4
        y = (y0 + y1 + y2 + y3) / 4

        x = round(x, 6)
        y = round(y, 6)

        return [x, y]


def build_rectangle(X_list, Y_list, dem=4):
    # 根据前四个标签坐标的值（当前可见度为10个点），矫正当前构建矩形
    # 返回构建好的矩形四点坐标值
    X_copy = X_list.copy()
    Y_copy = Y_list.copy()

    # polygon = []
    re_x, re_y = [], []

    for i in range(min(dem, len(X_copy))):
        # polygon.append([X_copy[i][-1], Y_copy[i][-1]])
        re_x.append(X_copy[i][-1])
        re_y.append(Y_copy[i][-1])

    # re_x,re_y = coordinateCorrection(polygon)

    return re_x, re_y


def toPolygon(re_x, re_y):
    polygon = []
    for i in range(len(re_x)):
        polygon.append([re_x[i], re_y[i]])
    return polygon


def coordinateCorrection(polygon):
    global preAngle, correctAngle
    # 坐标矫正
    # 根据obsX，obsY（可观测的xy数据）或开始标定的车辆长宽（CAR_WIDTH，CAR_LENGTH）等先验信息进行矫正
    # 目标是矫正后的矩形长宽与车辆的长宽吻合，且与四个标签构成的多边形尽量贴合

    # 计算多边形的中心点
    center_x = sum(p[0] for p in polygon) / 4
    center_y = sum(p[1] for p in polygon) / 4

    # 计算多边形的旋转角度
    angle = math.atan2(polygon[1][1] - polygon[0][1], polygon[1][0] - polygon[0][0])
    # preAngle.append(angle)
    # # 扩容缓冲池

    # if(len(preAngle)>2):
    #     preAngle.pop(0)

    # if len(preAngle) > 1 and math.fabs(preAngle[0]-preAngle[1]) > 0.7:
    #     # 如果两次角度大于一定阈值，则判定为异常旋转角度
    #     print("异常旋转角度：", angle, preAngle)
    #     angle = correctAngle # 如果异常，取上次正常的角度作为旋转角度
    # else:
    #     correctAngle = angle # 如果正常，则将正常角度变量赋值为当前变量   

    # 将多边形旋转到x轴上方
    # rotation_matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    # rotated_polygon = [(rotation_matrix[0][0] * (p[0] - center_x) + rotation_matrix[0][1] * (p[1] - center_y),
    #                     rotation_matrix[1][0] * (p[0] - center_x) + rotation_matrix[1][1] * (p[1] - center_y)) for p in polygon]

    # # 找到多边形的最小矩形边界
    # min_x = min(p[0] for p in rotated_polygon)
    # max_x = max(p[0] for p in rotated_polygon)
    # min_y = min(p[1] for p in rotated_polygon)
    # max_y = max(p[1] for p in rotated_polygon)

    # # 缩放多边形，使其适合矩形
    # scale_x = CAR_WIDTH / (max_x - min_x)
    # scale_y = CAR_LENGTH / (max_y - min_y)
    # scaled_polygon = [((p[0] - min_x) * scale_x, (p[1] - min_y) * scale_y) for p in rotated_polygon]

    scaled_polygon = [[-CAR_WIDTH / 2, CAR_LENGTH / 2], [CAR_WIDTH / 2, CAR_LENGTH / 2],
                      [CAR_WIDTH / 2, -CAR_LENGTH / 2], [-CAR_WIDTH / 2, -CAR_LENGTH / 2]]

    # 将多边形旋转回原来的角度
    inverted_rotation_matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    final_polygon = [(inverted_rotation_matrix[0][0] * p[0] + inverted_rotation_matrix[0][1] * p[1] + center_x,
                      inverted_rotation_matrix[1][0] * p[0] + inverted_rotation_matrix[1][1] * p[1] + center_y) for p in
                     scaled_polygon]

    re_x = [p[0] for p in final_polygon]
    re_y = [p[1] for p in final_polygon]

    # 返回结果
    return re_x, re_y


def kalman_filter(obsX, obsY):
    # Define state vector
    # x = [pos_x, vel_x, pos_y, vel_y]
    x = np.array([obsX[0], 0, obsY[0], 0])

    # Define state transition matrix
    F = np.array([[1, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1]])

    # Define measurement matrix
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    # Define measurement noise covariance matrix
    R = np.array([[0.1, 0],
                  [0, 0.1]])

    # Define process noise covariance matrix
    Q = np.array([[0.01, 0, 0, 0],
                  [0, 0.01, 0, 0],
                  [0, 0, 0.01, 0],
                  [0, 0, 0, 0.01]])

    # Initialize covariance matrix
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Perform EKF.py prediction and update for each time step
    for i in range(len(obsX)):
        # Prediction step
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q

        # Update step
        z = np.array([obsX[i], obsY[i]])
        y = z - H.dot(x)
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(4) - K.dot(H)).dot(P)

    # Predict next position
    x = F.dot(x)
    return x[0], x[2]


# def kalman_filter(obsX):
#     # Define state vector
#     # x = [pos_x, vel_x, pos_y, vel_y]
#     x = np.array([obsX[0], 0])

#     # Define state transition matrix
#     F = np.array([[1, 1],
#                   [0, 1]])

#     # Define measurement matrix
#     H = np.array([[1, 0],
#                   [0, 0]])

#     # Define measurement noise covariance matrix
#     R = np.array([[0.1, 0],
#                   [0, 0.1]])

#     # Define process noise covariance matrix
#     Q = np.array([[0.01, 0],
#                   [0, 0.01]])

#     # Initialize covariance matrix
#     P = np.array([[1, 0],
#                   [0, 1]])

#     # Perform EKF.py prediction and update for each time step
#     for i in range(len(obsX)):
#         # Prediction step
#         x = F.dot(x)
#         P = F.dot(P).dot(F.T) + Q

#         # Update step
#         z = np.array([obsX[i]])
#         y = z - H.dot(x)
#         S = H.dot(P).dot(H.T) + R
#         K = P.dot(H.T).dot(np.linalg.inv(S))
#         x = x + K.dot(y)
#         P = (np.eye(2) - K.dot(H)).dot(P)

#     # Predict next position
#     x = F.dot(x)
#     return x[0]


def isAbnormalPoints(x, y, obsX, obsY):
    # 过滤异常点
    # 根据obsX，obsY（可观测的xy数据）以及极限坐标值矩阵limitMatrix（[max_x, min_x, max_y, min_y]）判断（x，y）坐标是否是异常值
    # 可以考虑输入更多的先验数据，使用卡尔曼滤波

    xp, yp = my_filter(obsX, obsY)
    # print(x, y, xp, yp)
    # if math.sqrt((xp-x)**2+(yp-y)**2) > 30.0:
    return isAbnormal(x, y, xp, yp)


def isAbnormal(x, y, tx, ty):
    # 过滤异常点
    if math.sqrt((tx - x) ** 2 + (ty - y) ** 2) > 20:
        return True
    return False


def my_filter(obsX, obsY):
    # 简单过滤函数
    avgX = sum(obsX) / len(obsX)
    avgY = sum(obsY) / len(obsY)

    # return avgX, avgY
    return obsX[-1], obsY[-1]


def collisionDetection(x_list, y_list, x, y):
    # 判断行人坐标(x, y)是否与多边形距离小于指定距离
    # 假定坐标顺时针分布，多边形为四边形或三角形

    # 扩展多边形，扩展半径为安全阈值距离
    if type(x) != list:
        # 多边形与点的碰撞检测
        re_x, re_y = extendArea(x_list, y_list)

        if len(x_list) == 4:

            AB = [re_x[1] - re_x[0], re_y[1] - re_y[0]]
            BC = [re_x[2] - re_x[1], re_y[2] - re_y[1]]
            CD = [re_x[3] - re_x[2], re_y[3] - re_y[2]]
            DA = [re_x[0] - re_x[3], re_y[0] - re_y[3]]

            if (AB[0] * (y - re_y[0]) - AB[1] * (x - re_x[0]) > 0 and BC[0] * (y - re_y[1]) - BC[1] * (
                    x - re_x[1]) > 0 and CD[0] * (y - re_y[2]) - CD[1] * (x - re_x[2]) > 0 and DA[0] * (y - re_y[3]) -
                DA[1] * (x - re_x[3]) > 0) \
                    or (AB[0] * (y - re_y[0]) - AB[1] * (x - re_x[0]) < 0 and BC[0] * (y - re_y[1]) - BC[1] * (
                    x - re_x[1]) < 0 and CD[0] * (y - re_y[2]) - CD[1] * (x - re_x[2]) < 0 and DA[0] * (y - re_y[3]) -
                        DA[1] * (x - re_x[3]) < 0):
                # 点在多边形内，不安全
                return re_x, re_y, False
            else:
                return re_x, re_y, True

        elif len(x_list) == 3:

            AB = [re_x[1] - re_x[0], re_y[1] - re_y[0]]
            BC = [re_x[2] - re_x[1], re_y[2] - re_y[1]]
            CA = [re_x[0] - re_x[2], re_y[0] - re_y[2]]

            if (AB[0] * (y - re_y[0]) - AB[1] * (x - re_x[0]) > 0 and BC[0] * (y - re_y[1]) - BC[1] * (
                    x - re_x[1]) > 0 and CA[0] * (y - re_y[2]) - CA[1] * (x - re_x[2]) > 0) \
                    or (AB[0] * (y - re_y[0]) - AB[1] * (x - re_x[0]) < 0 and BC[0] * (y - re_y[1]) - BC[1] * (
                    x - re_x[1]) < 0 and CA[0] * (y - re_y[2]) - CA[1] * (x - re_x[2]) < 0):
                # 点在多边形内，不安全
                return re_x, re_y, False
            else:
                return re_x, re_y, True
        else:
            return re_x, re_y, True
    else:
        # 多边形与多边形的碰撞检测，待测试
        re_x1, re_y1 = extendArea(x_list, y_list)
        re_x2, re_y2 = extendArea(x, y)
        shape = []
        shape.append([(re_x1[i], re_y1[i], 0) for i in range(len(re_x1))])
        shape.append([(re_x2[i], re_y2[i], 0) for i in range(len(re_x2))])
        print(shape)
        return [re_x1, re_x2], [re_y1, re_y2], GJK.gjk(shape)


def extendArea(re_x, re_y):
    if len(re_x) == 4:
        # 该多边形为四边形

        AB1 = [THRES_DIS * (re_y[0] - re_y[1]) / (math.sqrt((re_y[0] - re_y[1]) ** 2 + (re_x[1] - re_x[0]) ** 2)),
               THRES_DIS * (re_x[1] - re_x[0]) / (math.sqrt((re_y[0] - re_y[1]) ** 2 + (re_x[1] - re_x[0]) ** 2))]
        k1 = (re_y[1] - re_y[0]) / (re_x[1] - re_x[0])
        b1 = re_y[0] + AB1[1] - k1 * (re_x[0] + AB1[0])

        BC1 = [THRES_DIS * (re_y[1] - re_y[2]) / (math.sqrt((re_y[1] - re_y[2]) ** 2 + (re_x[2] - re_x[1]) ** 2)),
               THRES_DIS * (re_x[2] - re_x[1]) / (math.sqrt((re_y[1] - re_y[2]) ** 2 + (re_x[2] - re_x[1]) ** 2))]
        k2 = (re_y[2] - re_y[1]) / (re_x[2] - re_x[1])
        b2 = re_y[1] + BC1[1] - k2 * (re_x[1] + BC1[0])

        CD1 = [THRES_DIS * (re_y[2] - re_y[3]) / (math.sqrt((re_y[2] - re_y[3]) ** 2 + (re_x[3] - re_x[2]) ** 2)),
               THRES_DIS * (re_x[3] - re_x[2]) / (math.sqrt((re_y[2] - re_y[3]) ** 2 + (re_x[3] - re_x[2]) ** 2))]
        k3 = (re_y[3] - re_y[2]) / (re_x[3] - re_x[2])
        b3 = re_y[2] + CD1[1] - k3 * (re_x[2] + CD1[0])

        DA1 = [THRES_DIS * (re_y[3] - re_y[0]) / (math.sqrt((re_y[3] - re_y[0]) ** 2 + (re_x[0] - re_x[3]) ** 2)),
               THRES_DIS * (re_x[0] - re_x[3]) / (math.sqrt((re_y[3] - re_y[0]) ** 2 + (re_x[0] - re_x[3]) ** 2))]
        k4 = (re_y[0] - re_y[3]) / (re_x[0] - re_x[3])
        b4 = re_y[3] + DA1[1] - k4 * (re_x[3] + DA1[0])

        if k1 == k4 or k2 == k1 or k3 == k2 or k4 == k3:
            # 处理分母为0的情况
            return [], []

        A = [(b4 - b1) / (k1 - k4), (k1 * b4 - b1 * k4) / (k1 - k4)]
        B = [(b1 - b2) / (k2 - k1), (k2 * b1 - b2 * k1) / (k2 - k1)]
        C = [(b2 - b3) / (k3 - k2), (k3 * b2 - b3 * k2) / (k3 - k2)]
        D = [(b3 - b4) / (k4 - k3), (k4 * b3 - b4 * k3) / (k4 - k3)]

        ex_x = [A[0], B[0], C[0], D[0]]
        ex_y = [A[1], B[1], C[1], D[1]]

        return ex_x, ex_y
    elif len(re_x) == 3:

        # 该多边形为三角形
        AB1 = [THRES_DIS * (re_y[0] - re_y[1]) / (math.sqrt((re_y[0] - re_y[1]) ** 2 + (re_x[1] - re_x[0]) ** 2)),
               THRES_DIS * (re_x[1] - re_x[0]) / (math.sqrt((re_y[0] - re_y[1]) ** 2 + (re_x[1] - re_x[0]) ** 2))]
        k1 = (re_y[1] - re_y[0]) / (re_x[1] - re_x[0])
        b1 = re_y[0] + AB1[1] - k1 * (re_x[0] + AB1[0])

        BC1 = [THRES_DIS * (re_y[1] - re_y[2]) / (math.sqrt((re_y[1] - re_y[2]) ** 2 + (re_x[2] - re_x[1]) ** 2)),
               THRES_DIS * (re_x[2] - re_x[1]) / (math.sqrt((re_y[1] - re_y[2]) ** 2 + (re_x[2] - re_x[1]) ** 2))]
        k2 = (re_y[2] - re_y[1]) / (re_x[2] - re_x[1])
        b2 = re_y[1] + BC1[1] - k2 * (re_x[1] + BC1[0])

        CA1 = [THRES_DIS * (re_y[2] - re_y[0]) / (math.sqrt((re_y[2] - re_y[0]) ** 2 + (re_x[0] - re_x[2]) ** 2)),
               THRES_DIS * (re_x[0] - re_x[2]) / (math.sqrt((re_y[2] - re_y[0]) ** 2 + (re_x[0] - re_x[2]) ** 2))]
        k3 = (re_y[0] - re_y[2]) / (re_x[0] - re_x[2])
        b3 = re_y[2] + CA1[1] - k3 * (re_x[2] + CA1[0])

        if k1 == k3 or k2 == k1 or k3 == k2:
            # 处理分母为0的情况
            return [], []

        A = [(b3 - b1) / (k1 - k3), (k1 * b3 - b1 * k3) / (k1 - k3)]
        B = [(b1 - b2) / (k2 - k1), (k2 * b1 - b2 * k1) / (k2 - k1)]
        C = [(b2 - b3) / (k3 - k2), (k3 * b2 - b3 * k2) / (k3 - k2)]

        ex_x = [A[0], B[0], C[0]]
        ex_y = [A[1], B[1], C[1]]

        return ex_x, ex_y

    elif len(re_x) == 2:
        # 该多边形为直线
        SQRT = math.sqrt((re_x[1] - re_x[0]) ** 2 + (re_y[1] - re_y[0]) ** 2)

        return [[re_x[0] - THRES_DIS * (re_x[1] - re_x[0]) / SQRT, re_x[1] + THRES_DIS * (re_x[1] - re_x[0]) / SQRT],
                [re_y[0] - THRES_DIS * (re_y[1] - re_y[0]) / SQRT, re_y[1] + THRES_DIS * (re_y[1] - re_y[0]) / SQRT]]
    else:
        return


# 上一时刻的方向向量
pre_dx = 0
pre_dy = 0

# 记录上一时刻和当前中心坐标
avg_x_list = []
avg_y_list = []

# 设置初始阈值
DTHRE = 1


def towardsVDir(totalShake, re_x, re_y, dx, dy):
    # 根据当前速度向量方向旋转矩形方向
    global pre_dx, pre_dy, DTHRE

    avg_x = sum(re_x) / len(re_x)
    avg_y = sum(re_y) / len(re_y)

    halfwidth = CAR_WIDTH / 2
    halflength = CAR_LENGTH / 2

    # 单位化方向向量，用于绘制箭头，以及计算车辆4个顶点坐标
    if dx == 0 and dy == 0:
        unitized_dx = unitized_dy = 0
    else:
        unitized_dx = dx / (dx ** 2 + dy ** 2) ** 0.5
        unitized_dy = dy / (dx ** 2 + dy ** 2) ** 0.5

    if pre_dx == 0 and pre_dy == 0:
        pre_dx = dx
        pre_dy = dy

    # print(totalShake)

    # 当前时刻车辆中心位置放入avg_x_list， avg_x_list是全局变量，初始为空
    avg_x_list.append(avg_x)
    avg_y_list.append(avg_y)

    # 第一次进入循环，avg_x_list插入一个元素（会导致车辆在开始时刻总是指向右上，有待改进）
    if len(avg_x_list) < 2:
        avg_x_list.insert(0, avg_x - 0.1)
        avg_y_list.insert(0, avg_y - 0.1)
        pre_dx = 0.1
        pre_dy = 0.1
    # 第二次及之后进入循环，保持avg_x_list长度为2
    if len(avg_x_list) > 2:
        avg_x_list.pop(0)
        avg_y_list.pop(0)

    if dx ** 2 + dy ** 2 > DTHRE:  # 设置方向变化阈值

        # print(dx**2+dy**2, DTHRE)
        # 采用梯度变化(坐标变化越块，方向改变越快，)
        # TODO 在低速行驶情况下让转向更符合轨迹方向
        if dx ** 2 + dy ** 2 > DTHRE + 0.2:
            dx = 0.14 * dx + 0.86 * pre_dx
            dy = 0.14 * dy + 0.86 * pre_dy
        elif dx ** 2 + dy ** 2 > DTHRE + 0.1:
            dx = 0.12 * dx + 0.88 * pre_dx
            dy = 0.12 * dy + 0.88 * pre_dy
        elif dx ** 2 + dy ** 2 > DTHRE:
            dx = 0.1 * dx + 0.9 * pre_dx
            dy = 0.1 * dy + 0.9 * pre_dy

        # if dx ** 2 + dy ** 2 > DTHRE + 0.2:
        #     dx = 0.4 * dx + 0.6 * pre_dx
        #     dy = 0.4 * dy + 0.6 * pre_dy
        # elif dx ** 2 + dy ** 2 > DTHRE + 0.1:
        #     dx = 0.3 * dx + 0.7 * pre_dx
        #     dy = 0.3 * dy + 0.7 * pre_dy
        # elif dx ** 2 + dy ** 2 > DTHRE:
        #     dx = 0.2 * dx + 0.8 * pre_dx
        #     dy = 0.2 * dy + 0.8 * pre_dy

        # 更新向量
        unitized_dx = dx / (dx ** 2 + dy ** 2) ** 0.5
        unitized_dy = dy / (dx ** 2 + dy ** 2) ** 0.5

        # 矫正车头方向，根据单位化后的方向向量和车两半长半宽计算车辆4个顶点坐标
        re_x[0] = avg_x - unitized_dx * halflength + unitized_dy * halfwidth
        re_x[3] = avg_x + unitized_dx * halflength + unitized_dy * halfwidth
        re_x[2] = avg_x + unitized_dx * halflength - unitized_dy * halfwidth
        re_x[1] = avg_x - unitized_dx * halflength - unitized_dy * halfwidth
        # X_switch[4] = avg_x - unitized_dx * halflength + unitized_dy * halfwidth
        re_y[0] = avg_y - unitized_dy * halflength - unitized_dx * halfwidth
        re_y[3] = avg_y + unitized_dy * halflength - unitized_dx * halfwidth
        re_y[2] = avg_y + unitized_dy * halflength + unitized_dx * halfwidth
        re_y[1] = avg_y - unitized_dy * halflength + unitized_dx * halfwidth
        # re_y[4] = avg_y - unitized_dy * halflength - unitized_dx * halfwidth

        # 更新pre_dx
        pre_dx = dx
        pre_dy = dy
        # 增大阈值，与下边减小阈值对应（重点在减小阈值）
        DTHRE += 0.05
        # print("高速情况")

    elif totalShake < 0.1:
        # 利用标签抖动判断车辆是否已经停止
        # print(dx**2+dy**2)
        x = re_x[1] - re_x[2]
        y = re_y[1] - re_y[2]
        # 更新向量
        unitized_dx = x / (x ** 2 + y ** 2) ** 0.5
        unitized_dy = y / (x ** 2 + y ** 2) ** 0.5
        # print("静止情况")
    else:
        # 车辆位置改变小于阈值，认为车辆方向未发生改变
        if dx == 0 and dy == 0:
            dx = pre_dx
            dy = pre_dy
        else:
            dx = 0.9 * dx + 0.1 * pre_dx
            dy = 0.9 * dy + 0.1 * pre_dy

        # 更新向量
        unitized_dx = dx / (dx ** 2 + dy ** 2) ** 0.5
        unitized_dy = dy / (dx ** 2 + dy ** 2) ** 0.5

        # 矫正车头方向，根据单位化后的方向向量和车两半长半宽计算车辆4个顶点坐标
        re_x[0] = avg_x - unitized_dx * halflength + unitized_dy * halfwidth
        re_x[3] = avg_x + unitized_dx * halflength + unitized_dy * halfwidth
        re_x[2] = avg_x + unitized_dx * halflength - unitized_dy * halfwidth
        re_x[1] = avg_x - unitized_dx * halflength - unitized_dy * halfwidth
        # X_switch[4] = avg_x - unitized_dx * halflength + unitized_dy * halfwidth
        re_y[0] = avg_y - unitized_dy * halflength - unitized_dx * halfwidth
        re_y[3] = avg_y + unitized_dy * halflength - unitized_dx * halfwidth
        re_y[2] = avg_y + unitized_dy * halflength + unitized_dx * halfwidth
        re_y[1] = avg_y - unitized_dy * halflength + unitized_dx * halfwidth
        # Y_switch[4] = avg_y - unitized_dy * halflength - unitized_dx * halfwidth

        # if dx**2+dy**2 > DTHRE * 3 / 4:
        #     # 更新pre_dx
        #     print("更新pre_dx")
        #     pre_dx = dx
        #     pre_dy = dy

        # 减小阈值，防止车辆长期慢速移动，方向实际发生改变，而图中车辆方向由于小于阈值而不发生改变
        DTHRE -= 0.1
        # print("低速情况")

    return re_x, re_y, unitized_dx, unitized_dy


# 计算静态方差
def cul_static_variance(obsX, obsY):
    d = []
    for i in range(len(obsX)):
        d.append(math.sqrt(obsX[i] ** 2 + obsY[i] ** 2))

    var = np.var(d)
    # print("抖动方差：", var)
    return var


# 计算车身转向抖动方差
def cul_towards_variance(towards_list):
    var = np.var(towards_list)
    return var


def two_point_location(disa, disb, disc, rxa, rxb, rxc):
    # 根据disa与disb计算坐标，用rxc来辅助

    m = -(disa ** 2 - disb ** 2 + (rxa[0] - rxb[0]) ** 2 - (rxa[1] ** 2 - rxb[1] ** 2)) / (2 * (rxa[0] - rxb[0]))
    A = ((rxa[1] - rxb[1]) / (rxa[0] - rxb[0])) ** 2 + 1
    B = 2 * (-m * (rxa[1] - rxb[1]) / (rxa[0] - rxb[0]) - rxa[1])
    C = rxa[1] ** 2 + m ** 2 - disa ** 2

    delta = B ** 2 - 4 * A * C
    delta = math.fabs(delta)

    y1 = (-B + math.sqrt(delta)) / (2 * A)
    y2 = (-B - math.sqrt(delta)) / (2 * A)

    x1 = m + rxa[0] - (rxa[1] - rxb[1]) / (rxa[0] - rxb[0]) * y1
    x2 = m + rxa[0] - (rxa[1] - rxb[1]) / (rxa[0] - rxb[0]) * y2

    err1 = math.fabs(disc - math.sqrt((rxc[0] - x1) ** 2 + (rxc[1] - y1) ** 2))
    err2 = math.fabs(disc - math.sqrt((rxc[0] - x2) ** 2 + (rxc[1] - y2) ** 2))

    return [x1, y1] if err1 < err2 else [x2, y2]


# 对列表内数据进行卡尔曼滤波处理
def smooth_list(values):
    for i in range(len(values) - 8):
        obs = values[i:i + 8]
        x = kalman_filter(obs)
        if abs(values[i + 8] - x) > 2:
            values[i + 8] = round(x, 6)
    return values


# # 对数据做滤波和平滑处理，将数据保存到新的xls文件
# def smooth(obsX, obsY):

#     # 将数据复制到new_x_values中，之后对new_x_values做滤波处理
#     new_x_values = obsX
#     new_y_values = obsY
#     # 滤波处理
#     new_x_values = smooth_list(new_x_values)
#     new_y_values = smooth_list(new_y_values)

#     # 比较滤波处理后的数据和原数据，设置计数器，当计数器超过阈值，表明后续的数据发生了整体偏移
#     count_x = 0
#     count_y = 0

#     for index in range(len(obsX)):
#         if obsX[index] != new_x_values[index]:
#             count_x += 1
#         else:
#             count_x = 0

#         # 发生整体偏移后，说明滤波处理的数据有误，从偏移点开始用原数据覆盖处理后的数据，并从偏移点开始重新滤波处理。并不断重复上述步骤
#         if count_x == 8:
#             x_slice = obsX[index-7:]
#             if len(x_slice) < 8:
#                 new_x_values[-len(x_slice):] = x_slice
#                 break
#             x_slice = smooth_list(x_slice)
#             new_x_values[-len(x_slice):] = x_slice
#             count_x = 0

#         if obsY[index] != new_y_values[index]:
#             count_y += 1
#         else:
#             count_y = 0
#         if count_y == 8:
#             y_slice = obsY[index-7:]
#             if len(y_slice) < 8:
#                 new_y_values[-len(y_slice):] = y_slice
#                 break
#             y_slice = smooth_list(y_slice)
#             new_y_values[-len(y_slice):] = y_slice
#             count_y = 0

#     # 对滤波无法处理的跳变点，用分摊的方式平滑跳变
#     for index in range(2, len(new_x_values)-2):
#         if abs(new_x_values[index]-new_x_values[index-1] > 1):
#             step = (new_x_values[index+2] - new_x_values[index-3])/5
#             for j in range(4):
#                 new_x_values[index+j-2] = new_x_values[index-3] + (j+1)*step
#             index += 3

#     for index in range(2, len(new_y_values)-2):
#         if abs(new_y_values[index]-new_y_values[index-1] > 1):
#             step = (new_y_values[index+2] - new_y_values[index-3])/5
#             for j in range(4):
#                 new_y_values[index+j-2] = new_y_values[index-3] + (j+1)*step
#             index += 3

#     return new_x_values, new_y_values

# 将xls文件中的数据做滤波和平滑处理，将数据保存到新的xls文件
def smooth(data, new_data):
    # 打开 Excel 文件
    workbook = xlrd.open_workbook(data)
    new_workbook = xlwt.Workbook(encoding='utf-8')
    # 获取第一个工作表
    worksheet = workbook.sheet_by_index(0)
    new_worksheet = new_workbook.add_sheet('Sheet1', cell_overwrite_ok=True)

    # 对xls文件的每一列数据分别处理
    for i in range(4):
        # 获取 i*3 列的行数
        m_col_rows = worksheet.nrows - worksheet.col_values(i * 3).count('')

        # 将一列的数据存放到x_values和y_values中
        x_values = []
        y_values = []

        for row_index in range(1, m_col_rows + 1):
            x_value = worksheet.cell_value(row_index, i * 3)
            y_value = worksheet.cell_value(row_index, i * 3 + 1)
            x_values.append(x_value)
            y_values.append(y_value)

        # 将数据复制到new_x_values中，之后对new_x_values做滤波处理
        new_x_values = x_values[:]
        new_y_values = y_values[:]
        # 滤波处理
        new_x_values = smooth_list(new_x_values)
        new_y_values = smooth_list(new_y_values)

        # 比较滤波处理后的数据和原数据，设置计数器，当计数器超过阈值，表明后续的数据发生了整体偏移
        count_x = 0
        count_y = 0

        for index in range(len(x_values)):
            if x_values[index] != new_x_values[index]:
                count_x += 1
            else:
                count_x = 0

            # 发生整体偏移后，说明滤波处理的数据有误，从偏移点开始用原数据覆盖处理后的数据，并从偏移点开始重新滤波处理。并不断重复上述步骤
            if count_x == 8:
                x_slice = x_values[index - 7:]
                if len(x_slice) < 8:
                    new_x_values[-len(x_slice):] = x_slice
                    break
                x_slice = smooth_list(x_slice)
                new_x_values[-len(x_slice):] = x_slice
                count_x = 0

            if y_values[index] != new_y_values[index]:
                count_y += 1
            else:
                count_y = 0
            if count_y == 8:
                y_slice = y_values[index - 7:]
                if len(y_slice) < 8:
                    new_y_values[-len(y_slice):] = y_slice
                    break
                y_slice = smooth_list(y_slice)
                new_y_values[-len(y_slice):] = y_slice
                count_y = 0

        # 对滤波无法处理的跳变点，用分摊的方式平滑跳变
        for index in range(2, len(new_x_values) - 2):
            if abs(new_x_values[index] - new_x_values[index - 1] > 1):
                step = (new_x_values[index + 2] - new_x_values[index - 3]) / 5
                for j in range(4):
                    new_x_values[index + j - 2] = new_x_values[index - 3] + (j + 1) * step
                index += 3

        for index in range(2, len(new_y_values) - 2):
            if abs(new_y_values[index] - new_y_values[index - 1] > 1):
                step = (new_y_values[index + 2] - new_y_values[index - 3]) / 5
                for j in range(4):
                    new_y_values[index + j - 2] = new_y_values[index - 3] + (j + 1) * step
                index += 3

        # 将处理后的数据保存到新工作表
        for index, item in enumerate(new_x_values):
            new_worksheet.write(index + 1, i * 3, item)
        for index, item in enumerate(new_y_values):
            new_worksheet.write(index + 1, i * 3 + 1, item)

    # 保存新文件
    new_workbook.save(new_data)


# ===================TDOA 部分====================


"""---------------------测线可能性分析-------------------"""


def Distance2_3D(pos_1, pos_2):
    if len(pos_1) == 3 and len(pos_2) == 3:
        distance = distance3D(pos_1, pos_2)
    else:
        pos_1 = pos_1[0:2]
        pos_2 = pos_2[0:2]
        distance = distance2D(pos_1, pos_2)
    return distance


"""------------------------低通滤波-----------------------"""


def Low_Pass_Filter(result, last, d, k):
    x_last = result[0] - last[0]
    y_last = result[1] - last[1]

    # print("差值为:", result, last)

    # if abs(x_last)<d:
    #     result[0]=last[0]+x_last*k
    #     print("修正X")
    # if abs(y_last)<d:
    #     result[1]=last[1]+y_last*k
    #     print("修正Y")

    if math.sqrt(x_last ** 2 + y_last ** 2) < d:
        # print("修正坐标")
        result[0] = last[0] + x_last * k
        result[1] = last[1] + y_last * k

    return result, result


"""------------------------抖动分析仪-------------------"""

# def DataAnalysis(Addrs,main_addr,Stamps):
#     for i in Addrs:
#         if i==main_addr:
#             continue        
#         if i not in analysis.keys():
#             analysis[i]=[]
#         else:
#             index_i=Addrs.index(i)
#             index_main=Addrs.index(main_addr)
#             stamp_i=Stamps[index_i]
#             stamp_main=Stamps[index_main]
#             analysis[i].append(stamp_i-stamp_main)
#     #print(analysis)
#     #time.sleep(0.3)


"""-------------------现实坐标转图像坐标------------------------"""


def axis_turn(a, b, img):  # 传入具体的坐标 转化为图像坐标
    x = int(img.shape[1] * a / 157.39)
    y = img.shape[0] - int(img.shape[0] * b / 91.52)
    return (x, y)


"""-------------------专用于地砖转换------------------------"""


def axis_turn_floor(a, b, img):  # 传入具体的坐标 转化为图像坐标
    global radius, floor_x, floor_y
    x = int(img.shape[1] * a / (radius * floor_x))
    y = img.shape[0] - int(img.shape[0] * b / (radius * floor_y))
    return (x, y)


"""-------------------------距离求取------------------------"""


def distance2D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


"""-----------------------3维距离求取------------------------"""


def distance3D(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


"""-----------------------2d辖区查询------------------------"""


def in_area(position, area):  # position是个数组  area是个数组数组
    x1, x2 = area[0][0], area[1][0]
    y1, y2 = area[0][1], area[1][1]
    # print(position)

    return position[0] >= x1 and position[0] <= x2 and position[1] >= y1 and position[1] <= y2


"""-----------------------一元二次方程求解------------------------"""


# Fang算法中解方程
def solve_equations(a, b, c):
    d = b ** 2 - 4 * a * c
    if (d < 0):
        print("无解")
        return 0, 0
    else:
        e = math.sqrt(d)
        x1 = ((-b + e) / (2 * a))
        x2 = ((-b - e) / (2 * a))
        return x1, x2


"""-----------------------坐标旋转2D------------------------"""


def rotate(BS):
    # len3 且第一个为[0,0]

    if (BS[1][1] == 0):
        return BS, 0
    if (BS[2][1] == 0):
        # temp=BS[1]
        # BS[1]=BS[2]
        # BS[2]=temp
        # return BS,0
        BS[2][1] == 0.001

    # 先求夹角
    axis_x = [1, 0]
    length = distance2D(BS[1], BS[0])

    product = BS[1][0]

    cos = product / length
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 正y顺时针(反转)
    if BS[1][1] > 0:
        radian = -1 * radian
        angle = -1 * radian

    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21

    BS = [[R11 * i[0] + R12 * i[1], R21 * i[0] + R22 * i[1]] for i in BS]

    return [BS, -radian]  # 用于转回


"""-----------------------坐标转回2D------------------------"""


def rotate_back(BS, radian):
    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21

    BS = [R11 * BS[0] + R12 * BS[1], R21 * BS[0] + R22 * BS[1]]

    return BS


"""-----------------------List包含关系------------------------"""


def List_in(a, b):
    for i in a:
        if i in b:
            continue
        else:
            return False
    return True


"""----------------------求取包含数组的所在Index---------------"""


def List_in_index(a, b):
    in_index = [b.index(i) for i in a]
    return in_index


"""----------------------检查是否使用到了主机站---------------"""


def have_main(array, dict):
    for i in array:
        if dict[i] == 1:
            return True
        else:
            continue
    return False


"""-----------------------坐标旋转3D------------------------"""


def rotate_3D(BS):
    # 数据修正
    for i in range(len(BS)):
        for j in range(len(BS[i])):
            if BS[i][j] == 0:
                BS[i][j] = 0.001

    # 第一步 将第二个点绕着y轴旋转到xOy上
    P2_xz = [BS[1][0], BS[1][2]]  # P2[x,z]

    # print(P2_xz)
    length = distance2D(P2_xz, [0, 0])
    product = P2_xz[0]

    if length == 0:
        length = 0.001
    cos = product / length  # P2绕X轴旋转至xOy平面上
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 负z顺时针(反转)
    if BS[1][2] < 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕y轴旋转矩阵

    R33 = R11 = np.cos(radian)
    R22 = 1
    R13 = np.sin(radian)
    R31 = -R13

    BS = [[R11 * i[0] + R13 * i[2], R22 * i[1], R31 * i[0] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian1 = -radian
    # print(BS)

    # 第二步 将第二个点绕着z轴旋转到x轴上

    # 先求夹角

    P2_xy = [BS[1][0], BS[1][1]]
    length = distance2D(P2_xy, BS[0])

    product = P2_xy[0]

    cos = product / length
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 正y顺时针(反转)
    if BS[1][1] > 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕z轴旋转矩阵

    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21
    R33 = 1

    BS = [[R11 * i[0] + R12 * i[1], R21 * i[0] + R22 * i[1], R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian2 = -radian
    # print(BS)

    # 第三步 将第三个点绕着x轴旋转到xOy上

    P3_yz = [BS[2][1], BS[2][2]]  # P2[x,z]

    length = distance2D(P3_yz, [0, 0])
    product = P3_yz[0]

    cos = product / length  # P2绕X轴旋转至xOy平面上
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 正z顺时针(反转)
    if BS[2][2] > 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕x轴旋转矩阵

    R11 = 1
    R22 = R33 = np.cos(radian)
    R32 = np.sin(radian)
    R23 = -R32

    BS = [[R11 * i[0], R22 * i[1] + R23 * i[2], R32 * i[1] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian3 = -radian
    # print(BS)
    # 此时若有4组数，已满足Fang3D运算需求

    return [BS, radian1, radian2, radian3]


"""-----------------------坐标转回3D------------------------"""


def rotate_back_3D(BS, radian1, radian2, radian3):
    radian = radian3
    R11 = 1
    R22 = R33 = np.cos(radian)
    R32 = np.sin(radian)
    R23 = -R32

    # print("输入参数为:",BS)
    BS = [R11 * BS[0], R22 * BS[1] + R23 * BS[2], R32 * BS[1] + R33 * BS[2]]
    BS = [round(i, 2) for i in BS]
    # print(BS)

    radian = radian2
    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21
    R33 = 1

    BS = [R11 * BS[0] + R12 * BS[1], R21 * BS[0] + R22 * BS[1], R33 * BS[2]]
    BS = [round(i, 2) for i in BS]
    # print(BS)

    radian = radian1
    R33 = R11 = np.cos(radian)
    R22 = 1
    R13 = np.sin(radian)
    R31 = -R13

    BS = [R11 * BS[0] + R13 * BS[2], R22 * BS[1], R31 * BS[0] + R33 * BS[2]]
    BS = [round(i, 2) for i in BS]

    # print(BS)

    return BS


"""-----------------------坐标转回3D/多个坐标------------------------"""


def rotate_back_3D_list(BS, radian1, radian2, radian3):
    radian = radian3
    R11 = 1
    R22 = R33 = np.cos(radian)
    R32 = np.sin(radian)
    R23 = -R32

    # print("输入参数为:",BS)
    BS = [[R11 * i[0], R22 * i[1] + R23 * i[2], R32 * i[1] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]
    # print(BS)

    radian = radian2
    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21
    R33 = 1

    BS = [[R11 * i[0] + R12 * i[1], R21 * i[0] + R22 * i[1], R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]
    # print(BS)

    radian = radian1
    R33 = R11 = np.cos(radian)
    R22 = 1
    R13 = np.sin(radian)
    R31 = -R13

    BS = [[R11 * i[0] + R13 * i[2], R22 * i[1], R31 * i[0] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    # print(BS)

    return BS


"""---------------------坐标旋转3D_debug------------------------"""


def rotate_3D_debug(BS, position):
    BS.append(position)
    # 第一步 将第二个点绕着y轴旋转到xOy上
    P2_xz = [BS[1][0], BS[1][2]]  # P2[x,z]

    print(P2_xz)
    length = distance2D(P2_xz, [0, 0])
    product = P2_xz[0]

    if length == 0:
        length = 0.001
    cos = product / length  # P2绕X轴旋转至xOy平面上
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 负z顺时针(反转)
    if BS[1][2] < 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕y轴旋转矩阵

    R33 = R11 = np.cos(radian)
    R22 = 1
    R13 = np.sin(radian)
    R31 = -R13

    BS = [[R11 * i[0] + R13 * i[2], R22 * i[1], R31 * i[0] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian1 = -radian
    # print(BS)

    # 第二步 将第二个点绕着z轴旋转到x轴上

    # 先求夹角

    P2_xy = [BS[1][0], BS[1][1]]
    length = distance2D(P2_xy, BS[0])

    product = P2_xy[0]

    cos = product / length
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 正y顺时针(反转)
    if BS[1][1] > 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕z轴旋转矩阵

    R22 = R11 = np.cos(radian)
    R21 = np.sin(radian)
    R12 = -R21
    R33 = 1

    BS = [[R11 * i[0] + R12 * i[1], R21 * i[0] + R22 * i[1], R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian2 = -radian
    # print(BS)

    # 第三步 将第三个点绕着x轴旋转到xOy上

    P3_yz = [BS[2][1], BS[2][2]]  # P2[x,z]

    length = distance2D(P3_yz, [0, 0])
    product = P3_yz[0]

    cos = product / length  # P2绕X轴旋转至xOy平面上
    radian = np.arccos(cos)  # 弧度
    angle = radian * 180 / np.pi  # 弧度=角度/180*pai

    # 正z顺时针(反转)
    if BS[2][2] > 0:
        radian = -1 * radian
        angle = -1 * radian

    # 构造绕x轴旋转矩阵

    R11 = 1
    R22 = R33 = np.cos(radian)
    R32 = np.sin(radian)
    R23 = -R32

    BS = [[R11 * i[0], R22 * i[1] + R23 * i[2], R32 * i[1] + R33 * i[2]] for i in BS]
    BS = [[round(j, 2) for j in i] for i in BS]

    radian3 = -radian
    # print(BS)
    # 此时若有4组数，已满足Fang3D运算需求

    return [BS, radian1, radian2, radian3]


## =========================== TDOA算法部分 ===============================

# Fang的外壳，修正时间戳-旋转-Fang-转回
def Wrap_Fang2D(Addrs, Stamps):
    BS = []  # 坐标数组
    rx_num = 3  # len()
    refer_addr = Addrs[0]

    for addr in Addrs:  # 取BS
        BS.append(RX_position[addr])

    for i in range(rx_num):
        distance = distance2D(RX_position[Addrs[i]], RX_position[refer_addr])
        add_stamp = distance / C / Per_Stamp
        Stamps[i] = Stamps[i] + add_stamp  # 添加距离补偿

    # print("Sim_fang")
    BSN = 3

    # 不开启尾处理、辖区判定  先把程序写明白

    R = []
    for i in range(1, BSN):
        R.append(C * Per_Stamp * (Stamps[i] - Stamps[0]))

    BS0 = BS[0]

    # 平移
    BS = [[i[0] - BS[0][0], i[1] - BS[0][1]] for i in BS]
    # 旋转
    BS, radian = rotate(BS)

    result, other = Fang(BS, R)
    if result == [0, 0]:
        return [0, 0], [0, 0]
    # 转回
    result = rotate_back(result, radian)
    other = rotate_back(other, radian)
    # 平移回
    result = [result[0] + BS0[0], result[1] + BS0[1]]
    other = [other[0] + BS0[0], other[1] + BS0[1]]
    return result, other


# 纯侧线方法
def Straight(BS, R):
    # 两个基站坐标  1个距离差  到R2的距离-到R1的距离

    straight_length = distance2D(BS[0], BS[1])  # 直线距离
    To_BS0 = (straight_length - R[0]) / 2
    diff_x = BS[1][0] - BS[0][0]
    diff_y = BS[1][1] - BS[0][1]
    portion = To_BS0 / straight_length

    if portion >= 1:
        portion = 1
    if portion <= 0:
        portion = 0

    # 需不需要限制portion
    x = BS[0][0] + portion * diff_x
    y = BS[0][1] + portion * diff_y

    x = round(x, 2)
    y = round(y, 2)

    result = [x, y]

    return result


# Fang的尾处理方法  且仅对某一轴向进行尾处理
def tail_find(BS, Stamps):
    # len 均为3
    for i in range(3):
        distance = distance2D(BS[i], BS[(i + 1) // 3])  # 0-1 1-2 2-3
        if distance <= 5:  # 进行尾处理
            break
    else:
        return False, 0, 0

    # 拿到了i
    p_1 = BS[i]
    p_2 = BS[(i + 1) // 3]
    # 哪个轴需要尾处理   axis=0 x轴  1 y轴
    # 通常设置为一个轴向坐标相同  另一个轴相错开
    if (abs(p_1[0] - p_2[0]) <= 1.3):
        axis = 1
    else:
        axis = 0

    if Stamps[i] == Stamps[(i + 1) // 3]:
        revise = (p_1[axis] + p_2[axis]) / 2

    elif Stamps[i] < Stamps[(i + 1) // 3]:

        Differ = (p_2[axis] - p_1[axis]) / 4
        revise = p_1[axis] + Differ
    else:
        Differ = (p_1[axis] - p_2[axis]) / 4
        revise = p_2[axis] + Differ

    return True, axis, revise


def Fang(BS, R):
    # len3 len2     R注意角标加2

    for i in range(len(BS)):
        for j in range(len(BS[i])):
            if BS[i][j] == 0:
                BS[i][j] = 0.001

    for i in range(len(R)):
        if R[i] == 0:
            R[i] = 0.001

    # 求gh
    temp = 0
    g = (R[1] * BS[1][0] / R[0] - BS[2][0]) / BS[2][1]

    temp = (BS[2][0] ** 2) + (BS[2][1] ** 2) - (R[1] ** 2)
    temp = temp + R[1] * R[0] * (1 - ((BS[1][0] / R[0]) ** 2))

    h = (temp / 2) / BS[2][1]

    # 求def
    temp = 0
    temp = (BS[1][0] / R[0]) ** 2
    d = -1 * ((1 - temp) + g * g)
    e = BS[1][0] * (1 - temp) - 2 * g * h
    temp = (R[0] ** 2) - (BS[1][0] ** 2)  # 先往里乘了个R0**4
    temp = temp ** 2

    f = (temp / (R[0] ** 2)) / 4 - h * h

    x1, x2 = solve_equations(d, e, f)
    if (x1 == x2 == 0):
        return [0, 0], [0, 0]

    y1 = g * x1 + h;
    y2 = g * x2 + h;

    # print("两个x:",x1,x2)
    # print("两个y:",y1,y2)

    tempR2_1 = math.sqrt(((x1 - BS[1][0]) ** 2) + (y1 ** 2))
    tempR1_1 = math.sqrt((x1 ** 2) + (y1 ** 2))
    tempslotR_1 = abs((tempR2_1 - tempR1_1) - R[0])

    tempR2_2 = math.sqrt(((x2 - BS[1][0]) ** 2) + (y2 ** 2))
    tempR1_2 = math.sqrt((x2 ** 2) + (y2 ** 2))
    tempslotR_2 = abs((tempR2_2 - tempR1_2) - R[0])

    tempR3_1 = math.sqrt(((x1 - BS[2][0]) ** 2) + ((y1 - BS[2][1]) ** 2))
    tempslotR_3 = abs((tempR3_1 - tempR1_1) - R[1])

    tempR3_2 = math.sqrt(((x2 - BS[2][0]) ** 2) + ((y2 - BS[2][1]) ** 2))
    tempslotR_4 = abs((tempR3_2 - tempR1_2) - R[1])

    # print(tempslotR_1,tempslotR_2)
    # print(tempslotR_3,tempslotR_4)
    # if(tempslotR_1<tempslotR_2):

    # print("Fang解算出的两次结果分别为",x1,y1,x2,y2)

    if (tempslotR_1 + tempslotR_3 < tempslotR_2 + tempslotR_4):
        result1 = [x1, y1]
        result2 = [x2, y2]
    else:
        result1 = [x2, y2]
        result2 = [x1, y1]

    return result1, result2


# 需要三维旋转
def Fang3D(BS, R):
    # 数据修正
    for i in range(len(BS)):
        for j in range(len(BS[i])):
            if BS[i][j] == 0:
                BS[i][j] = 0.001
    # print(BS)

    for i in range(len(R)):
        if R[i] == 0:
            R[i] = 0.001

    # len4    Ri,1:len3
    g = (R[1] * BS[1][0] / R[0] - BS[2][0]) / BS[2][1]
    # 我自己推是x3

    temp = 0
    temp = (BS[2][0] ** 2) + (BS[2][1] ** 2) - (R[1] ** 2)
    temp = temp + R[1] * R[0] * (1 - ((BS[1][0] / R[0]) ** 2))

    h = (temp / 2) / BS[2][1]  # y4  应该是y3!

    # print("g,h:",g,h)

    temp = 0
    temp = R[2] * BS[1][0] * BS[2][1] - R[0] * BS[3][0] * BS[2][1]
    temp = temp - R[1] * BS[1][0] * BS[3][1] + R[0] * BS[2][0] * BS[3][1]  # 最后一项R21
    # print(R[0],BS[2][1],BS[3][2])
    k = temp / R[0] / BS[2][1] / BS[3][2]

    temp = 0
    # temp=R[0]*(BS[3][0]**2+BS[3][1]**2+BS[3][2]**2)*BS[2][1]
    # l=R[0]*(R[1]**2)*BS[2][1]-(R[2]**2)*R[1]*BS[2][1]-R[2]*(BS[1][0]**2)*BS[2][1]
    # l=l+temp
    temp = R[2] * (R[0] ** 2 - BS[1][0] ** 2)
    temp = temp - R[0] * (R[2] ** 2 - ((BS[3][0] ** 2) + (BS[3][1] ** 2) + (BS[3][2] ** 2)))
    temp = temp - 2 * R[0] * BS[3][1] * h

    l = temp / 2 / R[0] / BS[3][2]

    # print("k,l:",k,l)

    temp = 0
    temp = R[0] ** 2 + (R[0] * g) ** 2 + (R[0] * k) ** 2 - BS[1][0] ** 2
    d = temp * 4

    temp = 0
    temp = 4 * BS[1][0] * (R[0] ** 2 - BS[1][0] ** 2)
    e = 8 * (R[0] ** 2) * g * h + 8 * (R[0] ** 2) * l * k - temp

    temp = 0
    temp = (R[0] ** 2 - BS[1][0] ** 2) ** 2
    f = (2 * R[0] * h) ** 2 + (2 * R[0] * l) ** 2 - temp

    x1, x2 = solve_equations(d, e, f)
    if (x1 == x2 == 0):
        return [0, 0, 0]

    y1 = g * x1 + h
    y2 = g * x2 + h

    z1 = k * x1 + l
    z2 = k * x2 + l

    # print("两个x:",x1,x2)
    tempR2_1 = math.sqrt(((x1 - BS[1][0]) ** 2) + (y1 ** 2) + (z1 ** 2))
    tempR1_1 = math.sqrt((x1 ** 2) + (y1 ** 2) + (z1 ** 2))
    tempslotR_1 = abs((tempR2_1 - tempR1_1) - R[0])

    tempR2_2 = math.sqrt(((x2 - BS[1][0]) ** 2) + (y2 ** 2) + (z2 ** 2))
    tempR1_2 = math.sqrt((x2 ** 2) + (y2 ** 2) + (z2 ** 2))
    tempslotR_2 = abs((tempR2_2 - tempR1_2) - R[0])

    tempR3_1 = math.sqrt(((x1 - BS[2][0]) ** 2) + ((y1 - BS[2][1]) ** 2))
    tempslotR_3 = abs((tempR3_1 - tempR1_1) - R[1])

    tempR3_2 = math.sqrt(((x2 - BS[2][0]) ** 2) + ((y2 - BS[2][1]) ** 2))
    tempslotR_4 = abs((tempR3_2 - tempR1_2) - R[1])

    # print("两个slot",tempslotR_1,tempslotR_2)
    if (tempslotR_1 + tempslotR_3 < tempslotR_2 + tempslotR_4):
        # if(tempslotR_1<tempslotR_2):
        result = [x1, y1, z1]
        print("返回x1")
    else:
        result = [x2, y2, z2]
        print("返回x2")
    return result


def Chan(BSN, BS, R):
    Q = np.eye(BSN - 1)

    K = []  # 第一个坐标设为 0,0
    # K.append(0)

    for i in range(BSN):
        K.append(BS[i][0] ** 2 + BS[i][1] ** 2)
    K = np.array(K)

    Ga = []  # G矩阵
    for i in range(1, BSN):
        Ga.append([BS[i][0] - BS[0][0], BS[i][1] - BS[0][1], R[i - 1]])
        # R阵少一个
    Ga = np.array(Ga)
    # print("行列式为",det(Ga))

    Ga = -Ga

    R = np.array(R)
    h = 0.5 * (R ** 2 - K[1:BSN] + K[0])  # h矩阵

    # 第一次WLS的粗略估计结果(远距算法)
    Za0 = inv((Ga.T).dot(inv(Q)).dot(Ga)).dot((Ga.T).dot(inv(Q)).dot(h))
    # print("远场结果为",Za0)

    # 第一次WLS计算(近距算法)

    r = np.sqrt(((BS[1:BSN] - Za0[0:2]) ** 2).sum(1))
    # 两行一列
    B = np.diag(r)
    Fa = B.dot(Q).dot(B)
    Za1 = inv((Ga.T).dot(inv(Fa)).dot(Ga)).dot((Ga.T)).dot(inv(Fa)).dot(h)

    Za_cov = inv((Ga.T).dot(inv(Fa)).dot(Ga))

    # print("近场结果为",Za1)

    # 第二次WLS计算
    Ga1 = np.array([[1, 0], [0, 1], [1, 1]])
    h1 = np.array([(Za1[0] - BS[0][0]) ** 2, (Za1[1] - BS[0][1]) ** 2, Za1[2] ** 2])
    B1 = np.diag([Za1[0] - BS[0][0], Za1[1] - BS[0][1], Za1[2]])
    Fa1 = 4 * (B1).dot(Za_cov).dot(B1)
    Za2 = inv((Ga1.T).dot(inv(Fa1)).dot(Ga1)).dot((Ga1.T)).dot(inv(Fa1)).dot(h1)

    """
    pos1 = np.sqrt(Za2) + BS[0];
    pos2 = -np.sqrt(Za2) + BS[0];

    
    pos3 = [np.sqrt(Za2[0]), -np.sqrt(Za2[1])] + BS[0]
    pos4 = [-np.sqrt(Za2[0]), np.sqrt(Za2[1])] + BS[0]
    pos = [pos1, pos2]#, pos3, pos4]  输出形式
    """
    pos = []
    Za = np.sqrt(np.abs(Za2))
    if Za1[0] < 0:
        pos.append(-1 * Za[0])
    else:
        pos.append(Za[0])

    if Za1[1] < 0:
        pos.append(-1 * Za[1])

    else:
        pos.append(Za[1])

    print("TDOA结算结果为", pos)

    return pos
    # 结果还需要一次扭转


# if __name__ == '__main__':

#     # 测试用函数，测试时需把开头的全局变量改成常量

#     fig = plt.figure()
#     x1 = [1,2,4,5]
#     y1 = [-1,1,2,-1]
#     x2 = [3,6,7,4]
#     y2 = [5,5,4,3]

#     x,y,isSave = collisionDetection(x1,y1,x2,y2)

#     print(isSave)
#     x1.append(x1[0])
#     y1.append(y1[0])
#     x2.append(x2[0])
#     y2.append(y2[0])

#     x[0].append(x[0][0])
#     x[1].append(x[1][0])
#     y[0].append(y[0][0])
#     y[1].append(y[1][0])

#     plt.plot(x1,y1)
#     plt.plot(x2,y2)
#     plt.plot(x[0],y[0])
#     plt.plot(x[1],y[1])


#     plt.show()

def pending_nlos(rx_rssi, fp_rssi, dis):
    # 判定是否为nlos场景
    single_sample = {
        'rx_rssi': rx_rssi,
        'fp_rssi': fp_rssi,
        'range': dis
    }

    # 没有接收信号强度数据
    if math.fabs(rx_rssi + 655.35) < 0.01 or math.fabs(fp_rssi + 655.35) < 0.01:
        return 1

    y_pred = clf_loaded.predict(pd.DataFrame([single_sample]))
    return y_pred[0]


def handle_imu_data(data):
    temp_list = [0, 0, 0]

    if (data[25] << 8 | data[24]) & 0x8000:  # 检查符号位
        temp_list[0] = (data[25] << 8 | data[24]) - 0x10000  # 转换为负数
    else:
        temp_list[0] = (data[25] << 8 | data[24])  # 正数直接返回

    if (data[27] << 8 | data[26]) & 0x8000:  # 检查符号位
        temp_list[1] = (data[27] << 8 | data[26]) - 0x10000  # 转换为负数
    else:
        temp_list[1] = (data[27] << 8 | data[26])  # 正数直接返回

    if (data[29] << 8 | data[28]) & 0x8000:  # 检查符号位
        temp_list[2] = (data[29] << 8 | data[28]) - 0x10000  # 转换为负数
    else:
        temp_list[2] = (data[29] << 8 | data[28])  # 正数直接返回

    imu_ax = temp_list[0] / 32768 * 16 * 9.8
    imu_ay = temp_list[1] / 32768 * 16 * 9.8
    imu_az = temp_list[2] / 32768 * 16 * 9.8
    print("imu data: ", imu_ax, imu_ay, imu_az)
    return temp_list
