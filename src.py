# -*- coding: utf-8 -*-
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import xlwt as xlwt
from PIL import Image

from EKF import *
from twr_interface import *

# 储存全局变量

if RX_NUM == 3:

    min_x = min(rx1[0], rx2[0], rx3[0])
    min_y = min(rx1[1], rx2[1], rx3[1])
    max_x = max(rx1[0], rx2[0], rx3[0])
    max_y = max(rx1[1], rx2[1], rx3[1])

elif RX_NUM == 4:
    min_x = min(rx1[0], rx2[0], rx3[0], rx4[0])
    min_y = min(rx1[1], rx2[1], rx3[1], rx4[1])
    max_x = max(rx1[0], rx2[0], rx3[0], rx4[0])
    max_y = max(rx1[1], rx2[1], rx3[1], rx4[1])

# 当前观测坐标
obsX, obsY = [[] for _ in range(TX_NUM)], [[] for _ in range(TX_NUM)]
obsV = [[] for _ in range(TX_NUM)]
obsTheta = [[] for _ in range(4)]
obsOmiga = [[] for _ in range(4)]

# 上一车辆区域中心坐标
pre_cx = 0
pre_cy = 0
unitized_dx = 1
unitized_dy = 1

fig = plt.figure()
ax = plt.gca()
ax.set_aspect(1)

wb = xlwt.Workbook()

ws = wb.add_sheet('location_data')  # 添加一个表
ri = [0 for _ in range(TX_NUM)]

re_x, re_y = [], []  # 构建多边形坐标值
ex_x, ex_y = [], []  # 扩展多边形坐标值
org_x, org_y = [], []  # 修正前矩形，用于展示

warnFlag = True

nlosFlag = False

# for i in range(TX_NUM):
#     ws.write(ri[i], 0 + 3*i, "rx"+str(i+1)+"_x") # 写入数据，3个参数分别为行号，列号，和内容
#     ws.write(ri[i], 1 + 3*i, "rx"+str(i+1)+"_y")

# ws.write(ri[])

# def read_serial(port, baudrate):
#     with serial.Serial(port, baudrate) as ser:

#         while True:
#             if ser.in_waiting > 0:
#                 data = ser.readline()
#                 data = bytes(data)
#                 decode(data=data)

STR_FLAG = False

matrix_list = [[[], []] for _ in range(4)]
S_list = []  # 储存统计的抖动方差
towards_list = []  # 车身转向列表

avg_dis_diff = []


def detection(tag_id, x, y):
    global unitized_dx, unitized_dy, ex_x, ex_y, obsX, obsY, re_x, re_y, warnFlag, \
        pre_cx, pre_cy, STR_FLAG, matrix_list, S_list, towards_list, org_x, org_y, avg_dis_diff

    if len(obsX[tag_id - TX_STR_NUM]) > 1:
        obsV[tag_id - TX_STR_NUM].append(math.sqrt((obsX[tag_id - TX_STR_NUM][-1] - obsX[tag_id - TX_STR_NUM][-2]) ** 2
                                                   + (obsY[tag_id - TX_STR_NUM][-1] - obsY[tag_id - TX_STR_NUM][
            -2]) ** 2) / LOCATION_FREQ)
    else:
        obsV[tag_id - TX_STR_NUM].append(0.0)
    if tag_id < TX_STR_NUM + TX_NUM - HUM_NUM:
        if len(towards_list) > 0:
            obsTheta[tag_id - TX_STR_NUM].append(towards_list[-1])
            if len(obsTheta) > 1:
                obsOmiga[tag_id - TX_STR_NUM].append(
                    obsTheta[tag_id - TX_STR_NUM][-1] - obsTheta[tag_id - TX_STR_NUM][-2])
            else:
                obsOmiga[tag_id - TX_STR_NUM].append(0)
        else:
            obsTheta[tag_id - TX_STR_NUM].append(0)
            obsOmiga[tag_id - TX_STR_NUM].append(0)

    if x > max_x * 1.5 or x < min_x - 5 or y > max_y * 1.5 or y < min_y - 5:
        print("越界异常坐标点：", x, y)
        return

    if FILTER_FLAG and len(obsX[tag_id - TX_STR_NUM]) >= 1 and \
            isAbnormalPoints(x, y, obsX[tag_id - TX_STR_NUM], obsY[tag_id - TX_STR_NUM]):
        print("过滤异常坐标点：", x, y)
        # 用卡尔曼滤波预测并重新赋值
        if HAVE_HUM and tag_id > TX_STR_NUM + TX_NUM - HUM_NUM:
            # 人员标签
            x, y = kalman_filter(obsX[tag_id - TX_STR_NUM], obsY[tag_id - TX_STR_NUM])
        print("重新赋值为：", x, y)
        return

    if FILTER_FLAG and len(obsX[tag_id - TX_STR_NUM]) >= 1 and tag_id <= TX_STR_NUM + TX_NUM - HUM_NUM:
        if len(obsTheta[tag_id - TX_STR_NUM]) > 0 and len(obsOmiga[tag_id - TX_STR_NUM]) > 0 \
                and len(obsV[tag_id - TX_STR_NUM]) > 0:
            # 对于车辆标签，执行扩展卡尔曼滤波，返回预测
            tx, ty = ekf(tag_id, x, y, obsV[tag_id - TX_STR_NUM][-1], obsTheta[tag_id - TX_STR_NUM][-1],
                         obsOmiga[tag_id - TX_STR_NUM][-1])
            # tx, ty = kalman_filter(obsX[tag_id - TX_STR_NUM], obsY[tag_id - TX_STR_NUM])
            dis = math.sqrt((tx - x) ** 2 + (ty - y) ** 2)

            avg_dis_diff.append(dis)
            print("平均预测偏差：", sum(avg_dis_diff) / len(avg_dis_diff))

            if dis > 6:
                print("重新赋值")
                x = (x + tx) / 2
                y = (y + ty) / 2

            resetState(tag_id, x, y, obsV[tag_id - TX_STR_NUM][-1], obsTheta[tag_id - TX_STR_NUM][-1],
                       obsOmiga[tag_id - TX_STR_NUM][-1])

    obsX[tag_id - TX_STR_NUM].append(x)
    obsY[tag_id - TX_STR_NUM].append(y)

    if len(obsY[tag_id - TX_STR_NUM]) > 64:
        obsX[tag_id - TX_STR_NUM].pop(0)
        obsY[tag_id - TX_STR_NUM].pop(0)
        if len(obsV[tag_id - TX_STR_NUM]) > 0:
            obsV[tag_id - TX_STR_NUM].pop(0)
        if tag_id <= TX_STR_NUM + TX_NUM - HUM_NUM and len(obsTheta[tag_id - TX_STR_NUM]) > 0 \
                and len(obsOmiga[tag_id - TX_STR_NUM]) > 0:
            obsTheta[tag_id - TX_STR_NUM].pop(0)
            obsOmiga[tag_id - TX_STR_NUM].pop(0)

    # if FILTER_FLAG and len(obsY[tag_id-TX_STR_NUM])>= 8:
    #     print("x：", obsX[tag_id-TX_STR_NUM])
    #     obsX[tag_id-TX_STR_NUM], obsY[tag_id-TX_STR_NUM] = smooth(obsX[tag_id-TX_STR_NUM], obsY[tag_id-TX_STR_NUM])
    #     print("newx：", obsX[tag_id-TX_STR_NUM])

    have_data = True

    for i in obsX:
        if len(i) == 0:
            have_data = False

    if BUILD_RECT_FLAG and ((HAVE_HUM and TX_NUM >= 5) or (HAVE_HUM == False and TX_NUM >= 4)) and have_data:
        # 若启动构建矩形标志为真，且当前标签数量不少于4，且每个标签都至少有一个坐标数据

        re_x_temp, re_y_temp = build_rectangle(obsX, obsY)

        org_x = re_x_temp.copy()
        org_y = re_y_temp.copy()

        polygon = toPolygon(re_x_temp, re_y_temp)

        re_x_temp, re_y_temp = coordinateCorrection(polygon)

        cur_cx = sum(p[0] for p in polygon) / 4
        cur_cy = sum(p[1] for p in polygon) / 4

        # 进行车辆转角修正
        shakeList = [cul_static_variance(obsX[i], obsY[i]) for i in range(4)]
        print("抖动方差：", sum(shakeList) / len(shakeList))
        totalShake = np.min(shakeList)  # 车辆标签位置抖动值

        if cur_cx - pre_cx != 0 or cur_cy - pre_cy != 0:
            STR_FLAG = True

        if pre_cx != 0 and STR_FLAG and TOWARDS_COR_FLAG:
            # 如果存在偏移，且上一车辆中心区域不为空，则进行转向矫正，否则不矫正
            re_x, re_y, unitized_dx, unitized_dy = towardsVDir(totalShake, re_x_temp, re_y_temp, cur_cx - pre_cx,
                                                               cur_cy - pre_cy)
            ex_x, ex_y = extendArea(re_x, re_y)
        else:
            re_x = re_x_temp
            re_y = re_y_temp
            dx = re_x[1] - re_x[2]
            dy = re_y[1] - re_y[2]
            unitized_dx = dx / (dx ** 2 + dy ** 2) ** 0.5
            unitized_dy = dy / (dx ** 2 + dy ** 2) ** 0.5

        pre_cx = cur_cx
        pre_cy = cur_cy

        # 测试抖动方差代码段

        # for i in range(4):
        #     Martex_list[i][0].append(re_x[i])
        #     Martex_list[i][1].append(re_y[i])

        #     if(len(Martex_list[i][0]) > 20):
        #         Martex_list[i][0].pop(0)
        #         Martex_list[i][1].pop(0)

        # for i in range(4):
        #     S_list.append(cul_static_variance(Martex_list[i][0], Martex_list[i][1]))
        #     print("抖动方差：", sum(S_list) / len(S_list))

        # 测试转向抖动代码段

        # towards_list[0].append(unitized_dx)
        # towards_list[1].append(unitized_dy)

        towards_list.append(math.atan(unitized_dy / unitized_dx))

        # print("车身转向抖动标准差：", cul_towards_variance(towards_list))

        # 如果有人员标签，且当前解算坐标的标签就是人员标签，进行碰撞检测
        if HAVE_HUM and tag_id > TX_STR_NUM + TX_NUM - HUM_NUM:
            ex_x, ex_y, is_save = collisionDetection(re_x, re_y, x, y)
            # print("当前安全情况：", isSave)
            warnFlag = is_save
            return is_save


def merge_location(tag_id, tx_location, imu_acc, nlos_num):
    if len(obsX[tag_id - TX_STR_NUM]) > 1:
        # print(obsX[tag_id - TX_STR_NUM][-1], obsY[tag_id - TX_STR_NUM][-1])
        # acc = math.sqrt(imu_acc[0] ** 2 + imu_acc[1] ** 2)
        speed_temp = [(obsX[tag_id - TX_STR_NUM][-1] - obsX[tag_id - TX_STR_NUM][-2]) * LOCATION_FREQ,
                      (obsY[tag_id - TX_STR_NUM][-1] - obsY[tag_id - TX_STR_NUM][-2]) * LOCATION_FREQ]
        dir_temp = [speed_temp[0] / math.sqrt(speed_temp[0] ** 2 + speed_temp[1] ** 2),
                    speed_temp[1] / math.sqrt(speed_temp[0] ** 2 + speed_temp[1] ** 2)]
        # (x, y) = (c⋅a+d⋅b, −c⋅b+d⋅a)
        acc_spec = [imu_acc[0] * dir_temp[1] + imu_acc[1] * dir_temp[0],
                    -imu_acc[0] * dir_temp[0] + imu_acc[1] * dir_temp[1]]
        v_temp = [
            obsX[tag_id - TX_STR_NUM][-1] + 0.5 * acc_spec[0] / (LOCATION_FREQ ** 2),
            obsY[tag_id - TX_STR_NUM][-1] + 0.5 * acc_spec[1] / (LOCATION_FREQ ** 2)]
        # 根据处于nlos的信号数量决定最终坐标点到虚拟点和定位点的权重
        weight = cul_weight(nlos_num)
        return [v_temp[0] + weight * (tx_location[0] - v_temp[0]), v_temp[1] + weight * (tx_location[1] - v_temp[1])]

    elif len(obsX[tag_id - TX_STR_NUM]) == 1:
        # 如果缓存池只有一个数据，直接用这个数据返回
        return [obsX[tag_id - TX_STR_NUM][0], obsY[tag_id - TX_STR_NUM][0]]

    return tx_location


def cul_weight(nlos_num):
    # 处于nlos的信号越多，最终坐标越往虚拟坐标点靠近（全nlos直接取虚拟坐标点）
    if nlos_num == 3:
        return 0
    elif nlos_num == 2:
        return 0.25
    else:
        return 0.5


gtx = [1.45, 1.45, 10.8, 10.8, 1.45]
gty = [9.45, 0.6, 0.6, 9.45, 9.45]


def plot_init():
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    sc = plt.scatter([rx1[0], rx2[0], rx3[0]], [rx1[1], rx2[1], rx3[1]], c="k")

    # 绘制ground true矩形
    rect, = plt.plot(gtx, gty, "k")
    return sc, rect,


def plot_update(i):
    global unitized_dx, unitized_dy, obsX, obsY, \
        re_x, re_y, ex_x, ex_y, org_x, org_y

    # 当前更新绘图方法，后续可进行轨迹优化

    ax.clear()

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    X_copy = obsX.copy()
    Y_copy = obsY.copy()

    c_list = ["r", "g", "m", "c", "y"]

    tup = tuple()

    if CAR_TX_RENDER_FLAG:
        sc = [_ for _ in range(TX_NUM)]
        for k in range(TX_NUM):
            if nlosFlag:
                sc[k] = plt.scatter(X_copy[k][-1:], Y_copy[k][-1:], marker='+', c='g')
            else:
                sc[k] = plt.scatter(X_copy[k][-1:], Y_copy[k][-1:], marker='+', c=c_list[k])
        tup = tuple(each for each in sc)
    elif CAR_TX_RENDER_FLAG is False and HAVE_HUM:
        sc = [[]]
        if len(X_copy) > 4:
            sc[0] = plt.scatter(X_copy[4][-1:], Y_copy[4][-1:], c=c_list[4])

        tup = tuple(each for each in sc)

    if RX_NUM == 3:
        ini = plt.scatter([rx1[0], rx2[0], rx3[0]], [rx1[1], rx2[1], rx3[1]], c="k")
    else:
        ini = plt.scatter([rx1[0], rx2[0], rx3[0], rx4[0]],
                          [rx1[1], rx2[1], rx3[1], rx4[1]], c="k")
    tup += (ini,)
    # print(re_x)
    if len(re_x) > 0:  # 矩形坐标有数据，绘制矩形

        reXCopy = re_x.copy()
        reYCopy = re_y.copy()

        orgXCopy = org_x.copy()
        orgYCopy = org_y.copy()

        # 绘制指向车前方的箭头
        avg_x = sum(reXCopy) / len(reXCopy)
        avg_y = sum(reYCopy) / len(reYCopy)
        # dx = (reXCopy[0] + reXCopy[1])/2 - avg_x
        # dy = (reYCopy[0] + reYCopy[1])/2 - avg_y

        arr = plt.arrow(avg_x, avg_y, unitized_dx * (CAR_LENGTH / 2), unitized_dy * (CAR_LENGTH / 2), head_width=0.3,
                        length_includes_head=True, fc='b', ec='r')

        reXCopy.append(reXCopy[0])
        reYCopy.append(reYCopy[0])
        rect, = plt.plot(reXCopy, reYCopy, "r")

        # 原始定位区域多边形
        # orgXCopy.append(orgXCopy[0])
        # orgYCopy.append(orgYCopy[0])
        # orgRect, = plt.plot(orgXCopy,orgYCopy,"r", alpha=0.5)

        # if TX_NUM >= 4 and BUILD_RECT_FLAG:
        #     # 绘制安全多边形
        #     # ex_x, ex_y = buildAnddetection.extendArea(re_x, re_y)
        #     if len(ex_x) > 0:
        #         ex_x.append(ex_x[0])
        #         ex_y.append(ex_y[0])
        #         ex_rect, = plt.plot(ex_x, ex_y, "b")
        #         fig.canvas.flush_events()
        #         return tup + (rect, ex_rect, arr)

        fig.canvas.flush_events()
        return tup + (rect, arr)

    # 重新渲染子图
    fig.canvas.flush_events()
    return tup


def save_data(tag_id, tx_location, dis_list, fp_rssi, rx_rssi, acc_data):
    global ri
    ri[tag_id - TX_STR_NUM] += 1
    ws.write(ri[tag_id - TX_STR_NUM], 0 + 3 * (tag_id - TX_STR_NUM), tx_location[0])
    ws.write(ri[tag_id - TX_STR_NUM], 1 + 3 * (tag_id - TX_STR_NUM), tx_location[1])

    if dis_list is not None:
        ws.write(ri[tag_id - TX_STR_NUM], 3 + 3 * (tag_id - TX_STR_NUM), dis_list[0])
        ws.write(ri[tag_id - TX_STR_NUM], 4 + 3 * (tag_id - TX_STR_NUM), dis_list[1])
        ws.write(ri[tag_id - TX_STR_NUM], 5 + 3 * (tag_id - TX_STR_NUM), dis_list[2])

    if fp_rssi is not None:
        ws.write(ri[tag_id - TX_STR_NUM], 7 + 3 * (tag_id - TX_STR_NUM), fp_rssi[0])
        ws.write(ri[tag_id - TX_STR_NUM], 8 + 3 * (tag_id - TX_STR_NUM), fp_rssi[1])
        ws.write(ri[tag_id - TX_STR_NUM], 9 + 3 * (tag_id - TX_STR_NUM), fp_rssi[2])

    if rx_rssi is not None:
        ws.write(ri[tag_id - TX_STR_NUM], 11 + 3 * (tag_id - TX_STR_NUM), rx_rssi[0])
        ws.write(ri[tag_id - TX_STR_NUM], 12 + 3 * (tag_id - TX_STR_NUM), rx_rssi[1])
        ws.write(ri[tag_id - TX_STR_NUM], 13 + 3 * (tag_id - TX_STR_NUM), rx_rssi[2])

    if acc_data is not None:
        ws.write(ri[tag_id - TX_STR_NUM], 15 + 3 * (tag_id - TX_STR_NUM), acc_data[0])
        ws.write(ri[tag_id - TX_STR_NUM], 16 + 3 * (tag_id - TX_STR_NUM), acc_data[1])
        ws.write(ri[tag_id - TX_STR_NUM], 17 + 3 * (tag_id - TX_STR_NUM), acc_data[2])

    wb.save(W_DATA_FILE_NAME)


def warnSystem(client):
    # 调用蜂鸣器
    global warnFlag
    while 1:
        beep = 't'
        if warnFlag:
            beep = 't'
        else:
            beep = 'f'
        client.send(beep.encode())
        time.sleep(0.1)


def dispose_client_request(client, tcp_client_address=0):
    if TDOA_FLAG:

        while True:

            data, addr = client.recvfrom(1048)
            # print(data,addr)
            # data=data.decode("utf-8")
            thd = threading.Thread(target=dataProcess, args=(data,))
            thd.setDaemon(True)
            thd.start()

            if data == b"exit":
                client.sendto(b"Good bye!\n")
                break

    else:

        # 循环接收和发送消息
        while True:

            recv_data = client.recv(128)
            # 有消息就回复数据，如果消息长度为0表示客户端下线了
            if recv_data:

                recv_data = bytes(recv_data)

                if len(recv_data) % 16 == 0:
                    # print((int)(len(recv_data)/16))
                    for k in range(int(len(recv_data) / 16)):
                        decode(data=recv_data[k * 16: 16 + k * 16])

            else:
                print("%s 客户端下线了.")
                client.close()
                break


def handle_uart_data():
    ser = serial.Serial(RX_COM, 115200, parity=serial.PARITY_NONE, stopbits=1, bytesize=8)

    while True:
        data = ser.read(32)
        data = bytes(data)
        decode_with_extension_data(data)


def visualization():
    # 可视化
    ani = animation.FuncAnimation(fig,  # 画布
                                  plot_update,  # 图像更新
                                  init_func=plot_init,  # 图像初始化
                                  frames=30,
                                  interval=5,  # 图像更新间隔
                                  blit=False)

    # img = Image.open('C:/Users/17005/Desktop/map.png')
    # img = np.array(img)
    # ax.imshow(img, extent=[-0.5, 16, -0.5, 19.3])  # extent 设置坐标范围

    plt.scatter(1000, 1000, marker='+', c='g', label='Coordinates after ')
    plt.scatter(1000, 1000, marker='+', c='r', label='Coordinates before ')
    plt.scatter(1000, 1000, color='gray', label='Coordinates before ')
    plt.plot(1000, 1000, c='k', label='Ground True')

    plt.title("target location")
    plt.legend(loc='upper left')
    plt.show()


def openData():
    # if (FILTER_FLAG):
    #     smooth(R_DATA_FILE_NAME, "data/new_data.xls")
    #     df = pd.read_excel("data/new_data.xls")
    # else:
    df = pd.read_excel(R_DATA_FILE_NAME)
    data = df.values
    # print("获取到所有的值:\n{}".format(data))

    testNum = 0
    warnNum = 0
    vaildIndexStart = 2  # 计算概率的有效行数起始索引
    # vaildIndexEnd = 539 # 计算概率的有效行数结束索引
    # print(len(data), TX_NUM)

    for k in range(len(data)):
        for j in range(TX_NUM):
            # print(data[k][3 * j])
            if not math.isnan(data[k][3 * j]):
                # print(data[i][3*j],data[i][3*j+1])
                isSafe = detection(j + 5, data[k][3 * j], data[k][3 * j + 1])

                if k >= vaildIndexStart and j == 4 and HAVE_HUM:
                    testNum += 1
                    if not isSafe:
                        warnNum += 1
                    print("当前概率：", calculateRate(warnNum, testNum))

        time.sleep(0.2)  # 控制读取数据的速度


def openDataV2():
    global nlosFlag

    df = pd.read_excel(R_DATA_FILE_NAME)
    data = df.values

    change_index = [29, 46, 59, 95, 111, 122, 141, 155, 165, 182]
    now_index = 0
    RMSE = 0
    STD = 0
    sum_temp = [0 for _ in range(len(change_index))]
    avg_temp = [42.050000000000004, 10.199999999999998, 140.4, 340.19999999999976, 23.199999999999992, 6.599999999999999, 205.20000000000007, 132.3, 14.499999999999996, 10.199999999999998, 151.20000000000002]
    cdf = []

    for k in range(1, change_index[-1]):
        if not math.isnan(data[k][0]):

            # 正常执行定位流程即可
            tx_location = cul_tx_location(5, [data[k][3], data[k][4], data[k][5]])

            acc_data = [data[k][15], data[k][16], data[k][17]]

            nlos0 = pending_nlos(data[k][11], data[k][7], data[k][3])
            nlos1 = pending_nlos(data[k][12], data[k][8], data[k][4])
            nlos2 = pending_nlos(data[k][13], data[k][9], data[k][5])
            print(nlos0, nlos1, nlos2)

            if nlos0 + nlos1 + nlos2 > 1 and NLOS_FIX_FLAG:
                # 引入imu测量加速度
                print("补偿前：", tx_location)
                tx_location = merge_location(5, tx_location, acc_data, nlos0 + nlos1 + nlos2)
                print("补偿后：", tx_location)
                nlosFlag = True
            else:
                nlosFlag = False

            detection(5, tx_location[0], tx_location[1])

            # 计算RMSE 与 std
            if k > change_index[now_index]:
                now_index += 1

            n_temp = change_index[now_index]
            if now_index > 1:
                n_temp = change_index[now_index] - change_index[now_index - 1]

            if now_index % 2 == 0:
                RMSE += (tx_location[0] - gtx[now_index % 4]) ** 2
                STD += (avg_temp[now_index] / n_temp - gtx[now_index % 4]) ** 2
                if now_index > 1:
                    cdf.append(math.fabs(tx_location[0] - gtx[now_index % 4]))
                sum_temp[now_index] += gtx[now_index % 4]
            else:
                RMSE += (tx_location[1] - gty[now_index % 4]) ** 2
                STD += (avg_temp[now_index] / n_temp - gty[now_index % 4]) ** 2
                if now_index > 1:
                    cdf.append(math.fabs(tx_location[1] - gty[now_index % 4]))
                sum_temp[now_index] += gty[now_index % 4]

            print("RMSE: ", math.sqrt(RMSE / k))
            print("STD: ", math.sqrt(STD / k))
            # print("sum_temp: ", sum_temp)

        time.sleep(0.05)  # 控制读取数据的速度

    print("cdf: ", cdf)

def calculateRate(warnNum, testNum):
    # 计算虚警概率与预警成功率，默认以数据集测试。

    # 虚警率：人与车的距离在2~3米的时候（此时客观上为安全状态）计算一定时间内的安全与预警的结果数量之比
    # 预警成功率：人进入了车的安全范围（此时客观上为预警状态）计算一定时间内的安全与预警的结果数量之比

    # 返回当前计算的虚警概率与预警成功率
    return warnNum / testNum
