# -*- coding: utf-8 -*-
from utils import *
from config import *
import src


def decode(data):
    global warnFlag

    # 解析数据
    # print(data)
    if len(data) != 16:
        print('error len data: ', len(data))
        return

    dis_list = []
    tag_id = data[3]

    if tag_id >= TX_NUM + TX_STR_NUM:
        return

    dis0 = (data[6] + data[7] * 256) / 100
    dis1 = (data[8] + data[9] * 256) / 100
    dis2 = (data[10] + data[11] * 256) / 100
    dis3 = 0

    dis_list = [dis0, dis1, dis2]

    print(tag_id, dis_list)

    if RX_NUM == 4:
        dis3 = (data[12] + data[13] * 256) / 100
        dis_list.append(dis3)

    tx_location = cul_tx_location(tag_id, dis_list)

    if tx_location != 0:

        # var = cul_static_variance(tag_id)
        # print("当前标签：", tag_id,"    坐标：", tx_location)

        if SAVE_DATA_FLAG:
            src.save_data(tag_id, tx_location)


def cul_tx_location(tag_id, dis_list):
    # 二维场景下计算标签坐标(x, y)
    # 默认以rx1为原点，rx1到rx2的向量为x轴建立坐标系

    global obsX, obsY, re_x, re_y, ex_x, ex_y

    if len(dis_list) == 3:

        [x0, y0] = two_point_location(dis_list[0], dis_list[1], dis_list[2], rx1, rx2, rx3)
        [x1, y1] = two_point_location(dis_list[1], dis_list[2], dis_list[0], rx2, rx3, rx1)
        [x2, y2] = two_point_location(dis_list[2], dis_list[0], dis_list[1], rx3, rx1, rx2)

        # 取三角形中心作为定位坐标
        x = (x0 + x1 + x2) / 3
        y = (y0 + y1 + y2) / 3

        x = round(x, 6)
        y = round(y, 6)

        src.detection(tag_id, x, y)
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

        src.detection(tag_id, x, y)
        return [x, y]
