# -*- coding: utf-8 -*-
import src
from utils import *
from config import *
from src import *


def decode(data):
    # 解析数据
    if len(data) != 16:
        print('error len data: ', len(data))
        return
    tag_id = data[3]

    if tag_id >= Config.TX_NUM + Config.TX_STR_NUM:
        return

    dis0 = (data[6] + data[7] * 256) / 100
    dis1 = (data[8] + data[9] * 256) / 100
    dis2 = (data[10] + data[11] * 256) / 100
    dis3 = 0

    dis_list = [dis0, dis1, dis2]

    print(tag_id, dis_list)

    if Config.RX_NUM == 4:
        dis3 = (data[12] + data[13] * 256) / 100
        dis_list.append(dis3)

    tx_location = cul_tx_location(tag_id, dis_list)
    src.detection(tag_id, tx_location[0], tx_location[1])

    if tx_location != 0:

        # var = cul_static_variance(tag_id)
        # print("当前标签：", tag_id,"    坐标：", tx_location)

        if Config.SAVE_DATA_FLAG:
            src.save_data(tag_id, tx_location)


def decode_with_extension_data(data):
    if len(data) < 32:
        return

    tag_id = data[3]
    # seq_num = data[4] + data[5] * 256

    dis0 = (data[6] + data[7] * 256) / 100
    dis1 = (data[8] + data[9] * 256) / 100
    dis2 = (data[10] + data[11] * 256) / 100

    dis_list = [dis0, dis1, dis2]

    fp_rssi0 = -(data[12] + data[13] * 256) / 100
    fp_rssi1 = -(data[14] + data[15] * 256) / 100
    fp_rssi2 = -(data[16] + data[17] * 256) / 100

    rx_rssi0 = -(data[18] + data[19] * 256) / 100
    rx_rssi1 = -(data[20] + data[21] * 256) / 100
    rx_rssi2 = -(data[22] + data[23] * 256) / 100

    nlos0 = pending_nlos(rx_rssi0, fp_rssi0, dis0)
    nlos1 = pending_nlos(rx_rssi1, fp_rssi1, dis1)
    nlos2 = pending_nlos(rx_rssi2, fp_rssi2, dis2)

    # 正常执行定位流程即可
    tx_location = cul_tx_location(tag_id, dis_list)

    acc_data = handle_imu_data(data)

    if nlos0 + nlos1 + nlos2 > 0:
        # 引入imu测量加速度
        tx_location = src.merge_location(tag_id, tx_location, acc_data, nlos0 + nlos1 + nlos2)

    src.detection(tag_id, tx_location[0], tx_location[1])

    json_data = {
        "tx_location": tx_location,
        "nlos_state": [0, 0, 0],
        "tag_id": tag_id,
        "area_location": [src.re_x, src.re_y]
    }
    print(json.dumps(json_data))

    if tx_location != 0 and Config.SAVE_DATA_FLAG:
        src.save_data(tag_id, tx_location, dis_list, [fp_rssi0, fp_rssi1, fp_rssi2],
                      [rx_rssi0, rx_rssi1, rx_rssi2], acc_data)

    if Config.INSERT_DATABASE_FLAG:
        insert(5, [float(tx_location[0]), float(tx_location[1])], 0)

