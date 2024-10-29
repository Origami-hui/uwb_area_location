import socket
import threading

import serial
import openpyxl
from config import *


# 采集数据类，包含当次数据的测距与信号强度信息
class RSSIAndDisData:

    def __init__(self, recv_data):
        # 15个槽位
        self.tagAddr = recv_data[0]
        self.rx_rssi = recv_data[1]  # 接收信号强度
        self.fp_rssi = recv_data[2]  # 第一路信号强度
        self.dis = recv_data[3]

        try:
            self.wb = openpyxl.load_workbook(NLOS_DATA_NAME)
            self.ws = self.wb.active
        except FileNotFoundError:
            self.wb = openpyxl.Workbook(NLOS_DATA_NAME)
            self.ws = self.wb.create_sheet('rssi_and_dis_data')

    def print(self):
        print(self.tagAddr, ':', self.rx_rssi, ':', self.fp_rssi, ':', self.dis)

    def save_data(self):
        self.ws.append([self.tagAddr, self.rx_rssi, self.fp_rssi, self.dis])
        self.wb.save(NLOS_DATA_NAME)


# 打开串口
ser = serial.Serial('COM10', 115200, parity=serial.PARITY_NONE, stopbits=1, bytesize=8)

try:
    while True:
        # 读取串口数据
        data = ser.readline()
        result = data.decode().strip().split(':')

        rssiAndDisData = RSSIAndDisData(result)

        # 处理数据
        rssiAndDisData.print()
        rssiAndDisData.save_data()

except KeyboardInterrupt:
    # 捕获Ctrl+C中断信号，关闭串口
    ser.close()

# def handle_recv_data(this_data):
#
#     result = this_data.decode().strip().split(':')
#     rssiAndDisData = RSSIAndDisData(result)
#
#     # 处理数据
#     rssiAndDisData.print()
#     print(round(abs(float(rssiAndDisData.rx_rssi) - float(rssiAndDisData.fp_rssi)), 2))
#     # rssiAndDisData.save_data()
#
#
# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# print(socket.gethostbyname(socket.gethostname()))
# s.bind((socket.gethostbyname(socket.gethostname()), 8080))
# print("等待主基站接入")
#
# while True:
#
#     data, addr = s.recvfrom(1048)
#
#     thd = threading.Thread(target=handle_recv_data, args=(data,))
#     thd.setDaemon(True)
#     thd.start()
