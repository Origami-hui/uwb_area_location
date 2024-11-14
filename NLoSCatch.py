import pandas as pd
import os

import serial
import serial.tools.list_ports
import joblib
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
            # 检查 CSV 文件是否存在
            if os.path.exists(NLOS_DATA_NAME):
                # 读取 CSV 文件
                self.df = pd.read_csv(NLOS_DATA_NAME)
            else:
                # 创建一个空的 DataFrame
                self.df = pd.DataFrame(columns=['tx', 'rx_rssi', 'fp_rssi', 'range', 'nlos'])

        except FileNotFoundError:
            self.wb = openpyxl.Workbook(NLOS_DATA_NAME)
            self.ws = self.wb.create_sheet('rssi_and_dis_data')

    def print(self):
        print(self.tagAddr, ':', self.rx_rssi, ':', self.fp_rssi, ':', self.dis)

    def save_data(self):
        new_data = {'tx': self.tagAddr, 'rx_rssi': self.rx_rssi, 'fp_rssi': self.fp_rssi,
                    'range': self.dis, 'nlos': 1 if IN_NLOS_FLAG else 0}
        self.df = self.df._append(new_data, ignore_index=True)

        # 保存 DataFrame 到 CSV 文件
        self.df.to_csv(NLOS_DATA_NAME, index=False)

    def pending_nlos(self):
        single_sample = {
            'rx_rssi': self.rx_rssi,
            'fp_rssi': self.fp_rssi,
            'range': self.dis
        }

        X = pd.DataFrame([single_sample])

        clf_loaded = joblib.load(NLOS_MODEL_NAME)
        y_pred = clf_loaded.predict(X)
        print(y_pred)


def find_serial_port(vendor_id=None, product_id=None):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # 检查设备的 vendor_id 和 product_id（可选）
        if (vendor_id is None and product_id is None) or \
           (port.vid == vendor_id and port.pid == product_id):
            return port.device  # 返回找到的串口号
    return None  # 没有找到符合条件的串口


# 打开串口
ser = serial.Serial('COM6', 115200, parity=serial.PARITY_NONE, stopbits=1, bytesize=8)

try:
    while True:
        # 读取串口数据
        data = ser.readline()
        result = data.decode().strip().split(':')

        rssiAndDisData = RSSIAndDisData(result)

        # 处理数据
        rssiAndDisData.print()
        rssiAndDisData.save_data()
        #rssiAndDisData.pending_nlos()

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
