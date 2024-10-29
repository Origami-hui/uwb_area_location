import socket
import threading
from config import *
import src

# 程序主函数

if __name__ == '__main__':

    if SAVE_DATA_FLAG:

        if TDOA_FLAG:
            for i in range(RX_NUM):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.bind(("192.168.1.42", 8081 + i))
                print("等待主基站接入")

                thd = threading.Thread(target=src.dispose_client_request, args=(s,))
                thd.start()
        else:
            # 创建服务端套接字对象
            tcp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 端口复用
            # tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
            # 绑定端口
            tcp_server.bind((socket.gethostbyname(socket.gethostname()), 8080))
            # 设置监听
            # tcp_server.listen(128)
            print('正在等待主基站连接.....\n')
            # 建立TCP连接
            #tcp_client, tcp_client_address = tcp_server.accept()
            #print('主基站接入:', tcp_client_address)
            #socks = tcp_client

            thd = threading.Thread(target=src.dispose_client_request, args=(tcp_server, ))
            thd.setDaemon(True)
            thd.start()

    else:

        thd = threading.Thread(target=src.openData, args=())
        thd.setDaemon(True)
        thd.start()

    if WARN_MOD_FLAG:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = socket.gethostbyname(socket.gethostname())  # 获取本地ip

        # 尝试连接预警模块
        port = 8084
        # print('服务端IP:%s端口:%d'% (host,port))
        s.bind((host,port))
        s.listen(5)
        print('正在等待预警模块连接.....\n')
        client,client_address = s.accept()
        print('预警模块接入',client_address)
        warn = threading.Thread(target=src.warnSystem, args=(client,))
        warn.start()

    src.visualization()

   
    
