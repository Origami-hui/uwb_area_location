# -*- coding: utf-8 -*-
import time

import utils
import src
from config import *

# 预设变量
tx_dict = {}
rx_msr = {8193: 8193, 2: 8193, 3: 8193, 4: 8193}
group_msr = {8193: [[2, 3, 4]]}
RX_position = {8193: rx1, 2: rx2, 3: rx3, 4: rx4}

# #定义抖动分析仪的全局变量
# analysis={}

#时间栈
class timer:
                         #栈
    def __init__(self,slot=15):
                    #15个槽位
        self.slot=slot
        self.tank=[0]*self.slot
        self.index=-1
        self.num=0              #栈长
        self.flag=False         #可以被用去定位    True 已经被用去定位
        self.main={}            #进入的主机站地址：计数

    def clear(self):
        self.num=0
        self.tank=[0]*self.slot
        self.flag=False

    def assign(self,index):
        self.index=index

    def add(self,time_stamp,rx_node_addr,main_addr):

        if(self.num==self.slot):
            print("本栈已满")
            pass
        entity=[rx_node_addr,time_stamp,main_addr]   #本机、时间戳、参考地址
        self.tank[self.num] = entity
        # print(self.tank, self.main)

        #主基站自己进来了
        if rx_msr[rx_node_addr] == rx_node_addr:     
            self.main[rx_node_addr]=0          #整体的计数被后置了

        self.num += 1                          #栈内长
        #anthor_address  distance

    def display(self):
        for i in range(self.slot):
            if self.tank[i]==0:
                if i!=self.slot-1:
                    print("×",end="")
                else:print("×")
            else:
                if i!=self.slot-1:
                    print("√",end="")
                else:print("√")
    def count(self):   #统计,检查自己应该对哪个主基站进行定位

        #先对字典初始化,防止多次统计
        for key in self.main.keys():
            self.main[key]=0

        for i in range(self.slot):
            if self.tank[i]==0:
                pass
            else:
                #entity=[rx_node_addr,time_stamp,main_addr]
                if(self.tank[i][2] in self.main.keys()):
                    self.main[self.tank[i][2]]+=1   # 当前基站的值+1
                else:
                    pass
    def check(self):                          #想要返回主基站数组
        check_array=[]
        
        self.count()    #进行一次统计

        #将字典按照键值从大到小排序
        temp_main=sorted(self.main.items(),key = lambda x:x[1],reverse = True)
        #temp_main为[(key,value)]     
        # print(temp_main)            

        # check_array记录当前时间戳下定位的基站数
        for main_array in temp_main:       #V5.3改为遍历整个temp_main
            #V3中修改了符号
            if main_array[1]>=RX_NUM:
                if RX_NUM==3:
                    check_array.append(main_array[0])   #返回可以定位的主机站地址
                elif RX_NUM==2:                                                   
                    check_array.insert(0,main_array[0])   #测线优先级更高
                    
        return check_array
        

    def draw(self,rx_main_addr,latest_Addrs):  #给出可定位的主基站 返回原始数据
        # print(rx_main_addr, latest_Addrs)
        #返回的数据
        Addrs=[]
        Stamps=[]
        #抽取主基站
        #print(self.num)
        for i in range(self.num):
            entity=self.tank[i]
            if entity[0]==rx_main_addr:
                Addrs.append(entity[0])
                Stamps.append(entity[1])
        #抽取从基站
        for i in range(self.num):
            entity=self.tank[i]
            if (entity[2]==rx_main_addr and entity[0]!=rx_main_addr ):
                Addrs.append(entity[0])
                Stamps.append(entity[1])        

        self.flag=True
        #print(Addrs,len(Addrs))

        #                            Cardination3 重大更新                        #

        #if len(Addrs)>=rx_main_count[main_array[0]]   一定发生
        main_addr=Addrs[0]

        # utils.DataAnalysis(Addrs,main_addr,Stamps)     #抖动分析 

        #不分子区，直接返回  ---在之后的版本中已经没有不分子区的了
        if main_addr not in group_msr.keys():
            return [],Addrs,Stamps
        else:            #划分子区，直接将返回结果
            #print("----------辖区检测---------")
            if TX_NUM==2:
                for pair in group_msr[main_addr]:
                    one=pair[0]
                    if one in Addrs:
                        #发现一个立即检测结果
                        temp_Addrs=[main_addr,pair[0]]
                        temp_Stamps=[Stamps[Addrs.index(m)] for m in temp_Addrs ]
                        distance=utils.distance2D(RX_position[temp_Addrs[0]],RX_position[temp_Addrs[1]])
                        add_stamp=distance/C/Per_Stamp
                        temp_Stamps[1]=temp_Stamps[1]+add_stamp    #添加距离补偿
                        BS=[]       #坐标数组
                        for addr in temp_Addrs:               #取BS
                            BS.append(RX_position[addr])
                        R=[]
                        R.append( C*Per_Stamp*(temp_Stamps[1]-temp_Stamps[0]) )

                        # print("debug",BS,R)       #首先我们解决Straight算法本身的问题

                        result=utils.Straight(BS,R)
                        #结果是否位于辖区之中
                        # area_flag=True
                        # for place in BS:
                        #     if distance2D(result,place)<=1:
                        #         area_flag=False
                        # if area_flag:
                        #     if debug:
                        #         print("本次辖区返回成功!!",result)
                        return result,[main_addr],[]                        

                #都不在 返回一个空结果 
                return [],Addrs,Stamps   #自生自灭,并由正常TDOA进行维度选择

            if RX_NUM==3:
                
                pair_num=len(group_msr[main_addr])

                for pair in group_msr[main_addr]:
                    # print(group_msr[main_addr], Addrs)
                    #print(pair)
                    if pair[0] not in Addrs:
                        #print('pair[0] not in Addrs')
                        continue
                    if pair[1] not in Addrs:
                        #print('pair[1] not in Addrs')
                        continue 
                        
                        #这里表示该组全部位于Addrs中，对其进行定位

                    #先检测是否上一次使用过,如果用过就直接换序
                    
                    temp_Addrs=[main_addr,pair[0],pair[1]]

                    if utils.List_in(temp_Addrs,latest_Addrs):
                       temp_Addrs=latest_Addrs
                    
                    temp_Stamps=[Stamps[Addrs.index(m)] for m in temp_Addrs ]
                    #定位
                    temp_result,temp_other=utils.Wrap_Fang2D(temp_Addrs,temp_Stamps)

                    # print("-----------------", temp_result)

                    if temp_result==[0,0]:
                        continue

                    area_msr = [[src.min_x - 5, src.min_y - 5],[src.max_x * 1.5, src.max_y * 1.5]]

                    #结果是否位于辖区之中
                    if pair_num==1:
                        if utils.in_area(temp_result,area_msr):
                            return temp_result,temp_Addrs,[]
                        if utils.in_area(temp_other,area_msr):
                            return temp_other,temp_Addrs,[]
                        pass
                    else:
                        if utils.in_area(temp_result,area_msr):
                            return temp_result,temp_Addrs,[]
                return [],Addrs,Stamps   #自生自灭,并由正常TDOA进行维度选择

#标签
class tag:
    def __init__(self,index):
        global radius,floor_x,floor_y
        
        self.index=index  #int类地址
 
        self.timer1=timer()
        self.timer2=timer()
        self.timer3=timer()
        self.timer4=timer()
        self.timer5=timer()

        #顺序循环栈队列
        self.timers=[self.timer1,self.timer2,self.timer3,self.timer4,self.timer5]
        self.pointer=0

        #上次使用的定位算法
        self.latest_algorithm=""   #""、"Fang2D"、"Chan2D"
        self.latest_Addrs=[]          #上次使用的基站基站地址

        self.last_position=[]

        #上传数据
        self.upload_thread=None


    #循环队列寻找本次消息传入接口
    def get_pointer(self,index):
                        #该标签的第几次测距
        
        for i in range(0,5):
            if self.timers[i].index==index:
                return i

        else:
            self.pointer = (self.pointer + 1)%5    
            return self.pointer
        #如果没找到接口指针右移
 

    def read(self,tx_index,time_stamp,rx_node_addr,main_addr,k_Addr,hold_time):

        # print("当前读入标签: ", self.index)
        # global location_result
        #get pointer 取得本次指针
        self.pointer=self.get_pointer(tx_index)     # 该标签的第几次定位
        # print(self.pointer)
        
        #检查清理
        if tx_index!=self.timers[self.pointer].index:     
            self.timers[self.pointer].clear()
            self.timers[self.pointer].assign(tx_index)

        #获得栈指针
        timer_ptr=self.timers[self.pointer]     # 用第几个Timer

        #填写：本次数据的时间戳、定位基站、主基站
        timer_ptr.add(time_stamp,rx_node_addr,main_addr)    

    
        #收到信号定位上一次，上一次至少有一个
        #鲁棒性：本.1收入可能不进行上次定位，可能本.2再定位
        #只往前看一个栈即可  认为需要再往前看概率小
        temp_pointer=(self.pointer-1)%5

        if(self.timers[temp_pointer].flag==True):
            #最近一次已经被用已被用于定位了
            return 0

        #定位数组
        rx_main_addrs=self.timers[temp_pointer].check()

        #print("主程序断点",rx_main_addrs)
        
        if(len(rx_main_addrs)==0):   #这个栈里没有能定位的主机站
            # print("本栈中没有可定位的主机站")
            pass

        #组装基站信息执行定位
        #返回rx_num 和BS

        for rx_main_addr in rx_main_addrs:
        
            result,Addrs,Stamps=self.timers[temp_pointer].draw(rx_main_addr,self.latest_Addrs)
            # print("主程序断点",result,Addrs,Stamps)

            if result:
                # print("----------最终结果为------------")
                print(result, self.index)

                #初始化一次
                #print(self.last_position)
                if self.last_position==[]:
                    self.last_position=result

                #低通滤波
                result,self.last_position=utils.Low_Pass_Filter(result,self.last_position,0.3,0.3)
                
                
                #本次将用来定位 则绘图
                self.latest_Addrs=Addrs
                if SAVE_DATA_FLAG:
                    src.save_data(self.index, [result[0], result[1]])
                src.detection(self.index, result[0], result[1])

                #求主基站相对位置
                # rx_main_location=RX_position[rx_main_addr]
                # result[0]=result[0]-rx_main_location[0]
                # result[1]=result[1]-rx_main_location[1]
                #用线程执行upload,此时Addrs拿到的是主机站地址
                # self.upload_thread= Thread(target=upload_position,
                #                   args=(self.index,result,Addrs[0],'dimension'))
                #self.upload_thread.start()
                #进行数据上传

                break

            #if (rx_main_count[Addrs[0]]!=len(Addrs)):
            elif (not Addrs) or (Addrs[0] not in rx_msr.keys()):
                print(self.index,"所在位置定位信息不完整")
                continue

            else:
                #已经取消了单区域的流程
                print("丢弃")
                continue
                                
        pass        

def dataProcess(data):

    discode_data = []

    if len(data) % 36 == 0:
        # main_data = data[1:35]
        for i in range((int)(len(data) / 36)):
            discode_data.append(data[1 + i * 36 : 35 + i * 36])

        k_Addr=int(discode_data[0][31:35],16)
        hold_time=int(discode_data[0][27:30],16)

    else:
        return 
    
    for i in range(len(discode_data)):

        tx_addr=int(discode_data[i][0:4],16)             #标签地址
        tx_index=int(discode_data[i][5:7],16)           #标签序号
        time_stamp=int(discode_data[i][8:18],16)       #区域时间戳
        refer_index=int(discode_data[i][19:21],16)     #参考对表序号
        rx_node_addr=int(discode_data[i][22:26],16)   #本基站地址
    
        # print("基站",rx_node_addr)
        #main_addr=int(i.group("main_addr"),16)
        
        addr=rx_msr[rx_node_addr]

        #防止对表失败
        if(addr==0):
            #main_addr=8193
            print("基站",rx_node_addr,"出现对表失败情况")

        #均为字符串类型

        #若tx地址未收录，先为其创建实例
        
        if tx_addr not in tx_dict.keys():
            
            tx_dict[tx_addr]=tag(tx_addr)

            print("实例化",tx_addr)

        # print("tx_addr: ", other_tx_addr)
        tx_ptr=tx_dict[tx_addr]
        #将本次数据入栈
        tx_ptr.read(tx_index,time_stamp,rx_node_addr,addr,k_Addr,hold_time)
        #tx_ptr.display()

    time.sleep(0.1)