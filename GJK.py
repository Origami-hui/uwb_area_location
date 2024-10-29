#GJK碰撞检测算法
shape={}

#预备工作
def cal_normal_3d(shape):#求平面三点的法向量
    nx=(shape[1][1]-shape[0][1])*(shape[2][2]-shape[0][2])-(shape[1][2]-shape[0][2])*(shape[2][1]-shape[0][1])
    ny=(shape[1][2]-shape[0][2])*(shape[2][0]-shape[0][0])-(shape[1][0]-shape[0][0])*(shape[2][2]-shape[0][2])
    nz=(shape[1][0]-shape[0][0])*(shape[2][1]-shape[0][1])-(shape[1][1]-shape[0][1])*(shape[2][0]-shape[0][0])
    #print(nx,ny,nz)
    return (nx,ny,nz)

def cal_point_3d(shape):#求平面两向量点乘
    nx=shape[0][0]*shape[1][0]
    ny=shape[0][1]*shape[1][1]
    nz=shape[0][2]*shape[1][2]
    return nx+ny+nz

def check_in_2d(shape):#已知原点落在三角形平面，判断原点是否在平面三角形内
    n=cal_normal_3d([shape[0],shape[1],shape[2]])
    n1=cal_normal_3d([n+shape[1],shape[1],shape[2]])
    n2=cal_normal_3d([n+shape[2],shape[2],shape[0]])
    n3=cal_normal_3d([n+shape[0],shape[0],shape[1]])
    if (cal_point_3d([n1,shape[1]])*cal_point_3d([n1,(shape[1][0]-shape[0][0],shape[1][1]-shape[0][1],shape[1][2]-shape[0][2])])<0):
        return False
    elif (cal_point_3d([n2,shape[2]])*cal_point_3d([n2,(shape[2][0]-shape[1][0],shape[2][1]-shape[1][1],shape[2][2]-shape[1][2])])<0):
        return False
    elif (cal_point_3d([n3,shape[0]])*cal_point_3d([n3,(shape[0][0]-shape[2][0],shape[0][1]-shape[2][1],shape[0][2]-shape[2][2])])<0):
        return False
    else:
        return True

def check_in_3d(shape):#判断原点是否在四面体内部
    n1=cal_normal_3d([shape[0],shape[1],shape[2]])#求每个面的法向量
    n2=cal_normal_3d([shape[0],shape[1],shape[3]])
    n3=cal_normal_3d([shape[0],shape[2],shape[3]])
    n4=cal_normal_3d([shape[1],shape[2],shape[3]])
    if (cal_point_3d([n4,shape[1]])*cal_point_3d([n4,(shape[1][0]-shape[0][0],shape[1][1]-shape[0][1],shape[1][2]-shape[0][2])])<0):
        return False
    elif (cal_point_3d([n1,shape[2]])*cal_point_3d([n1,(shape[2][0]-shape[3][0],shape[2][1]-shape[3][1],shape[2][2]-shape[3][2])])<0):
        return False
    elif (cal_point_3d([n2,shape[3]])*cal_point_3d([n2,(shape[3][0]-shape[2][0],shape[3][1]-shape[2][1],shape[3][2]-shape[2][2])])<0):
        return False
    elif (cal_point_3d([n3,shape[0]])*cal_point_3d([n3,(shape[0][0]-shape[1][0],shape[0][1]-shape[1][1],shape[0][2]-shape[1][2])])<0):
        return False
    else:
        return True


def check(shape):#判定单纯形是否包含原点
    d=len(shape)
    if(d==1):
        if(shape==(0,0,0)):
            return True
        else:
            return False
    elif(d==2):
        minx=min(shape[0][0],shape[1][0])
        maxx=max(shape[0][0],shape[1][0])
        comp=[shape[0],shape[1],(0,0,0)]
        if(cal_normal_3d(comp)==(0,0,0) and minx<=0 and maxx>=0):
            return True
        else:
            return False
    elif (d==3):
        n=cal_normal_3d([shape[0],shape[1],shape[2]])#求平面法向量
        if(cal_point_3d([n,shape[0]])==0 and check_in_2d([shape[0],shape[1],shape[2]])==True):
            return True
        else:
            return False
    else:#d=4
        return check_in_3d([shape[0],shape[1],shape[2],shape[3]])

def support(shape,target):#找出多面体给定方向上的两个极端点(给定方向均为平行与坐标轴的单位向量)
    for i in range(3):#若有多个极端点，返回离原点最远或最近的两个极端点
        if (target[i]==1):
            comp=[x[i] for x in shape]
            tup=[(i,comp[i])for i in range(len(comp))]
            max_list=[shape[i]for i,n in tup if n==max(comp)]
            min_list=[shape[i]for i,n in tup if n==min(comp)]
            maxmark=find_max(max_list)
            minmark=find_min(min_list)
            return [max_list[maxmark],min_list[minmark]]
        elif(target[i]==-1):
            comp=[x[i] for x in shape]
            tup=[(i,comp[i])for i in range(len(comp))]
            max_list=[shape[i]for i,n in tup if n==max(comp)]
            min_list=[shape[i]for i,n in tup if n==min(comp)]
            maxmark=find_max(max_list)
            minmark=find_min(min_list)
            return [min_list[minmark],max_list[maxmark]]
            


def find_max(shape):#找到多面体离原点最远的点
    comp=[i[0]**2+i[1]**2+i[2]**2 for i in shape]
    return comp.index(max(comp))

def find_min(shape):#找到多面体离原点最近的点
    comp=[i[0]**2+i[1]**2+i[2]**2 for i in shape]
    return comp.index(min(comp))

def find_mink_point(shape1,shape2,target):#找到两多面体在指定方向的极端值相减得到的坐标
    #print(support(shape1,target),support(shape2,target))
    tx=support(shape1,target)[0][0]-support(shape2,target)[1][0]
    ty=support(shape1,target)[0][1]-support(shape2,target)[1][1]
    tz=support(shape1,target)[0][2]-support(shape2,target)[1][2]
    return (tx,ty,tz)

def check_same_3d(shape):#判断四面体四个点是否在同一平面
    comp1=cal_normal_3d([shape[0],shape[1],shape[2]])
    comp2=cal_normal_3d([shape[3],shape[1],shape[2]])
    if(cal_normal_3d([(0,0,0),comp1,comp2])==(0,0,0)):
        return True
    else:
        return False

def check_same_2d(shape):#判断平面三个点是否共线
    comp1=(shape[0][0]-shape[1][0],shape[0][1]-shape[1][1],shape[0][2]-shape[1][2])
    comp2=(shape[2][0]-shape[1][0],shape[2][1]-shape[1][1],shape[2][2]-shape[1][2])
    if(cal_normal_3d([(0,0,0),comp1,comp2])==(0,0,0)):
        return True
    else:
        return False

#进入迭代
def gjk(shape):
    num=len(shape)
    for i in range(num):
        for j in range(i):#每个图形逐一比较(本次比较shape[i]与shape[j])
            pure_shape=[]#创建单纯形    
            flag=0#碰撞检测标志 
            # print('当前单纯形：')       
            for k in range(6):#往六个方向迭代
                if (k==0):
                    target=(1,0,0)
                elif (k==1):
                    target=(0,1,0)
                elif (k==2):
                    target=(0,0,1)
                elif (k==3):
                    target=(-1,0,0)
                elif (k==4):
                    target=(0,-1,0)
                else:
                    target=(0,0,-1)
                pure_shape.append(find_mink_point(shape[i],shape[j],target))
                if(len(pure_shape)==4 and check_same_3d(pure_shape)==True):
                    pure_shape.pop(find_max(pure_shape))
                if(len(pure_shape)==3 and check_same_2d(pure_shape)==True):
                    pure_shape.pop(find_max(pure_shape))
                if(len(pure_shape)==2 and pure_shape[1]==pure_shape[0]):
                    pure_shape.pop(0)
                print(pure_shape)
                if(check(pure_shape)==True):
                    print('shape{}与shape{}发生了碰撞'.format(i,j))
                    flag=1
                    return False
                else:
                    if(len(pure_shape)==4):#当当前单纯形为四面体，弹出离原点最远的点进行下一次迭代
                        pure_shape.pop(find_max(pure_shape))
            if(flag==0):
                print('shape{}与shape{}没有发生碰撞'.format(i,j))
                return True
          

# shape[0]=[(0 ,0 ,0), (1 ,0 ,0), (1 ,1 ,0), (0 ,1 ,0), (0 ,0 ,1), (1 ,0 ,1), (1 ,1 ,1), (0 ,1 ,1)]
# shape[1]=[(0.9 ,0.9 ,0.9), (2 ,1 ,1), (2 ,2 ,1), (1 ,2 ,1), (1 ,1 ,2), (2 ,1 ,2), (2 ,2 ,2), (1 ,2 ,2)]
# shape[2]=[(3 ,3 ,2), (5 ,3 ,2), (4 ,5 ,2), (4 ,4 ,4)]

# shape[0]=[(1, 1 ,0), (2 ,3 ,0), (4 ,5 ,0), (5 ,1 ,0)]
# shape[1]=[(3 ,5 ,0), (7 ,4 ,0), (4 ,3, 0), (6, 5, 0)]

# gjk(shape)