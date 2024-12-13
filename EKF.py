import math
from config import *
import numpy as np
import numdifftools as nd


# def prediction_step(x, P, F, Q):
#     x_pred = np.dot(F, x)
#     P_pred = np.dot(F, np.dot(P, F.T)) + Q
#     return x_pred, P_pred
#
#
# def update_step(x_pred, P_pred, z, H, R):
#     y = z - np.dot(H, x_pred)
#     S = np.dot(H, np.dot(P_pred, H.T)) + R
#     K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))
#     x_new = x_pred + np.dot(K, y)
#     P_new = P_pred - np.dot(K, np.dot(H, P_pred))
#     return x_new, P_new


def control_psi(psi):
    while psi > np.pi or psi < -np.pi:
        if psi > np.pi:
            psi = psi - 2 * np.pi
        if psi < -np.pi:
            psi = psi + 2 * np.pi
    return psi


# def ekf(obsX, obsY):
#     # 测试数据
#     dt = 0.5  # 时间间隔
#
#     F = np.array([[1, dt, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, dt],
#                   [0, 0, 0, 1]])  # 状态转移矩阵
#     H = np.array([[1, 0, 0, 0],
#                   [0, 0, 1, 0]])  # 观测矩阵
#     Q = np.diag([0.01, 0.01, 0.01, 0.01])  # 系统噪声协方差矩阵
#     R = np.diag([1, 1])  # 观测噪声协方差矩阵
#
#     # 初始化状态向量和协方差矩阵
#     x = np.array([obsX[0], 0, obsY[0], 0])  # 初始状态向量 [位置x, 速度x, 位置y, 速度y]
#     if len(obsX) > 1:
#         x = np.array([obsX[0], (obsX[1] - obsX[0]) / dt, obsY[0], (obsY[1] - obsY[0]) / dt])
#     P = np.diag([1, 1, 1, 1])  # 初始协方差矩阵
#
#     # # 模拟观测数据
#     # true_states = []
#     # observations = []
#     # for t in range(100):
#     #     #     true_states.append([t * dt, 1, t * dt, 1])
#     #     #     z = np.dot(H, true_states[-1]) + np.random.multivariate_normal([0, 0], R)
#     #     #     # print(z)
#     #     #     observations.append(z)
#
#     # EKF 迭代
#     for i in range(len(obsX)):
#         x_pred, P_pred = prediction_step(x, P, F, Q)
#         x, P = update_step(x_pred, P_pred, np.array([obsX[i], obsY[i]]), H, R)
#
#     print("Origin state:", np.array([obsX[-1], obsY[-1]]))
#     print("Final state estimate:", x)
#     return x[0], x[2]


transition_function = lambda y, dt: np.vstack((
    y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * dt) - np.sin(y[3])),
    y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * dt) + np.cos(y[3])),
    y[2],
    y[3] + y[4] * dt,
    y[4]))

# when omega is 0
transition_function_1 = lambda m, dt: np.vstack((m[0] + m[2] * np.cos(m[3]) * dt,
                                                 m[1] + m[2] * np.sin(m[3]) * dt,
                                                 m[2],
                                                 m[3] + m[4] * dt,
                                                 m[4]))

J_A = nd.Jacobian(transition_function)
J_A_1 = nd.Jacobian(transition_function_1)

x_list = [[] for _ in range(4)]
y_list = [[] for _ in range(4)]
v_list = [[] for _ in range(4)]
theta_list = [[] for _ in range(4)]
omiga_list = [[] for _ in range(4)]

dt = 0.3

# 初始化
state = [np.zeros([5, 1]) for _ in range(4)]

P = [np.diag([1.0, 1.0, 1.0, 1.0, 1.0]) for _ in range(4)]
H = [np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]]) for _ in range(4)]

R = [np.array([[0.0225, 0.], [0., 0.0225]]) for _ in range(4)]
# process noise standard deviation for a
std_noise_a = 2.0
# process noise standard deviation for yaw acceleration
std_noise_yaw_dd = 0.3

I = np.eye(5)


def saveStates(tag_id, x, y, v, theta, omiga):
    x_list[tag_id - Config.TX_STR_NUM].append(x)
    y_list[tag_id - Config.TX_STR_NUM].append(y)
    v_list[tag_id - Config.TX_STR_NUM].append(v)
    theta_list[tag_id - Config.TX_STR_NUM].append(theta)
    omiga_list[tag_id - Config.TX_STR_NUM].append(omiga)


def ekf(tag_id, x, y, v, theta, omiga):
    global state, dt, P, H, R, std_noise_a, std_noise_yaw_dd, I

    saveStates(tag_id, x, y, v, theta, omiga)

    if len(x_list[tag_id - Config.TX_STR_NUM]) <= 1:
        state[tag_id - Config.TX_STR_NUM][0, 0] = x
        state[tag_id - Config.TX_STR_NUM][1, 0] = y

    t_measurement = [x, y, v / dt, theta, omiga / dt]
    z = np.array([[t_measurement[0]], [t_measurement[1]]])

    # print(state[tag_id - Config.TX_STR_NUM])

    if np.abs(state[tag_id - Config.TX_STR_NUM][4, 0]) < 0.1:
        state[tag_id - Config.TX_STR_NUM] = transition_function_1(state[tag_id - Config.TX_STR_NUM].ravel().tolist(), dt)
        state[tag_id - Config.TX_STR_NUM][3, 0] = control_psi(state[tag_id - Config.TX_STR_NUM][3, 0])
        JA = J_A_1(state[tag_id - Config.TX_STR_NUM].ravel().tolist(), dt)
    else:
        state[tag_id - Config.TX_STR_NUM] = transition_function(state[tag_id - Config.TX_STR_NUM].ravel().tolist(), dt)
        state[tag_id - Config.TX_STR_NUM][3, 0] = control_psi(state[tag_id - Config.TX_STR_NUM][3, 0])
        JA = J_A(state[tag_id - Config.TX_STR_NUM].ravel().tolist(), dt)

    JA = np.squeeze(JA)
    # print(JA)

    G = np.zeros([5, 2])
    G[0, 0] = 0.5 * dt * dt * np.cos(state[tag_id - Config.TX_STR_NUM][3, 0])
    G[1, 0] = 0.5 * dt * dt * np.sin(state[tag_id - Config.TX_STR_NUM][3, 0])
    G[2, 0] = dt
    G[3, 1] = 0.5 * dt * dt
    G[4, 1] = dt

    Q_v = np.diag([std_noise_a * std_noise_a, std_noise_yaw_dd * std_noise_yaw_dd])
    Q = np.dot(np.dot(G, Q_v), G.T)

    P[tag_id - Config.TX_STR_NUM] = np.dot(np.dot(JA, P[tag_id - Config.TX_STR_NUM]), JA.T) + Q

    S = np.dot(np.dot(H[tag_id - Config.TX_STR_NUM], P[tag_id - Config.TX_STR_NUM]), H[tag_id - Config.TX_STR_NUM].T) + R[
        tag_id - Config.TX_STR_NUM]
    K = np.dot(np.dot(P[tag_id - Config.TX_STR_NUM], H[tag_id - Config.TX_STR_NUM].T), np.linalg.inv(S))

    ky = z - np.dot(H[tag_id - Config.TX_STR_NUM], state[tag_id - Config.TX_STR_NUM])
    ky[1, 0] = control_psi(ky[1, 0])
    state[tag_id - Config.TX_STR_NUM] = state[tag_id - Config.TX_STR_NUM] + np.dot(K, ky)
    state[tag_id - Config.TX_STR_NUM][3, 0] = control_psi(state[tag_id - Config.TX_STR_NUM][3, 0])
    # Update the error covariance
    P[tag_id - Config.TX_STR_NUM] = np.dot((I - np.dot(K, H[tag_id - Config.TX_STR_NUM])), P[tag_id - Config.TX_STR_NUM])

    # state[tag_id - Config.TX_STR_NUM] = state[tag_id - Config.TX_STR_NUM].ravel().tolist()
    # print(state[tag_id - Config.TX_STR_NUM])
    return state[tag_id - Config.TX_STR_NUM][0, 0], state[tag_id - Config.TX_STR_NUM][1, 0]


def resetState(tag_id, x, y, v, theta, omiga):
    state[tag_id - Config.TX_STR_NUM][0, 0] = x
    state[tag_id - Config.TX_STR_NUM][1, 0] = y
    state[tag_id - Config.TX_STR_NUM][2, 0] = v
    state[tag_id - Config.TX_STR_NUM][3, 0] = theta
    state[tag_id - Config.TX_STR_NUM][4, 0] = omiga
