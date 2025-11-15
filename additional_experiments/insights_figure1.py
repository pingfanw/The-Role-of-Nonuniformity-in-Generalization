import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8)
x = np.arange(-0.5, 1.5, 0.01)
f_1 = np.zeros_like(x)
f_2 = np.zeros_like(x)
for i in range(len(x)):
    f_1[i] = min(x[i] ** 2, 0.1 * (x[i] - 1) ** 2)
    f_2[i] = min(x[i] ** 2, 1.9 * (x[i] - 1) ** 2)
f = 0.5 * (f_1 + f_2)

# plt.plot(x, f, '.-', label='f = 1/2*(f_1 + f_2)')
# plt.plot(x, f_1, '--', label='f_1')
# plt.plot(x, f_2, '--', label='f_2')
# plt.legend()
# plt.show()

int_value = 1.0+0.0001

# SGD
x0 = int_value
eta = 0.6
iteration = 200
SGD = np.zeros(iteration)
SGD[0] = x0

# SGD 更新
for i in range(iteration - 1):
    index = np.random.randint(1, 3)
    if index == 1:
        if x0 ** 2 >= 0.1 * (x0 - 1) ** 2:
            x1 = x0 - eta * 0.2 * (x0 - 1)
        else:
            x1 = x0 - eta * 2 * x0
    elif index == 2:
        if x0 ** 2 >= 1.9 * (x0 - 1) ** 2:
            x1 = x0 - eta * 3.8 * (x0 - 1)
        else:
            x1 = x0 - eta * 2 * x0

    SGD[i + 1] = x1
    x0 = x1

Loss = np.zeros(iteration)
gsgd_norm = np.zeros(iteration)
Tgsgd_norm = np.zeros(iteration)

for i in range(iteration):
    if SGD[i] ** 2 >= 0.1 * (SGD[i] - 1) ** 2 and SGD[i] ** 2 >= 1.9 * (SGD[i] - 1) ** 2:
        Loss[i] = 0.5 * (0.1 * (SGD[i] - 1) ** 2 + 1.9 * (SGD[i] - 1) ** 2)
        gsgd_norm[i] = 0.5 * np.linalg.norm(0.2 * (SGD[i] - 1) + 3.8 * (SGD[i] - 1))
        Tgsgd_norm[i] = 0.5 * (np.linalg.norm(0.2 * (SGD[i] - 1)) + np.linalg.norm(3.8 * (SGD[i] - 1)))
    elif SGD[i] ** 2 >= 0.1 * (SGD[i] - 1) ** 2 and SGD[i] ** 2 <= 1.9 * (SGD[i] - 1) ** 2:
        Loss[i] = 0.5 * (0.1 * (SGD[i] - 1) ** 2 + SGD[i] ** 2)
        gsgd_norm[i] = 0.5 * np.linalg.norm(0.2 * (SGD[i] - 1) + 2 * SGD[i])
        Tgsgd_norm[i] = 0.5 * (np.linalg.norm(0.2 * (SGD[i] - 1)) + np.linalg.norm(2 * SGD[i]))
    elif SGD[i] ** 2 <= 0.1 * (SGD[i] - 1) ** 2 and SGD[i] ** 2 >= 1.9 * (SGD[i] - 1) ** 2:
        Loss[i] = 0.5 * (SGD[i] ** 2 + 1.9 * (SGD[i] - 1) ** 2)
        gsgd_norm[i] = 0.5 * np.linalg.norm(0.2 * SGD[i] + 3.8 * (SGD[i] - 1))
        Tgsgd_norm[i] = 0.5 * (np.linalg.norm(0.2 * SGD[i]) + np.linalg.norm(3.8 * (SGD[i] - 1)))
    elif SGD[i] ** 2 <= 0.1 * (SGD[i] - 1) ** 2 and SGD[i] ** 2 <= 1.9 * (SGD[i] - 1) ** 2:
        Loss[i] = SGD[i] ** 2
        gsgd_norm[i] = 0.5 * np.linalg.norm(4 * SGD[i])
        Tgsgd_norm[i] = 0.5 * np.linalg.norm(4 * SGD[i])

plt.subplot(2, 2, 1)
plt.plot(x, f, '-', label='f', linewidth=3)
plt.plot(x, f_1, '--', label='f_1', linewidth=2)
plt.plot(x, f_2, '--', label='f_2', linewidth=2)
plt.plot(SGD, Loss, 'ro-', label='SGD')
plt.plot(SGD[-1], Loss[-1], '*', label='minima', color='#00FF00', markersize=15)
plt.legend()
plt.grid()

# MSGD
mx0 = int_value
iteration = 200
mu = 0.5
v0 = 0.05
MSGD = np.zeros(iteration)
MSGD[0] = mx0

# MSGD 更新
for i in range(iteration - 1):
    index = np.random.randint(1, 3)
    if index == 1:
        if mx0 ** 2 >= 0.1 * (mx0 - 1) ** 2:
            v1 = mu * v0 - eta * 0.2 * (mx0 - 1)
        else:
            v1 = mu * v0 - eta * 2 * mx0
    elif index == 2:
        if mx0 ** 2 >= 1.9 * (mx0 - 1) ** 2:
            v1 = mu * v0 - eta * 3.8 * (mx0 - 1)
        else:
            v1 = mu * v0 - eta * 2 * mx0
    mx1 = mx0 + v1
    MSGD[i + 1] = mx1
    mx0 = mx1
    v0 = v1

MLoss = np.zeros(iteration)
gM_norm = np.zeros(iteration)
TgM_norm = np.zeros(iteration)

for i in range(iteration):
    if MSGD[i] ** 2 >= 0.1 * (MSGD[i] - 1) ** 2 and MSGD[i] ** 2 >= 1.9 * (MSGD[i] - 1) ** 2:
        MLoss[i] = 0.5 * (0.1 * (MSGD[i] - 1) ** 2 + 1.9 * (MSGD[i] - 1) ** 2)
        gM_norm[i] = 0.5 * np.linalg.norm(0.2 * (MSGD[i] - 1) + 3.8 * (MSGD[i] - 1))
        TgM_norm[i] = 0.5 * (np.linalg.norm(0.2 * (MSGD[i] - 1)) + np.linalg.norm(3.8 * (MSGD[i] - 1)))
    elif MSGD[i] ** 2 >= 0.1 * (MSGD[i] - 1) ** 2 and MSGD[i] ** 2 <= 1.9 * (MSGD[i] - 1) ** 2:
        MLoss[i] = 0.5 * (0.1 * (MSGD[i] - 1) ** 2 + MSGD[i] ** 2)
        gM_norm[i] = 0.5 * np.linalg.norm(0.2 * (MSGD[i] - 1) + 2 * MSGD[i])
        TgM_norm[i] = 0.5 * (np.linalg.norm(0.2 * (MSGD[i] - 1)) + np.linalg.norm(2 * MSGD[i]))
    elif MSGD[i] ** 2 <= 0.1 * (MSGD[i] - 1) ** 2 and MSGD[i] ** 2 >= 1.9 * (MSGD[i] - 1) ** 2:
        MLoss[i] = 0.5 * (MSGD[i] ** 2 + 1.9 * (MSGD[i] - 1) ** 2)
        gM_norm[i] = 0.5 * np.linalg.norm(0.2 * MSGD[i] + 3.8 * (MSGD[i] - 1))
        TgM_norm[i] = 0.5 * (np.linalg.norm(0.2 * MSGD[i]) + np.linalg.norm(3.8 * (MSGD[i] - 1)))
    elif MSGD[i] ** 2 <= 0.1 * (MSGD[i] - 1) ** 2 and MSGD[i] ** 2 <= 1.9 * (MSGD[i] - 1) ** 2:
        MLoss[i] = MSGD[i] ** 2
        gM_norm[i] = 0.5 * np.linalg.norm(4 * MSGD[i])
        TgM_norm[i] = 0.5 * np.linalg.norm(4 * MSGD[i])

plt.subplot(2, 2, 2)
plt.plot(x, f, '-', label='f', linewidth=3)
plt.plot(x, f_1, '--', label='f_1', linewidth=2)
plt.plot(x, f_2, '--', label='f_2', linewidth=2)
plt.plot(MSGD, MLoss, 'ro-', label='MSGD')
plt.plot(MSGD[-1], MLoss[-1], '*', label='minima', color='#00FF00', markersize=15)
plt.legend()
plt.grid()

# NSGD
iteration = 200
nx0 = int_value
mu = 0.5
v0 = 0.01
NSGD = np.zeros(iteration)
NSGD[0] = nx0

# NSGD 更新
for i in range(iteration - 1):
    mu = (i) / (i + 2)
    index = np.random.randint(1, 3)
    if index == 1:
        if nx0 ** 2 >= 0.1 * (nx0 - 1) ** 2:
            v1 = mu * v0 - eta * 0.2 * ((nx0 + mu * v0) - 1)
            nx1 = nx0 + v1
        else:
            v1 = mu * v0 - eta * 2 * (nx0 + mu * v0)
            nx1 = nx0 + v1
    else:
        if nx0 ** 2 >= 1.9 * (nx0 - 1) ** 2:
            v1 = mu * v0 - eta * 3.8 * ((nx0 + mu * v0) - 1)
            nx1 = nx0 + v1
        else:
            v1 = mu * v0 - eta * 2 * (nx0 + mu * v0)
            nx1 = nx0 + v1

    NSGD[i + 1] = nx1
    nx0 = nx1
    v0 = v1

NLoss = np.zeros(iteration)
gN_norm = np.zeros(iteration)
TgN_norm = np.zeros(iteration)

for i in range(iteration):
    x_i = NSGD[i]
    x_i_minus_1 = x_i - 1
    term1 = 0.1 * x_i_minus_1 ** 2
    term2 = 1.9 * x_i_minus_1 ** 2
    x_i_squared = x_i ** 2

    if x_i_squared >= term1 and x_i_squared >= term2:
        NLoss[i] = 0.5 * (term1 + term2)
        gN_norm[i] = 0.5 * np.linalg.norm(0.2 * x_i_minus_1 + 3.8 * x_i_minus_1)
        TgN_norm[i] = 0.5 * (np.linalg.norm(0.2 * x_i_minus_1) + np.linalg.norm(3.8 * x_i_minus_1))
    elif x_i_squared >= term1 and x_i_squared <= term2:
        NLoss[i] = 0.5 * (term1 + x_i_squared)
        gN_norm[i] = 0.5 * np.linalg.norm(0.2 * x_i_minus_1 + 2 * x_i)
        TgN_norm[i] = 0.5 * (np.linalg.norm(0.2 * x_i_minus_1) + np.linalg.norm(2 * x_i))
    elif x_i_squared <= term1 and x_i_squared >= term2:
        NLoss[i] = 0.5 * (x_i_squared + term2)
        gN_norm[i] = 0.5 * np.linalg.norm(0.2 * x_i + 3.8 * x_i_minus_1)
        TgN_norm[i] = 0.5 * (np.linalg.norm(0.2 * x_i) + np.linalg.norm(3.8 * x_i_minus_1))
    elif x_i_squared <= term1 and x_i_squared <= term2:
        NLoss[i] = x_i_squared
        gN_norm[i] = 0.5 * np.linalg.norm(4 * x_i)
        TgN_norm[i] = 0.5 * np.linalg.norm(4 * x_i)

plt.subplot(2, 2, 3)
plt.plot(x, f, '-', label='f', linewidth=3)
plt.plot(x, f_1, '--', label='f_1', linewidth=2)
plt.plot(x, f_2, '--', label='f_2', linewidth=2)
plt.plot(NSGD, NLoss, 'ro-', label='NSGD')
plt.plot(NSGD[-1], NLoss[-1], '*', label='minima', color='#00FF00', markersize=15)
plt.legend()
plt.grid()
# SAM
iteration = 200
sx0 = int_value
rho = 0.01
SAM = np.zeros(iteration)
SAM[0] = sx0

# SAM 更新
for i in range(iteration - 1):
    index = np.random.randint(1, 3)
    if index == 1:
        if sx0 ** 2 >= 0.1 * (sx0 - 1) ** 2:
            sx1 = sx0 - eta * 0.2 * ((sx0 + rho) - 1)
        else:
            sx1 = sx0 - eta * 2 * (sx0 + rho)
    else:
        if sx0 ** 2 >= 1.9 * (sx0 - 1) ** 2:
            sx1 = sx0 - eta * 3.8 * ((sx0 + rho) - 1)
        else:
            sx1 = sx0 - eta * 2 * (sx0 + rho)

    SAM[i + 1] = sx1
    sx0 = sx1

sLoss = np.zeros(iteration)
sg_norm = np.zeros(iteration)
Tsg_norm = np.zeros(iteration)

for i in range(iteration):
    x_i = SAM[i]
    x_i_minus_1 = x_i - 1
    term1 = 0.1 * x_i_minus_1 ** 2
    term2 = 1.9 * x_i_minus_1 ** 2
    x_i_squared = x_i ** 2

    if x_i_squared >= term1 and x_i_squared >= term2:
        sLoss[i] = 0.5 * (term1 + term2)
        sg_norm[i] = 0.5 * np.linalg.norm(0.2 * x_i_minus_1 + 3.8 * x_i_minus_1)
        Tsg_norm[i] = 0.5 * (np.linalg.norm(0.2 * x_i_minus_1) + np.linalg.norm(3.8 * x_i_minus_1))
    elif x_i_squared >= term1 and x_i_squared <= term2:
        sLoss[i] = 0.5 * (term1 + x_i_squared)
        sg_norm[i] = 0.5 * np.linalg.norm(0.2 * x_i_minus_1 + 2 * x_i)
        Tsg_norm[i] = 0.5 * (np.linalg.norm(0.2 * x_i_minus_1) + np.linalg.norm(2 * x_i))
    elif x_i_squared <= term1 and x_i_squared >= term2:
        sLoss[i] = 0.5 * (x_i_squared + term2)
        sg_norm[i] = 0.5 * np.linalg.norm(2 * x_i + 3.8 * x_i_minus_1)
        Tsg_norm[i] = 0.5 * (np.linalg.norm(2 * x_i) + np.linalg.norm(3.8 * x_i_minus_1))
    elif x_i_squared <= term1 and x_i_squared <= term2:
        sLoss[i] = x_i_squared
        sg_norm[i] = 0.5 * np.linalg.norm(4 * x_i)
        Tsg_norm[i] = 0.5 * np.linalg.norm(4 * x_i)

plt.subplot(2, 2, 4)
plt.plot(x, f, '-', label='f', linewidth=3)
plt.plot(x, f_1, '--', label='f_1', linewidth=2)
plt.plot(x, f_2, '--', label='f_2', linewidth=2)
plt.plot(SAM, sLoss, 'ro-', label='SAM')
plt.plot(SAM[-1], sLoss[-1], '*', label='minima', color='#00FF00', markersize=15)
plt.legend()
plt.grid()
plt.show()


