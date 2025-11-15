import numpy as np
import matplotlib.pyplot as plt
import palettable

np.random.seed(8)

n = 5000
iteration = 10000

eta = 0.01
rho = 0.005

b = 100
_lambda = 0.3  

d = 1
a = (n - b) / (b * (n - 1))

H = np.eye(d)
dataX = np.random.multivariate_normal(mean=np.zeros(d), cov=H, size=n) * 2  # shape (n, d)
dataX = dataX.flatten()  

theta_star1 = 0.1
theta_star2 = -0.1
es = np.random.randn(n)
dataY = dataX * (theta_star1 * theta_star2) + es

sam = np.array([0.1, 0.19])
cov = sam.copy()
sgd = sam.copy()

SAM = [sam.copy()]
COV = [cov.copy()]
SGD = [sgd.copy()]

def loss_fn(params):
    return 1/(2*n) * np.linalg.norm(params[0] * params[1] * dataX - dataY)**2

loss_sam = [loss_fn(sam)]
loss_cov = [loss_fn(cov)]
loss_sgd = [loss_fn(sgd)]

for k in range(iteration):
    idx = np.random.randint(0, n, size=b)
    dataXX = dataX[idx]
    dataYY = dataY[idx]

    gsam = np.zeros(2)
    for i in range(b):
        residual = sam[0] * sam[1] * dataXX[i] - dataYY[i]
        gsam += residual * dataXX[i] * np.array([sam[1], sam[0]]) / b
    gsam_norm = np.linalg.norm(gsam) + 1e-5

    pert_sam = sam + rho * gsam / gsam_norm
    rgsam = np.zeros(2)
    for i in range(b):
        residual = pert_sam[0] * pert_sam[1] * dataXX[i] - dataYY[i]
        rgsam += residual * dataXX[i] * np.array([pert_sam[1], pert_sam[0]])
    sam -= eta * rgsam / b
    SAM.append(sam.copy())


    Gcov = np.zeros(2)
    H_b = np.zeros((2, 2))
    hv_b = np.zeros(2)
    for i in range(b):
        # gradient component
        residual = cov[0] * cov[1] * dataXX[i] - dataYY[i]
        gcov_i = residual * dataXX[i] * np.array([cov[1], cov[0]])
        Gcov += gcov_i
        H_i = np.array([
            [cov[1]**2 * dataXX[i]**2, 2*cov[0]*cov[1]*dataXX[i]**2 - dataXX[i]*dataYY[i]],
            [2*cov[0]*cov[1]*dataXX[i]**2 - dataXX[i]*dataYY[i], cov[0]**2 * dataXX[i]**2]
        ])
        hv_b += H_i @ gcov_i / b
        H_b += H_i
    Hv_b = hv_b - (H_b @ Gcov) / (b**2)
    cov -= eta * (Gcov / b + a * _lambda * Hv_b)
    COV.append(cov.copy())

 
    Gsgd = np.zeros(2)
    for i in range(b):
        residual = sgd[0] * sgd[1] * dataXX[i] - dataYY[i]
        Gsgd += residual * dataXX[i] * np.array([sgd[1], sgd[0]])
    sgd -= eta * Gsgd / b
    SGD.append(sgd.copy())

   
    loss_sam.append(loss_fn(sam))
    loss_cov.append(loss_fn(cov))
    loss_sgd.append(loss_fn(sgd))

theta_starx = np.arange(-0.25, 0.25, 0.01)
theta_stary = np.arange(-0.25, 0.25, 0.01)
f = np.zeros((len(theta_starx), len(theta_stary)))
for i, tx in enumerate(theta_starx):
    for j, ty in enumerate(theta_stary):
        f[i, j] = 1/(2*n) * np.linalg.norm(tx * ty * dataX - dataY)**2

plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "serif",  
    "font.serif": ["Computer Modern Roman"]  
})
colors = palettable.colorbrewer.qualitative.Set2_5.mpl_colors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))  

CS = ax1.contour(theta_starx, theta_stary, f.T, levels=30)
ax1.clabel(CS, inline=1, fontsize=8)

SAM_arr = np.array(SAM)
SGD_arr = np.array(SGD)
COV_arr = np.array(COV)

ax1.plot(SGD_arr[:, 0], SGD_arr[:, 1], linewidth=2, label='SGD')
ax1.plot(SAM_arr[:, 0], SAM_arr[:, 1], linewidth=2, label='SAM')
ax1.plot(COV_arr[:, 0], COV_arr[:, 1], linewidth=2, label='COV')

z = np.sum(dataY * dataX) / np.sum(dataX**2)
O1 = np.linspace(-0.2, -0.05, 16)
O2 = z / O1

ax1.plot(-np.sqrt(abs(z)), np.sqrt(abs(z)), '*', ms=15, label='minima', color=colors[3])
ax1.plot(np.sqrt(abs(z)), -np.sqrt(abs(z)), '*', ms=15, color=colors[3])

ax1.legend(fontsize=20)
ax1.set_aspect('equal', 'box')
ax1.set_xlabel(r'$\theta_1$', fontsize=25, fontweight='bold')
ax1.set_ylabel(r'$\theta_2$', fontsize=25, fontweight='bold')
ax1.tick_params(axis='both', labelsize=25)

sgd_norm = [np.linalg.norm(x) for x in SGD]
sam_norm = [np.linalg.norm(x) for x in SAM]
cov_norm = [np.linalg.norm(x) for x in COV]

ax2.plot(sgd_norm, label='SGD')
ax2.plot(sam_norm, label='SAM')
ax2.plot(cov_norm, label='COV')

ax2.legend(fontsize=20)
ax2.set_xlabel('Iteration', fontsize=25, fontweight='bold')
ax2.set_ylabel(r'$L_2$ Norm', fontsize=25, fontweight='bold')
ax2.tick_params(axis='both', labelsize=25)

plt.tight_layout() 
plt.show()
