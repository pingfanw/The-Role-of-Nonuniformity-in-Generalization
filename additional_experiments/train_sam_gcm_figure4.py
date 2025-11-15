import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import torch
from sam_gcm import *


fig, axs = plt.subplots(1, 3, figsize=(7, 2), dpi=300)

n, n_test, d, r = 100, 1000, 150, 3

H = np.eye(d)
dataX = np.random.multivariate_normal(np.zeros(d), H, n)
PdataX = np.random.multivariate_normal(np.zeros(d), H, n_test)

theta_star = np.random.randn(d)
theta_star[r:] = 0
theta_star = np.abs(theta_star) / np.linalg.norm(theta_star)

dataY = dataX @ theta_star + np.random.randn(n) * 0.2
PdataY = PdataX @ theta_star

num_epochs = 400
int1, int2, ep = np.random.randn(d), np.random.randn(d), 0.2
eta, b = 0.05, 10

rho_sam, rho_cov, xi = 0.3, 0.0005, 0.000001

## inttional
SGD_w0, SGD_ww0 = ep * int1, ep * int2
SAM_w0, SAM_ww0 = ep * int1, ep * int2
CSAM0_w0, CSAM0_ww0 = ep * int1, ep * int2



## Train
SGD_Loss_train, SGD_Loss_test, SGD_L1 = sgd(SGD_w0, SGD_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs)
SAM_Loss_train, SAM_Loss_test, SAM_L1 = sam(SAM_w0, SAM_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_sam, xi)
CSAM0_Loss_train, CSAM0_Loss_test, CSAM0_L1 = SAMGCM(CSAM0_w0, CSAM0_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs,  rho_cov, xi, d)


# plt.figure(figsize=(15,4), dpi=300)
plt.subplot(1,3,1)
plt.yscale('log')
plt.plot(np.arange(1, num_epochs+1),SGD_Loss_test, label=r'SGD', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),SAM_Loss_test, label=r'SAM', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),CSAM0_Loss_test, label=r'SAMGCM', linewidth=2)
plt.xlabel('Epoch', labelpad=5)
plt.ylabel('Test loss', labelpad=5)
legend_properties = {'size': 5}
plt.legend(prop=legend_properties)
# plt.legend()

plt.subplot(1,3,2)
plt.yscale('log')
plt.plot(np.arange(1, num_epochs+1),SGD_Loss_train, label=r'SGD', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),SAM_Loss_train, label=r'SAM', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),CSAM0_Loss_train, label=r'SAMGCM', linewidth=2)
plt.xlabel('Epoch', labelpad=5)
plt.ylabel('Train loss', labelpad=5)

plt.subplot(1,3,3)
plt.plot(np.arange(1, num_epochs+1),SGD_L1, label=r'SGD', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),SAM_L1, label=r'SAM', linewidth=2)
plt.plot(np.arange(1, num_epochs+1),CSAM0_L1, label=r'SAMGCM', linewidth=2)
plt.xlabel('Epoch', labelpad=5)
plt.ylabel('L_1 norm', labelpad=5)

plt.tight_layout()
plt.show()

