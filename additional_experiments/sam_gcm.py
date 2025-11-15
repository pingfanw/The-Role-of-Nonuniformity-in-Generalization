import numpy as np
from tqdm import tqdm
import torch

def sgd(SGD_w0, SGD_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs):
    num_samples = dataX.shape[0]
    num_iters = num_samples // b
    SGD_Loss_train, SGD_Loss_test, SGD_L1, SGD_L2 = [], [], [], []
    for epoch in tqdm(range(num_epochs), desc="SGD"):
        # indices = np.arange(num_samples)
        loss_test, loss_train,SGD_l1 ,SGD_l2 =0.0,0.0,0.0,0.0
        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, b):
            batch_indices = indices[i:i + b]
            X_batch = dataX[batch_indices]
            y_batch = dataY[batch_indices]
            predicted = X_batch @ (SGD_w0 * SGD_ww0)
            error = predicted - y_batch
            gradient_w = (X_batch.T @ error) * SGD_ww0 / b
            gradient_ww = (X_batch.T @ error) * SGD_w0 / b
            SGD_w0 = SGD_w0 - eta * gradient_w
            SGD_ww0 = SGD_ww0 - eta * gradient_ww
            loss_train += np.linalg.norm(dataX @ (SGD_w0 * SGD_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters
            loss_test += np.linalg.norm(PdataX @ (SGD_w0 * SGD_ww0) - PdataY) ** 2 / (2 * PdataX.shape[0])/num_iters
            SGD_l1 += np.linalg.norm((SGD_w0 * SGD_ww0), ord=1)/num_iters
            SGD_l2 += np.linalg.norm((SGD_w0 * SGD_ww0))/num_iters
        SGD_Loss_train.append(loss_train)
        SGD_Loss_test.append(loss_test)
        SGD_L1.append(SGD_l1)
        SGD_L2.append(SGD_l2)
    return SGD_Loss_train, SGD_Loss_test, SGD_L1

def sam(SAM_w0, SAM_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_sam, xi):
    num_samples = dataX.shape[0]
    num_iters = num_samples // b
    SAM_Loss_train, SAM_Loss_test, SAM_L1, SAM_L2, Metric_SAM = [], [], [], [], []
    for epoch in tqdm(range(num_epochs), desc="SAM"):
        # indices = np.arange(num_samples)
        indices = torch.randperm(num_samples)
        loss_test, loss_train, SGD_l1, SGD_l2, metric = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(0, num_samples, b):
            batch_indices = indices[i:i + b]
            X_batch = dataX[batch_indices]
            y_batch = dataY[batch_indices]
            predicted = X_batch @ (SAM_w0 * SAM_ww0)
            error = predicted - y_batch
            gradient_w = (X_batch.T @ error) * SAM_ww0 / b
            gradient_ww = (X_batch.T @ error) * SAM_w0 / b

            W_sam = SAM_w0 + rho_sam * gradient_w/(np.linalg.norm(np.concatenate((gradient_w, gradient_ww)))+xi)
            WW_sam = SAM_ww0 + rho_sam * gradient_ww / (np.linalg.norm(np.concatenate((gradient_w, gradient_ww)))+xi)

            metric = np.linalg.norm(dataX @ (W_sam * WW_sam) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters - np.linalg.norm(dataX @ (SAM_w0 * SAM_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters

            g_sam=(X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * WW_sam / b
            gg_sam = (X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * W_sam / b


            SAM_w0 = SAM_w0 - eta * g_sam
            SAM_ww0 = SAM_ww0 - eta * gg_sam

            loss_train += np.linalg.norm(dataX @ (SAM_w0 * SAM_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters
            loss_test += np.linalg.norm(PdataX @ (SAM_w0 * SAM_ww0) - PdataY) ** 2 / (2 * PdataX.shape[0])/num_iters
            SGD_l1 += np.linalg.norm((SAM_w0 * SAM_ww0), ord=1)/num_iters
            SGD_l2 += np.linalg.norm((SAM_w0 * SAM_ww0))/num_iters
        SAM_Loss_train.append(loss_train)
        SAM_Loss_test.append(loss_test)
        SAM_L1.append(SGD_l1)
        SAM_L2.append(SGD_l2)
        Metric_SAM.append(metric)
    return SAM_Loss_train, SAM_Loss_test, SAM_L1

def SAMGCM(CSAM0_w0, CSAM0_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_cov, xi, d):
    num_samples = dataX.shape[0]
    num_iters = num_samples//b
    CSAM0_Loss_train, CSAM0_Loss_test, CSAM0_L1, CSAM0_L2= [], [], [], []
    for epoch in tqdm(range(num_epochs), desc="SAMGCM"):
        indices = torch.randperm(num_samples)
        CSAM0_loss_train, CSAM0_loss_test, CSAM_l1, CSAM_l2 = 0.0, 0.0, 0.0, 0.0
        for i in range(0, num_samples, b):
            batch_indices = indices[i:i + b]
            X_batch = dataX[batch_indices]
            y_batch = dataY[batch_indices]
            predicted = X_batch @ (CSAM0_w0 * CSAM0_ww0)
            error = predicted - y_batch
            Fgradient_w = (X_batch.T @ error) * CSAM0_ww0 / b
            Fgradient_ww = (X_batch.T @ error) * CSAM0_w0 / b
            C_SAM = np.zeros((2 * d, 2 * d))

            # 计算协方差矩阵
            for i in range(b):
                gradient_w = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_ww0
                gradient_ww = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_w0
                C_SAM += np.outer(np.array([gradient_w, gradient_ww]), np.array([gradient_w, gradient_ww]))

            # 计算协方差矩阵的期望值
            Cov_SAM = ((1 / b) * C_SAM - np.outer(np.concatenate((Fgradient_w, Fgradient_ww)), np.concatenate((Fgradient_w, Fgradient_ww))) + xi * np.eye(2 * d))

            esplion = np.linalg.inv(Cov_SAM) @ np.concatenate((Fgradient_w, Fgradient_ww)) / np.sqrt(
                    np.concatenate((Fgradient_w, Fgradient_ww)).T @ np.linalg.inv(Cov_SAM) @ np.concatenate(
                        (Fgradient_w, Fgradient_ww)))
            CSAM0_w0 = CSAM0_w0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
                        CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
                                                               CSAM0_ww0 + rho_cov * esplion[d:2 * d])) / b
            CSAM0_ww0 = CSAM0_ww0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
                        CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
                                                                 CSAM0_w0 + rho_cov * esplion[:d])) / b
            CSAM0_loss_train += np.linalg.norm(dataX @ (CSAM0_w0 * CSAM0_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters
            CSAM0_loss_test += np.linalg.norm(PdataX @ (CSAM0_w0 * CSAM0_ww0) - PdataY) ** 2 / (2 * PdataX.shape[0])/num_iters
            CSAM_l1+= np.linalg.norm((CSAM0_w0 * CSAM0_ww0), ord=1)/num_iters
            CSAM_l2 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0))/num_iters
        CSAM0_Loss_train.append(CSAM0_loss_train)
        CSAM0_Loss_test.append(CSAM0_loss_test)
        CSAM0_L1.append(CSAM_l1)
        CSAM0_L2.append(CSAM_l2)
    return CSAM0_Loss_train, CSAM0_Loss_test, CSAM0_L1

















# def CSAM(CSAM0_w0, CSAM0_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_t, rho_cov, xi, d, a):
#     num_samples = dataX.shape[0]
#     num_iters = num_samples // b
#     CSAM_Loss_train, CSAM_Loss_test, CSAM_L1, CSAM_L2, Metric_CSAM= [], [], [], [], []
#     for epoch in tqdm(range(num_epochs), desc="CSAM"):
#         indices = torch.randperm(num_samples)
#         CSAM0_loss_train, CSAM0_loss_test, CSAM_l1, CSAM_l2, m_t, metric = 0.0, 0.0, 0.0, 0.0, np.zeros(2*d).flatten(), 0.0
#         for i in range(0, num_samples, b):
#             batch_indices = indices[i:i + b]
#             X_batch = dataX[batch_indices]
#             y_batch = dataY[batch_indices]
#             predicted = X_batch @ (CSAM0_w0 * CSAM0_ww0)
#             error = predicted - y_batch
#             g_1 = (X_batch.T @ error) * CSAM0_ww0 / b
#             gg_1 = (X_batch.T @ error) * CSAM0_w0 / b
#             W_sam = CSAM0_w0 + rho_t * g_1 / (np.linalg.norm(np.concatenate((g_1, gg_1))) + xi)
#             WW_sam = CSAM0_ww0 + rho_t * gg_1 / (np.linalg.norm(np.concatenate((g_1, gg_1))) + xi)
#
#             g_2 = (X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * WW_sam / b
#             gg_2 = (X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * W_sam / b
#
#
#             m_t= a*m_t+(1-a)*(np.concatenate((g_1, gg_1))-np.concatenate((g_2, gg_2)))
#
#             gradient = np.concatenate((g_2, gg_2))-np.concatenate((g_1, gg_1))-m_t
#             C_SAM = np.zeros((2 * d, 2 * d))
#
#             # 计算协方差矩阵
#             for i in range(b):
#                 gradient_w = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_ww0
#                 gradient_ww = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_w0
#                 C_SAM += np.outer(np.array([gradient_w, gradient_ww]), np.array([gradient_w, gradient_ww]))
#
#             # 计算协方差矩阵的期望值
#             Cov_SAM = ((1 / b) * C_SAM - np.outer(np.concatenate((g_1, gg_1)),np.concatenate((g_1, gg_1))) + xi * np.eye(2 * d))
#
#             esplion = np.linalg.inv(Cov_SAM) @ gradient.T / np.sqrt(gradient @ np.linalg.inv(Cov_SAM) @ gradient.T)
#
#             metric = np.abs(np.linalg.norm(np.concatenate(((X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                  CSAM0_ww0 + rho_cov * esplion[d:2 * d])) / b, (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                    CSAM0_w0 + rho_cov * esplion[:d])) / b))) - np.linalg.norm(np.concatenate(((dataX.T @ ((dataX @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - dataY)) * (
#                                                  CSAM0_ww0 + rho_cov * esplion[d:2 * d])) / b, (dataX.T @ ((dataX @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - dataY)) * (
#                                                    CSAM0_w0 + rho_cov * esplion[:d])) / b))))
#
#             CSAM0_w0 = CSAM0_w0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                  CSAM0_ww0 + rho_cov * esplion[d:2 * d])) / b
#             CSAM0_ww0 = CSAM0_ww0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                     CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                    CSAM0_w0 + rho_cov * esplion[:d])) / b
#
#             CSAM0_loss_train += np.linalg.norm(dataX @ (CSAM0_w0 * CSAM0_ww0) - dataY) ** 2 / (
#                         2 * dataX.shape[0]) / num_iters
#             CSAM0_loss_test += np.linalg.norm(PdataX @ (CSAM0_w0 * CSAM0_ww0) - PdataY) ** 2 / (
#                         2 * PdataX.shape[0]) / num_iters
#             CSAM_l1 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0), ord=1) / num_iters
#             CSAM_l2 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0)) / num_iters
#         CSAM_Loss_train.append(CSAM0_loss_train)
#         CSAM_Loss_test.append(CSAM0_loss_test)
#         CSAM_L1.append(CSAM_l1)
#         CSAM_L2.append(CSAM_l2)
#         Metric_CSAM.append(metric)
#     return CSAM_Loss_train, CSAM_Loss_test, CSAM_L1, CSAM_L2, Metric_CSAM
























# def Fcsam(CSAM0_w0, CSAM0_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_cov, xi, d):
#     num_samples = dataX.shape[0]
#     num_iters = num_samples//b
#     FCSAM0_Loss_train, FCSAM0_Loss_test, FCSAM0_L1, FCSAM0_L2= [], [], [], []
#     for epoch in tqdm(range(num_epochs), desc="FSAM"):
#         indices = torch.randperm(num_samples)
#         CSAM0_loss_train, CSAM0_loss_test, CSAM_l1, CSAM_l2 = 0.0, 0.0, 0.0, 0.0
#         for i in range(0, num_samples, b):
#             batch_indices = indices[i:i + b]
#             X_batch = dataX[batch_indices]
#             y_batch = dataY[batch_indices]
#             predicted = X_batch @ (CSAM0_w0 * CSAM0_ww0)
#             error = predicted - y_batch
#             Fgradient_w = (X_batch.T @ error) * CSAM0_ww0 / b
#             Fgradient_ww = (X_batch.T @ error) * CSAM0_w0 / b
#             C_SAM , bg = np.zeros((2 * d, 2 * d)), np.zeros((2 * d)).flatten()
#
#
#
#
#
#             # 计算协方差矩阵
#             for i in range(b):
#                 gradient_w = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_ww0
#                 gradient_ww = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_w0
#                 C_SAM += np.outer(np.array([gradient_w, gradient_ww]), np.array([gradient_w, gradient_ww]))
#                 bg += np.array([gradient_w, gradient_ww]).flatten()/b
#
#             # 计算协方差矩阵的期望值
#             Cov_SAM = ((1 / b) * C_SAM - np.outer(np.concatenate((Fgradient_w, Fgradient_ww)), np.concatenate((Fgradient_w, Fgradient_ww))) + xi * np.eye(2 * d))
#             ng = np.array([Fgradient_w, Fgradient_ww]).flatten()
#
#             esplion = 0.5*np.linalg.inv(Cov_SAM) @ (bg) / np.sqrt((bg).T @ np.linalg.inv(Cov_SAM) @ (bg)) + 0.5*np.array([Fgradient_w, Fgradient_ww]).flatten()/ np.linalg.norm(np.array([Fgradient_w, Fgradient_ww]).flatten())
#
#             CSAM0_w0 = CSAM0_w0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                         CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                                CSAM0_ww0 + rho_cov * esplion[d:2 * d])) / b
#             CSAM0_ww0 = CSAM0_ww0 - eta * (X_batch.T @ ((X_batch @ ((CSAM0_w0 + rho_cov * esplion[:d]) * (
#                         CSAM0_ww0 + rho_cov * esplion[d:2 * d])) - y_batch)) * (
#                                                                  CSAM0_w0 + rho_cov * esplion[:d])) / b
#             CSAM0_loss_train += np.linalg.norm(dataX @ (CSAM0_w0 * CSAM0_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/num_iters
#             CSAM0_loss_test += np.linalg.norm(PdataX @ (CSAM0_w0 * CSAM0_ww0) - PdataY) ** 2 / (2 * PdataX.shape[0])/num_iters
#             CSAM_l1 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0), ord=1)/num_iters
#             CSAM_l2 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0))/num_iters
#         FCSAM0_Loss_train.append(CSAM0_loss_train)
#         FCSAM0_Loss_test.append(CSAM0_loss_test)
#         FCSAM0_L1.append(CSAM_l1)
#         FCSAM0_L2.append(CSAM_l2)
#     return FCSAM0_Loss_train, FCSAM0_Loss_test, FCSAM0_L1, FCSAM0_L2

# def mcsam(CSAM0_w0, CSAM0_ww0, dataX, dataY, PdataX, PdataY, eta, b, num_epochs, rho_sam, rho_cov_m, xi, m_batch, d, a):
#     num_samples = dataX.shape[0]
#     num_iters = num_samples // b
#     m_CSAM0_Loss_train, m_CSAM0_Loss_test, m_CSAM0_L1, m_CSAM0_L2= [], [], [], []
#     for epoch in tqdm(range(num_epochs), desc="mCSAM"):
#         #indices = np.arange(num_samples)
#         indices=torch.randperm(num_samples)
#         m_CSAM0_loss_train, m_CSAM0_loss_test = 0.0, 0.0
#         m_CSAM_l1, m_CSAM_l2 = 0.0, 0.0
#         for i in range(0, num_samples, b):
#             m_g = []
#             batch_indices = indices[i:i + b]
#             X_batch = dataX[batch_indices]
#             y_batch = dataY[batch_indices]
#             predicted = X_batch @ (CSAM0_w0 * CSAM0_ww0)
#             error = predicted - y_batch
#             Fgradient_w = (X_batch.T @ error) * CSAM0_ww0 / b
#             Fgradient_ww = (X_batch.T @ error) * CSAM0_w0 / b
#             W_sam = CSAM0_w0 + rho_sam * Fgradient_w / (np.linalg.norm(np.concatenate((Fgradient_w, Fgradient_ww))) + xi)
#             WW_sam = CSAM0_ww0 + rho_sam * Fgradient_ww / (np.linalg.norm(np.concatenate((Fgradient_w, Fgradient_ww))) + xi)
#
#             g = (X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * WW_sam / b
#             gg = (X_batch.T @ (X_batch @ (W_sam * WW_sam) - y_batch)) * W_sam / b
#             C_SAM = np.zeros((2 * d, 2 * d))
#
#             # 计算协方差矩阵
#             for i in range(b):
#                 gradient_w = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_ww0
#                 gradient_ww = (X_batch[i, :].T * (X_batch[i, :] @ (CSAM0_w0 * CSAM0_ww0) - y_batch[i])) * CSAM0_w0
#                 C_SAM += np.outer(np.array([gradient_w, gradient_ww]), np.array([gradient_w, gradient_ww]))
#
#             # 计算协方差矩阵的期望值
#             Cov_SAM = ((1 / b) * C_SAM - np.outer(np.concatenate((Fgradient_w, Fgradient_ww)), np.concatenate((Fgradient_w, Fgradient_ww)))  + xi * np.eye(2 * d))
#             g_sam, gg_sam = np.zeros((d, 1)).flatten(), np.zeros((d, 1)).flatten()
#             for z in range(0, b, m_batch):
#                 predicted = X_batch[z:z + m_batch] @ (CSAM0_w0 * CSAM0_ww0)
#                 error = predicted - y_batch[z:z + m_batch]
#                 m_gradient_w = (X_batch[z:z + m_batch].T @ error) * CSAM0_ww0 / m_batch
#                 m_gradient_ww = (X_batch[z:z + m_batch].T @ error) * CSAM0_w0 / m_batch
#                 W_sam = CSAM0_w0 + rho_sam * m_gradient_w / (np.linalg.norm(np.concatenate((m_gradient_w, m_gradient_ww))) + xi)
#                 WW_sam = CSAM0_ww0 + rho_sam * m_gradient_ww / (np.linalg.norm(np.concatenate((m_gradient_w, m_gradient_ww))) + xi)
#                 g1 = (X_batch[z:z + m_batch].T @ (X_batch[z:z + m_batch] @ (W_sam * WW_sam) - y_batch[z:z + m_batch]))* WW_sam / m_batch
#                 g2 = (X_batch[z:z + m_batch].T @ (X_batch[z:z + m_batch] @ (W_sam * WW_sam) - y_batch[z:z + m_batch])) * W_sam / m_batch
#                 g_sam = g_sam + g1
#                 gg_sam = gg_sam + g2
#                 m_g.append(np.concatenate((g1,g2)))
#             V_g = np.concatenate((g_sam / (b / m_batch), gg_sam / (b / m_batch)))
#             co = (np.array(m_g) @ V_g/np.linalg.norm(np.array(m_g) @ V_g))/np.linalg.norm(np.array(m_g) @ V_g/np.linalg.norm(np.array(m_g) @ V_g))
#             g_finally=np.zeros((2 * d, 1)).flatten()
#             for i in range(int(b / m_batch)):
#                 g_finally = g_finally + co[i] * np.array(m_g[i])
#
#             gsam = a * g_finally[:d] / (b / m_batch)+ (1-a) * g
#             ggsam = a * g_finally[d:2*d] / (b / m_batch) + (1-a)* gg
#
#             esplion = np.linalg.inv(Cov_SAM) @ np.concatenate((gsam, ggsam)) / np.sqrt(np.concatenate((gsam, ggsam)).T @ np.linalg.inv(Cov_SAM) @ np.concatenate((gsam, ggsam)))
#             CSAM0_w0 = CSAM0_w0 - eta * (X_batch[z:z + m_batch].T @ ((X_batch[z:z + m_batch] @ ((CSAM0_w0 + rho_cov_m * esplion[:d]) * (
#                         CSAM0_ww0 + rho_cov_m * esplion[d:2 * d])) - y_batch[z:z + m_batch])) * (CSAM0_ww0 + rho_cov_m * esplion[d:2 * d])) / b
#             CSAM0_ww0 = CSAM0_ww0 - eta * (X_batch[z:z + m_batch].T @ ((X_batch[z:z + m_batch] @ ((CSAM0_w0 + rho_cov_m * esplion[:d]) * (
#                         CSAM0_ww0 + rho_cov_m * esplion[d:2 * d])) - y_batch[z:z + m_batch])) * (CSAM0_w0 + rho_cov_m * esplion[:d])) / b
#             m_CSAM0_loss_train += np.linalg.norm(dataX @ (CSAM0_w0 * CSAM0_ww0) - dataY) ** 2 / (2 * dataX.shape[0])/ num_iters
#             m_CSAM0_loss_test += np.linalg.norm(PdataX @ (CSAM0_w0 * CSAM0_ww0) - PdataY) ** 2 / (2 * PdataX.shape[0])/ num_iters
#             m_CSAM_l1 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0), ord=1)/ num_iters
#             m_CSAM_l2 += np.linalg.norm((CSAM0_w0 * CSAM0_ww0))/ num_iters
#         m_CSAM0_Loss_train.append(np.array(m_CSAM0_loss_train))
#         m_CSAM0_Loss_test.append(m_CSAM0_loss_test)
#         m_CSAM0_L1.append(m_CSAM_l1)
#         m_CSAM0_L2.append(m_CSAM_l2)
#     return m_CSAM0_Loss_train, m_CSAM0_Loss_test, m_CSAM0_L1, m_CSAM0_L2