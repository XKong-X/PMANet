import numpy as np
import torch

def calculate_mae(pred, true):
    # 计算平均绝对误差
    error = np.abs(pred - true)
    mae = np.mean(error)
    return mae

def calculate_mse(pred, true):
    # 计算均方误差
    error = (pred - true) ** 2
    mse = np.mean(error)
    return mse

def calculate_rmse(pred, true):
    # 计算均方根误差
    mse = calculate_mse(pred, true)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mape(pred, true):
    # 计算平均绝对百分比误差
    error = np.abs((pred - true) / true)
    mape = np.mean(error) * 100
    return mape
def calculate_mspe(pred, true):
    # 计算均方百分比误差
    error = ((pred - true) / true) ** 2
    mspe = np.mean(error)
    return mspe
def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def r(preds, trues):
    preds = np.squeeze(preds)
    trues = np.squeeze(trues)
    return np.corrcoef(preds, trues)[0, 1]

def metric(pred, true):
    # 计算指标
    mae = calculate_mae(pred, true)
    mse = calculate_mse(pred, true)
    rmse = calculate_rmse(pred, true)
    mape = calculate_mape(pred, true)
    mspe = calculate_mspe(pred, true)
    rse = RSE(pred, true)

    return mae, mse, rmse, mape, mspe, rse
