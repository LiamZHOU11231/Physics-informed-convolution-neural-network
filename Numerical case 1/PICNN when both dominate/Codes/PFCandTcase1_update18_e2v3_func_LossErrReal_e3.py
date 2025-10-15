# ========================================================================================================
# 功能：用相对误差来计算损失，即MAE再除以真值。
# _e3：在e2基础上，把剔除0真值加上。
# ========================================================================================================
# import pandas as pd
import torch
# import numpy as np
import math
# import cmath

pi = math.pi

def LossErrReal_USE(batchS, outputNN, outputGT):
    #  传进来的outputNN、outputGT，类型都是tensor，维度都是40*2，40是批大小，2是每个样本的输出维度是1*2。
    # ⭐️ --------------------------------------------------------------- ⭐️
    # ⭐️ --------------------------------------------------------------- ⭐️
    # Caution(1/3)  “0真值”的标准。=========================================
    #  所谓“0真值”的判定标准：小于 myJudgeZero 的就认为是“0真值”。
    myJudgeZero = 1e-4

    # Caution(2/3)  确认归一化区间 =========================================
    #  确认归一化的情况 ！！！
    NormMin = 1e-3
    NormMax = 1

    # Caution(3/3)  确认原数据的最大最小值 ==================================
    WithDim_output1_max = 374.531747812539
    WithDim_output1_min = 67.1133099274076
    WithDim_output2_max = 161.460000000000
    WithDim_output2_min = -162
    WithDim_output3_max = 374.531747812539
    WithDim_output3_min = 67.1133099274076
    WithDim_output4_max = 161.460000000000
    WithDim_output4_min = -162
    # ⭐️ --------------------------------------------------------------- ⭐️
    # ⭐️ --------------------------------------------------------------- ⭐️
    factor_NN_0 = (WithDim_output1_max - WithDim_output1_min) / (NormMax - NormMin)
    factor_NN_1 = (WithDim_output2_max - WithDim_output2_min) / (NormMax - NormMin)
    factor_NN_2 = (WithDim_output3_max - WithDim_output3_min) / (NormMax - NormMin)
    factor_NN_3 = (WithDim_output4_max - WithDim_output4_min) / (NormMax - NormMin)


    # STEP 1/3  进行反归一化
    outputNN_WithDim = torch.zeros(batchS, 4)
    outputGT_WithDim = torch.zeros(batchS, 4)

    outputNN_WithDim[:, 0] = (outputNN[:, 0] - NormMin)*factor_NN_0 + WithDim_output1_min
    outputNN_WithDim[:, 1] = (outputNN[:, 1] - NormMin)*factor_NN_1 + WithDim_output2_min
    outputNN_WithDim[:, 2] = (outputNN[:, 2] - NormMin)*factor_NN_2 + WithDim_output3_min
    outputNN_WithDim[:, 3] = (outputNN[:, 3] - NormMin)*factor_NN_3 + WithDim_output4_min

    outputGT_WithDim[:, 0] = (outputGT[:, 0] - NormMin)*factor_NN_0 + WithDim_output1_min
    outputGT_WithDim[:, 1] = (outputGT[:, 1] - NormMin)*factor_NN_1 + WithDim_output2_min
    outputGT_WithDim[:, 2] = (outputGT[:, 2] - NormMin)*factor_NN_2 + WithDim_output3_min
    outputGT_WithDim[:, 3] = (outputGT[:, 3] - NormMin)*factor_NN_3 + WithDim_output4_min


    # STEP 2/3  剔除0真值
    outputGT_WithDim_flat   = outputGT_WithDim.view(-1)
    MyCond_GT               = torch.nonzero(abs(outputGT_WithDim_flat)>myJudgeZero)
    MyCond_GT_flat          = MyCond_GT.view(-1)
    outputGT_After          = torch.index_select(outputGT_WithDim_flat, dim=0, index=MyCond_GT_flat)

    outputNN_WithDim_flat   = outputNN_WithDim.view(-1)
    outputNN_After          = torch.index_select(outputNN_WithDim_flat, dim=0, index=MyCond_GT_flat)


    # STEP 3/3  计算相对误差
    MAE_total       = abs(outputNN_After - outputGT_After)
    Err_total       = abs(torch.div(MAE_total, outputGT_After))
    NumElem         = Err_total.numel()
    Err_total_sum   = Err_total.sum()
    Err_FINAL       = Err_total_sum / NumElem


    return Err_FINAL







