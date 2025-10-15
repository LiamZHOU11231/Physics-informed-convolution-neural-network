# ========================================================================================================
# 功能：测试集回归分析。
# ========================================================================================================

import sys
import os

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as torchData

from tqdm import tqdm

# Caution 1/5  导入dataset数据集 ================================================================================
# -------------------------------------------------------------------------------------------------------------
from PFCandTcase1_update18_e2v3_func2 import DB_PFCandTcase1_update18_e2v3
from PFCandTcase1_update18_e2v3_func3_Kernel72 import MyConvNet
# -------------------------------------------------------------------------------------------------------------

# Caution 2/5  确认损失函数 =======================================================================================
# -------------------------------------------------------------------------------------------------------------
from PFCandTcase1_update18_e2v3_func_LossErrReal_e3 import LossErrReal_USE
# -------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

# Caution 3/5  指定测试集输出的保存路径 ==========================================================================
# -------------------------------------------------------------------------------------------------------------

    loc_test_out = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                    + '\OutputTest_ShuffleFalse_NETother4.xlsx'
    loc_test_outGT = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                    + '\OutputTestGT_ShuffleFalse.xlsx'

# -------------------------------------------------------------------------------------------------------------

# Caution 4/5  导入训练好的参数 ================================================================================
# -------------------------------------------------------------------------------------------------------------
    model_weight_path = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3' \
                        '\PFCandT_case1_update18_e2v3_weight_other_70dB' \
                        + '\MyConvNet_other4.pth'
# -------------------------------------------------------------------------------------------------------------

# Caution 5/5  测试集涉及的元素个数、网络参数指定 ==================================================================
# -------------------------------------------------------------------------------------------------------------
# 样本参数指定
    batchS_test     = 1
    input_height    = 1024
    input_width     = 5
    out_height      = 1
    out_width       = 4
# -------------------------------------------------------------------------------------------------------------



    # 设备指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 确认路径
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # 网络实例化
    net_USE = MyConvNet().to(device)
    net_USE.load_state_dict(torch.load(model_weight_path, map_location=device))



    # 加载测试集数据
    test_dataset = DB_PFCandTcase1_update18_e2v3(mode='test')
    test_data_loader = torchData.DataLoader(dataset=test_dataset,
                                            batch_size=batchS_test,
                                            shuffle=False)

    # 样本数目
    test_num = len(test_dataset)
    # 每一个epoch中的迭代次数
    test_iters = len(test_data_loader)

    outTEMP_stor_test   = torch.zeros(test_iters, batchS_test * out_height, out_width)
    stor_rst_test       = np.zeros(((test_iters, batchS_test * out_height, out_width)))
    outTEMP_stor_testGT = torch.zeros(test_iters, batchS_test * out_height, out_width)
    stor_rst_testGT     = np.zeros(((test_iters, batchS_test * out_height, out_width)))

    # ⭐️ ------------------------------------------------------------------------- ⭐️
    loss_function = nn.L1Loss()
    # ⭐️ ------------------------------------------------------------------------- ⭐️

    # 网络进入测试模式
    net_USE.eval()
    running_Crtn1_MAE_test = 0.0
    running_Crtn1_ErrReal_test = 0.0
    idx_test = 0
    with torch.no_grad():
        # with torch.no_grad() 的作用如下，
        # 在该模块下，所有计算得出的 tensor 的 requires_grad 都自动设置为False，
        # 这样就可以保证在验证阶段，不改变网络权重。
            test_bar = tqdm(test_data_loader, file=sys.stdout)
            for sample in test_bar:
                img_test = sample['img'].view(batchS_test, 1, input_height, input_width)
                gt_test  = sample['gt']

                # 验证集数据输入网络并计算出预测结果
                outputs_test = net_USE(img_test.to(device))
                # print('test_outputs: %.3f' % outputs)

                # 存储结果
                outTEMP_stor_test[idx_test, :, :]   = outputs_test
                outTEMP_stor_testGT[idx_test, :, :] = gt_test
                idx_test = idx_test + 1

                # 计算预测结果的评价指标
                Crtn1_MAE_test      = loss_function(outputs_test, gt_test.to(device))
                Crtn1_ErrReal_test  = LossErrReal_USE(batchS_test, outputs_test, gt_test.to(device))

                # 计算MAE、Err
                running_Crtn1_MAE_test      += Crtn1_MAE_test.item()
                running_Crtn1_ErrReal_test  += Crtn1_ErrReal_test.item()

    MAE_1_test          = running_Crtn1_MAE_test / test_iters
    ErrReal_1_test      = running_Crtn1_ErrReal_test / test_iters
    stor_rst_test       = (outTEMP_stor_test.detach().numpy())/1.0
    stor_rst_testGT     = (outTEMP_stor_testGT.detach().numpy())/1.0

    print('MAE_1_test: %.3f   ErrReal_1_test: %.3f   ErrReal_1_test_percent: %.3f%%' %
          (MAE_1_test, ErrReal_1_test, ErrReal_1_test * 100))
    print("----------------------")


    # 保存测试集不平衡量结果
    test_out_np = stor_rst_test.reshape(test_num, out_width)
    test_out_pd = pd.DataFrame(test_out_np)
    writer_USE  = pd.ExcelWriter(loc_test_out)
    test_out_pd.to_excel(writer_USE, 'sheet1', float_format='%.10f')
    writer_USE._save()
    writer_USE.close()

    # 保存测试集不平衡量结果-真值
    testGT_out_np = stor_rst_testGT.reshape(test_num, out_width)
    testGT_out_pd = pd.DataFrame(testGT_out_np)
    writerGT_USE  = pd.ExcelWriter(loc_test_outGT)
    testGT_out_pd.to_excel(writerGT_USE, 'sheet1', float_format='%.10f')
    writerGT_USE._save()
    writerGT_USE.close()



