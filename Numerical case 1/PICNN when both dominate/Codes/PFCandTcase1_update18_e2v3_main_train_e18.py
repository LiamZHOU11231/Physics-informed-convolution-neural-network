# ========================================================================================================
# 功能：回归分析，训练模式。
# ========================================================================================================
import sys

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchData
import numpy as np
import random
import pandas as pd

from tqdm import tqdm

# Caution 1/12  导入dataset数据集 ================================================================================
# -------------------------------------------------------------------------------------------------------------
#           一定记得改路径！！！
from PFCandTcase1_update18_e2v3_func2 import DB_PFCandTcase1_update18_e2v3
# -------------------------------------------------------------------------------------------------------------

# Caution 2/12  确认网络 ==========================================================================================
# -------------------------------------------------------------------------------------------------------------
from PFCandTcase1_update18_e2v3_func3_Kernel72 import MyConvNet
# -------------------------------------------------------------------------------------------------------------

# Caution 3/12  确认损失函数 =======================================================================================
# -------------------------------------------------------------------------------------------------------------
from PFCandTcase1_update18_e2v3_func_LossErrReal_e3 import LossErrReal_USE

# 五角星Attention (1/3)
# ⭐️ --------------------------------------------------------------- ⭐️
#  确认归一化的情况 ！！！
# ⭐️ --------------------------------------------------------------- ⭐️
# -------------------------------------------------------------------------------------------------------------


from torch.utils.tensorboard import SummaryWriter


# Caution 4/12  指定最后 训练集 输出的保存路径 ======================================================================
# -------------------------------------------------------------------------------------------------------------
loc_train_2nd_out = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                    + '\OutputTrain2nd_ShuffleTure_BestEpoch.xlsx'
loc_train_2nd_outGT = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                      + '\OutputTrain2ndGT_ShuffleTure.xlsx'
# -------------------------------------------------------------------------------------------------------------

# Caution 5/12  指定最后 训练集 损失、迭代评价指标的保存路径 ===========================================================
# -------------------------------------------------------------------------------------------------------------
loc_train_loss1_MAE     = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                              + '\OutputTrain_loss1_MAE.xlsx'
loc_train_2nd_Crtn1_MAE = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                              + '\OutputTrain2nd_Crtn1_MAE.xlsx'
loc_train_2nd_Crtn1_ErrReal = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                              + '\OutputTrain2nd_Crtn1_ErrReal.xlsx'
# -------------------------------------------------------------------------------------------------------------

# Caution 6/12  指定最后 验证集 输出不平衡量结果的保存路径 ============================================================
# -------------------------------------------------------------------------------------------------------------
loc_val_out = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
               + '\OutputVal_ShuffleTure_BestEpoch.xlsx'
# 由于 shuffle 的原因，真值也需要保存。
loc_val_outGT = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
               + '\OutputValGT_ShuffleTure.xlsx'
# -------------------------------------------------------------------------------------------------------------

# Caution 7/12  指定最后 验证集 迭代评价指标的保存路径 ==============================================================
# -------------------------------------------------------------------------------------------------------------
loc_val_Crtn1_MAE = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                    + '\OutputVal_Crtn1_MAE.xlsx'
loc_val_Crtn1_ErrReal = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_result_70dB' \
                        + '\OutputVal_Crtn1_ErrReal.xlsx'
# -------------------------------------------------------------------------------------------------------------

# Caution 8/12  指定最后 网络权重 的保存路径 ========================================================================
# -------------------------------------------------------------------------------------------------------------
save_path = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\PFCandT_case1_update18_e2v3_weight_70dB' \
            + '\MyConvNet.pth'
# -------------------------------------------------------------------------------------------------------------

# Caution 9/12  指定最后 结果图 的保存路径 ==========================================================================
# -------------------------------------------------------------------------------------------------------------
writer_path = "PFCandT_case1_update18_e2v3_result_70dB"
# -------------------------------------------------------------------------------------------------------------
writer = SummaryWriter(writer_path)
# tensorboard --logdir=D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1\case1_3_result
# 换成绝对路径更稳妥！

# Caution 10/12  验证集涉及的元素个数、训练参数指定 =================================================================
# -------------------------------------------------------------------------------------------------------------
batchS_train    = 40
batchS_val      = 1

epochs_USE      = 2000
LR_USE_array    = 1e-2

input_height    = 1024
input_width     = 5
out_height      = 1
out_width       = 4

# 五角星Attention (2/3)
# ⭐️ ----------------------------------------------------- ⭐️
save_EveryEpoch = 10    # 声明每多少轮保存一下结果。
# ⭐️ ----------------------------------------------------- ⭐️
# -------------------------------------------------------------------------------------------------------------


# Caution 11/12  指定 其他网络权重 的保存路径 ========================================================================
# -------------------------------------------------------------------------------------------------------------
WeightOtherPathPrefix = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3' \
                        + '\PFCandT_case1_update18_e2v3_weight_other_70dB\\'

# -------------------------------------------------------------------------------------------------------------

# Caution 12/12  指定 smaller 结果的保存路径 =======================================================================
# -------------------------------------------------------------------------------------------------------------
loc_train_2nd_SmallerCrtn1  = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3' \
                              + '\PFCandT_case1_update18_e2v3_result_70dB' \
                              + '\OutputTrain2nd_SmallerCrtn1_Two.xlsx'
loc_val_SmallerCrtn1    = 'D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3' \
                          + '\PFCandT_case1_update18_e2v3_result_70dB' \
                          + '\OutputVal_SmallerCrtn1_Two.xlsx'
# -------------------------------------------------------------------------------------------------------------




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # 训练设备指定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 设置一个玄学种子
    set_seed(300)

    # 加载训练集
    train_dataset = DB_PFCandTcase1_update18_e2v3(mode='train')
    train_data_loader = torchData.DataLoader(dataset=train_dataset,
                                             batch_size=batchS_train,
                                             shuffle=True)

    # 加载验证集
    val_dataset = DB_PFCandTcase1_update18_e2v3(mode='val')
    val_data_loader = torchData.DataLoader(dataset=val_dataset,
                                           batch_size=batchS_val,
                                           shuffle=True)

    # 每个 epoch 中实际迭代的次数。
    train_iters = len(train_data_loader)
    val_iters = len(val_data_loader)

    # 训练、验证、测试的样本数量
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # 实例化网络模型
    net = MyConvNet().to(device)

    # 优化器
    params = [p for p in net.parameters() if p.requires_grad]



    # 损失函数。换成MAE形式。
    loss_function = nn.L1Loss()
    # 默认会对一个 Batch 的所有样本计算损失，并求均值

    best_epoch = 0

    best_epoch_count = 0
    idx_weight = np.arange(10) + 1
    num_repeat = int(epochs_USE/10)
    idx_weight_extend = np.tile(idx_weight, num_repeat)
    ForWeight_name1 = "MyConvNet_other"
    ForWeight_name3 = ".pth"
    # 存储经过比较之后，较小的那个loss以及对应的轮次。
    # 这里存储轮次的时候，是按最小值为1这个标准，即代码里的epoch 0 存储时候是 epoch 1。
    stor_Crtn1_small_val        = torch.zeros(epochs_USE, 3)
    # 对应的也存一下train的相应结果。
    stor_Crtn1_small_train_2nd  = torch.zeros(epochs_USE, 3)


    stor_loss1_MAE_train = torch.zeros(epochs_USE)


    best_ErrReal_1_val  = 10.0
    best_MAE_1_val      = 10.0
    stor_Crtn1_MAE_val      = torch.zeros(epochs_USE)
    stor_Crtn1_ErrReal_val  = torch.zeros(epochs_USE)
    outTEMP_stor_val    = torch.zeros(val_iters, batchS_val * out_height, out_width)
    stor_rst_val        = np.zeros(((val_iters, batchS_val * out_height, out_width)))
    best_ValOut         = np.zeros(((val_iters, batchS_val * out_height, out_width)))
    outTEMP_stor_valGT  = torch.zeros(val_iters, batchS_val * out_height, out_width)
    stor_rst_valGT      = np.zeros(((val_iters, batchS_val * out_height, out_width)))
    best_ValOutGT       = np.zeros(((val_iters, batchS_val * out_height, out_width)))

    best_ErrReal_1_train_2nd    = 10.0
    best_MAE_1_train_2nd        = 10.0
    stor_Crtn1_MAE_train_2nd        = torch.zeros(epochs_USE)
    stor_Crtn1_ErrReal_train_2nd    = torch.zeros(epochs_USE)
    outTEMP_stor_train_2nd          = torch.zeros(train_iters, batchS_train * out_height, out_width)
    stor_rst_train_2nd              = np.zeros(((train_iters, batchS_train * out_height, out_width)))
    best_TrainOut_2nd               = np.zeros(((train_iters, batchS_train * out_height, out_width)))
    outTEMP_stor_train_2nd_GT       = torch.zeros(train_iters, batchS_train * out_height, out_width)
    stor_rst_train_2nd_GT           = np.zeros(((train_iters, batchS_train * out_height, out_width)))
    best_TrainOut_2nd_GT             = np.zeros(((train_iters, batchS_train * out_height, out_width)))


    for epoch in range(epochs_USE):

        # 五角星Attention (3/3)
        # ⭐️ ------------------------------------------------------------------------- ⭐️
        # 学习率策略给定。
        # ⭐️ ------------------------------------------------------------------------- ⭐️
        # ----------------------------------------------------------------------------------------------
        LR_NOW      = LR_USE_array
        optimizer   = optim.Adam(params, lr=float(LR_NOW))
        # ----------------------------------------------------------------------------------------------


        # -------  训练  --------
        net.train()         # 网络进入训练模式
        train_bar = tqdm(train_data_loader, file=sys.stdout)
        # tqdm 创建进度条
        # stdout 参数直接显示在控制台窗口上
        running_loss1_MAE_train = 0.0
        for step, sample_train in enumerate(train_bar):

            img_train = (sample_train['img']).view(batchS_train, 1, input_height, input_width)
            gt_train  = sample_train['gt']

            # 数据输入网络并计算出预测结果
            outputs_train                           = net(img_train.to(device))

            # 计算预测结果的损失
            loss1_MAE_train     = loss_function(outputs_train, gt_train.to(device))
            loss1_ErrReal_train = LossErrReal_USE(batchS_train, outputs_train, gt_train.to(device))

            # 记录训练过程的损失
            running_loss1_MAE_train += loss1_MAE_train.item()

            # 三件套
            loss1_MAE_train.backward()  # 1、模型的反向传播。
            optimizer.step()  # 2、更新参数。
            optimizer.zero_grad()  # 3、清空计算图。

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs_USE, loss1_MAE_train)

        MAE_1_train = running_loss1_MAE_train / train_iters
        stor_loss1_MAE_train[epoch] = MAE_1_train




        # -------  验证 --------
        net.eval()
        running_Crtn1_MAE_val       = 0.0
        running_Crtn1_ErrReal_val   = 0.0
        idx_val = 0
        idx_train_2nd = 0
        running_Crtn1_MAE_train_2nd       = 0.0
        running_Crtn1_ErrReal_train_2nd   = 0.0
        with torch.no_grad():
        # with torch.no_grad() 的作用如下，
        # 在该模块下，所有计算得出的 tensor 的 requires_grad 都自动设置为False，
        # 这样就可以保证在验证阶段，不改变网络权重。
            val_bar = tqdm(val_data_loader, file=sys.stdout)
            for sample_val in val_bar:
                img_val = sample_val['img'].view(batchS_val, 1, input_height, input_width)
                gt_val  = sample_val['gt']

                # 验证集数据输入网络并计算出预测结果
                outputs_val                         = net(img_val.to(device))
                outTEMP_stor_val[idx_val, :, :]     = outputs_val
                outTEMP_stor_valGT[idx_val, :, :]   = gt_val
                idx_val = idx_val + 1

                # 计算预测结果的评价指标
                Crtn1_MAE_val       = loss_function(outputs_val, gt_val.to(device))
                Crtn1_ErrReal_val   = LossErrReal_USE(batchS_val, outputs_val, gt_val.to(device))

                # 计算MAE、Err
                running_Crtn1_MAE_val       += Crtn1_MAE_val.item()
                running_Crtn1_ErrReal_val   += Crtn1_ErrReal_val.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs_USE)

            MAE_1_val                       = running_Crtn1_MAE_val / val_iters
            ErrReal_1_val                   = running_Crtn1_ErrReal_val / val_iters
            stor_Crtn1_MAE_val[epoch]       = MAE_1_val
            stor_Crtn1_ErrReal_val[epoch]   = ErrReal_1_val
            stor_rst_val                    = (outTEMP_stor_val.detach().numpy()) / 1.0
            stor_rst_valGT                  = (outTEMP_stor_valGT.detach().numpy()) / 1.0

            # 补充train的评估结果
            train_bar_2nd = tqdm(train_data_loader, file=sys.stdout)
            for sample_train_2nd in train_bar_2nd:
                img_train_2nd = sample_train_2nd['img'].view(batchS_train, 1, input_height, input_width)
                gt_train_2nd  = sample_train_2nd['gt']

                # 验证集数据输入网络并计算出预测结果
                outputs_train_2nd                               = net(img_train_2nd.to(device))
                outTEMP_stor_train_2nd[idx_train_2nd, :, :]     = outputs_train_2nd
                outTEMP_stor_train_2nd_GT[idx_train_2nd, :, :]  = gt_train_2nd
                idx_train_2nd = idx_train_2nd + 1

                # 计算预测结果的损失
                Crtn1_MAE_train_2nd       = loss_function(outputs_train_2nd, gt_train_2nd.to(device))
                Crtn1_ErrReal_train_2nd   = LossErrReal_USE(batchS_train, outputs_train_2nd, gt_train_2nd.to(device))

                # 计算MAE、Err
                running_Crtn1_MAE_train_2nd       += Crtn1_MAE_train_2nd.item()
                running_Crtn1_ErrReal_train_2nd   += Crtn1_ErrReal_train_2nd.item()

                train_bar_2nd.desc = "valid train_2nd epoch[{}/{}]".format(epoch + 1, epochs_USE)

            MAE_1_train_2nd                       = running_Crtn1_MAE_train_2nd / train_iters
            ErrReal_1_train_2nd                   = running_Crtn1_ErrReal_train_2nd / train_iters
            stor_Crtn1_MAE_train_2nd[epoch]       = MAE_1_train_2nd
            stor_Crtn1_ErrReal_train_2nd[epoch]   = ErrReal_1_train_2nd
            stor_rst_train_2nd                    = (outTEMP_stor_train_2nd.detach().numpy()) / 1.0
            stor_rst_train_2nd_GT                 = (outTEMP_stor_train_2nd_GT.detach().numpy()) / 1.0


       # 打印该epoch训练结果
        print('LR_NOW = %f' % LR_NOW)
        print("----------------------")

        print('[epoch %d]   MAE_1_train_2nd: %.3f   ErrReal_1_train_2nd: %.3f   ErrReal_1_train_2nd_percent: %.3f%%' %
              (epoch + 1, MAE_1_train_2nd, ErrReal_1_train_2nd, ErrReal_1_train_2nd * 100))
        print("----------------------")

        print('[epoch %d]   MAE_1_val: %.3f   ErrReal_1_val: %.3f   ErrReal_1_val_percent: %.3f%%' %
              (epoch + 1, MAE_1_val, ErrReal_1_val, ErrReal_1_val * 100))
        print("----------------------")

        # 设置观测变量
        writer.add_scalar('MAE_1_train_2nd', MAE_1_train_2nd, global_step=epoch)
        writer.add_scalar('MAE_1_val', MAE_1_val, global_step=epoch)

        # 找到MAE更低的权重则保存，
        # 这里仍按 MAE_1_val，即仅关注最后 outputs_val 与 gt_val 的符合程度。
        if ErrReal_1_val < best_ErrReal_1_val:

            print("⭐️⭐️⭐️----------------------⭐️⭐️⭐️")
            print('[epoch %d]' % (epoch + 1))
            print("⭐️⭐️⭐️----------------------⭐️⭐️⭐️")

            # part1/4: best epoch 相关内容。
            # ---------------------------------------------------------------------------------
            best_epoch = epoch + 1
            # ---------------------------------------------------------------------------------

            # part2/4: train 相关内容。
            # ---------------------------------------------------------------------------------
            best_MAE_1_train_2nd = MAE_1_train_2nd
            #  ★★★★★  注意这里一定要加一个运算“/1.0”，为的是避免拷贝前后二者关联。
            best_TrainOut_2nd           = stor_rst_train_2nd / 1.0
            best_TrainOut_2nd_GT        = stor_rst_train_2nd_GT / 1.0
            best_ErrReal_1_train_2nd    = ErrReal_1_train_2nd
            # ---------------------------------------------------------------------------------

            # part3/4: val 相关内容。
            # ---------------------------------------------------------------------------------
            best_MAE_1_val = MAE_1_val
            #  ★★★★★  注意这里一定要加一个运算“/1.0”，为的是避免拷贝前后二者关联。
            best_ValOut = stor_rst_val / 1.0
            best_ValOutGT = stor_rst_valGT / 1.0
            best_ErrReal_1_val = ErrReal_1_val
            torch.save(net.state_dict(), save_path)
            # ---------------------------------------------------------------------------------

            # part4/4: 挑选准则2 相关内容。
            # ---------------------------------------------------------------------------------
            stor_Crtn1_small_val[best_epoch_count, 0]   = best_epoch
            stor_Crtn1_small_val[best_epoch_count, 1]   = best_MAE_1_val
            stor_Crtn1_small_val[best_epoch_count, 2]   = best_ErrReal_1_val
            stor_Crtn1_small_train_2nd[best_epoch_count, 0] = best_epoch
            stor_Crtn1_small_train_2nd[best_epoch_count, 1] = best_MAE_1_train_2nd
            stor_Crtn1_small_train_2nd[best_epoch_count, 2] = best_ErrReal_1_train_2nd

            temp_num        = idx_weight_extend[best_epoch_count]
            ForWeight_name2 = f"{temp_num}"
            ForWeight_name  = ForWeight_name1 + ForWeight_name2 + ForWeight_name3
            save_path_other = WeightOtherPathPrefix + ForWeight_name

            torch.save(net.state_dict(), save_path_other)

            best_epoch_count = best_epoch_count + 1
            # ---------------------------------------------------------------------------------

        if ((epoch+1)%save_EveryEpoch) < 1e-12:
            # 每50轮保存一下结果。

            # 保存(1/4)-训练阶段-训练集的损失保存
            # 损失-loss1-MAE-train
            train_loss1_np = (stor_loss1_MAE_train).detach().numpy()
            train_loss1_pd = pd.DataFrame(train_loss1_np)
            writer_train_loss1_USE = pd.ExcelWriter(loc_train_loss1_MAE)
            train_loss1_pd.to_excel(writer_train_loss1_USE, 'sheet1', float_format='%.10f')
            writer_train_loss1_USE._save()
            writer_train_loss1_USE.close()

            # 保存(2/4)-评估阶段-训练集相关内容
            # ---------------------------------------------------------------------------------
            # 保存训练集-不平衡量识别结果
            train_2nd_out_np = best_TrainOut_2nd.reshape(train_num, out_width)
            train_2nd_out_pd = pd.DataFrame(train_2nd_out_np)
            writer_train_2nd_USE = pd.ExcelWriter(loc_train_2nd_out)
            train_2nd_out_pd.to_excel(writer_train_2nd_USE, 'sheet1', float_format='%.10f')
            writer_train_2nd_USE._save()
            writer_train_2nd_USE.close()

            # 保存训练集-不平衡量识别结果-真值
            train_2nd_GT_out_np = best_TrainOut_2nd_GT.reshape(train_num, out_width)
            train_2nd_GT_out_pd = pd.DataFrame(train_2nd_GT_out_np)
            writer_train_2nd_GT_USE = pd.ExcelWriter(loc_train_2nd_outGT)
            train_2nd_GT_out_pd.to_excel(writer_train_2nd_GT_USE, 'sheet1', float_format='%.10f')
            writer_train_2nd_GT_USE._save()
            writer_train_2nd_GT_USE.close()

            # 保存训练集-损失迭代结果
            # 损失-Crtn1-MAE
            train_2nd_Crtn1_np = (stor_Crtn1_MAE_train_2nd).detach().numpy()
            train_2nd_Crtn1_pd = pd.DataFrame(train_2nd_Crtn1_np)
            writer_train_2nd_Crtn1_USE = pd.ExcelWriter(loc_train_2nd_Crtn1_MAE)
            train_2nd_Crtn1_pd.to_excel(writer_train_2nd_Crtn1_USE, 'sheet1', float_format='%.10f')
            writer_train_2nd_Crtn1_USE._save()
            writer_train_2nd_Crtn1_USE.close()
            # 损失-Crtn1-Err
            train_2nd_Crtn1_np = (stor_Crtn1_ErrReal_train_2nd).detach().numpy()
            train_2nd_Crtn1_pd = pd.DataFrame(train_2nd_Crtn1_np)
            writer_train_2nd_Crtn1_USE = pd.ExcelWriter(loc_train_2nd_Crtn1_ErrReal)
            train_2nd_Crtn1_pd.to_excel(writer_train_2nd_Crtn1_USE, 'sheet1', float_format='%.10f')
            writer_train_2nd_Crtn1_USE._save()
            writer_train_2nd_Crtn1_USE.close()
            # ---------------------------------------------------------------------------------

            # 保存(3/4)-评估阶段-验证集相关内容
            # ---------------------------------------------------------------------------------
            # 保存验证集-不平衡量识别结果
            val_out_np = best_ValOut.reshape(val_num, out_width)
            val_out_pd = pd.DataFrame(val_out_np)
            writer_val_USE = pd.ExcelWriter(loc_val_out)
            val_out_pd.to_excel(writer_val_USE, 'sheet1', float_format='%.10f')
            writer_val_USE._save()
            writer_val_USE.close()

            # 保存验证集-不平衡量识别结果-真值
            valGT_out_np = best_ValOutGT.reshape(val_num, out_width)
            valGT_out_pd = pd.DataFrame(valGT_out_np)
            writer_valGT_USE = pd.ExcelWriter(loc_val_outGT)
            valGT_out_pd.to_excel(writer_valGT_USE, 'sheet1', float_format='%.10f')
            writer_valGT_USE._save()
            writer_valGT_USE.close()

            # 保存验证集-损失迭代结果
            # 损失-Crtn1-MAE
            val_Crtn1_np = (stor_Crtn1_MAE_val).detach().numpy()
            val_Crtn1_pd = pd.DataFrame(val_Crtn1_np)
            writer_val_Crtn1_USE = pd.ExcelWriter(loc_val_Crtn1_MAE)
            val_Crtn1_pd.to_excel(writer_val_Crtn1_USE, 'sheet1', float_format='%.10f')
            writer_val_Crtn1_USE._save()
            writer_val_Crtn1_USE.close()
            # 损失-Crtn1-Err
            val_Crtn1_np = (stor_Crtn1_ErrReal_val).detach().numpy()
            val_Crtn1_pd = pd.DataFrame(val_Crtn1_np)
            writer_val_Crtn1_USE = pd.ExcelWriter(loc_val_Crtn1_ErrReal)
            val_Crtn1_pd.to_excel(writer_val_Crtn1_USE, 'sheet1', float_format='%.10f')
            writer_val_Crtn1_USE._save()
            writer_val_Crtn1_USE.close()
            # ---------------------------------------------------------------------------------

            # 保存(4/4)-附加内容
            # ---------------------------------------------------------------------------------
            # 保存训练集、验证集的 smaller 损失迭代结果
            train_2nd_SmallerCrtn1_np = (stor_Crtn1_small_train_2nd).detach().numpy()
            train_2nd_SmallerCrtn1_pd = pd.DataFrame(train_2nd_SmallerCrtn1_np)
            writer_train_2nd_SmallerCrtn1_USE = pd.ExcelWriter(loc_train_2nd_SmallerCrtn1)
            train_2nd_SmallerCrtn1_pd.to_excel(writer_train_2nd_SmallerCrtn1_USE, 'sheet1', float_format='%.10f')
            writer_train_2nd_SmallerCrtn1_USE._save()
            writer_train_2nd_SmallerCrtn1_USE.close()

            val_SmallerCrtn1_np = (stor_Crtn1_small_val).detach().numpy()
            val_SmallerCrtn1_pd = pd.DataFrame(val_SmallerCrtn1_np)
            writer_val_SmallerCrtn1_USE = pd.ExcelWriter(loc_val_SmallerCrtn1)
            val_SmallerCrtn1_pd.to_excel(writer_val_SmallerCrtn1_USE, 'sheet1', float_format='%.10f')
            writer_val_SmallerCrtn1_USE._save()
            writer_val_SmallerCrtn1_USE.close()
            # ---------------------------------------------------------------------------------

    print('Finished Training')



