# ========================================================================================================
# 功能：建立数据集，用Norm数据。
# ========================================================================================================
import os
import torch.utils.data as data
import numpy as np
import torch


# Caution 1/2  只修改了下面标出范围内的部分 =============================================================
# ---------------------------------------------------------------------------------------------------
class DB_PFCcase1_update12_e2v3(data.Dataset):
# ---------------------------------------------------------------------------------------------------
    def __init__(self, mode='train'):
        self.datas_id = []
        # 得到的 datas_id 是 list 变量类型
        self.mode = mode

# Caution 2/2  只修改了下面标出范围内的部分 =============================================================
# ---------------------------------------------------------------------------------------------------
        data_dir = "D:\RFEMNN_InPyCharm\project1\ForPFC\PFC_case1_update12_e2v3\DB_Input_70dB\\"
# ---------------------------------------------------------------------------------------------------

        if mode == 'train':
            imgset_path = data_dir + 'TrainNormPathTXT.txt'
        elif mode == 'val':
            imgset_path = data_dir + 'ValNormPathTXT.txt'
        elif mode == 'test':
            imgset_path = data_dir + 'TestNormPathTXT.txt'
        else:
            imgset_path = 'WRONG!'

        # 打开txt文件
        imgset_file = open(imgset_path)

        for line in imgset_file:
            group = {}
            # 得到的 group 是 dict 变量类型
            group['img_path'] = line.strip("\n").split(" ")[0]  # 输入特征的路径
            # 可以理解为，group 里的某一行，行号记为 'img_path'，存放着上述等式右端的字符串。
            group['gt_path']  = line.strip("\n").split(" ")[1]  # 输出标签的路径

            self.datas_id.append(group)
            # .append 是向列表末尾添加元素。

    def __getitem__(self, index):
        # 确保图片路径存在
        assert os.path.exists(self.datas_id[index]['img_path']), \
            ('{} does not exist'.format(self.datas_id[index]['img_path']))
        # 如果 assert 后面的语句出错，就会提示 AssertionError
        # 逗号后面的是报错时，额外会提示的语句
        assert os.path.exists(self.datas_id[index]['gt_path']), (
            '{} does not exist'.format(self.datas_id[index]['gt_path']))

        # .npy读取
        img = np.load(self.datas_id[index]['img_path']).astype(np.float32)
        gt = np.load(self.datas_id[index]['gt_path']).astype(np.float32)

        sample = {'img': img, 'gt': gt}

        # tensor化
        sample['img'] = torch.tensor(sample['img'])
        #sample['img'] = torch.unsqueeze(sample['img'],0)
        # 输入特征tensor化。
        sample['gt']  = torch.tensor(sample['gt'])
        #sample['img'] = torch.unsqueeze(sample['img'], 0)

        return sample

    def __len__(self):

        return len(self.datas_id)





