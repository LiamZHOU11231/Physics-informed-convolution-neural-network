# ========================================================================================================
# 功能：读取excel数据进行切片。
# ========================================================================================================
import pandas as pd
import numpy as np
import os

# 训练集输入的切片（Part1/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\PFT_InputTrainNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2400

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       filename = f"SampleTrainNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_1.npy")
check_data = np.load(check_route)

print(check_data)
'''

# 训练集输出的切片（Part2/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\PFT_OutputTrainNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\\trainNorm"      # 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2400

for i in range(num_train):
       sample_data = dataframe.iloc[i, :].values.copy()

       filename = f"SampleTrainOutNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleTrainOutNorm_1.npy")
check_data = np.load(check_route)

print(check_data)
'''

# 验证集输入的切片（Part3/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\PFT_InputValNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\\"
           "valNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_val = 300

for i in range(num_val):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       filename = f"SampleValNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleValNorm_1.npy")
check_data = np.load(check_route)

print(check_data)
'''

# 验证集输出的切片（Part4/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\PFT_OutputValNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\\valNorm"      # 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_val = 300

for i in range(num_val):
       sample_data = dataframe.iloc[i, :].values.copy()

       filename = f"SampleValOutNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleValOutNorm_21.npy")
check_data = np.load(check_route)

print(check_data)
'''

# 测试集输入的切片（Part5/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\PFT_InputTestNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_Input_70dB\\"
           "testNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_test = 300

for i in range(num_test):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       filename = f"SampleTestNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleTestNorm_1.npy")
check_data = np.load(check_route)

print(check_data)
'''

# 测试集输出的切片（Part6/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\PFT_OutputTestNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFT_NEW\PFT_case1_update12_e2v3\DB_out\\testNorm"      # 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_test = 300

for i in range(num_test):
       sample_data = dataframe.iloc[i, :].values.copy()

       filename = f"SampleTestOutNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleTestOutNorm_12.npy")
check_data = np.load(check_route)

print(check_data)
'''

