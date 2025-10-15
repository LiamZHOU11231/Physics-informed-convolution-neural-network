# ========================================================================================================
# 功能：读取excel数据进行切片。
# ========================================================================================================
import pandas as pd
import numpy as np
import os

# 训练集输入的切片（Part1/6）----------------------------------------------------------------------------------
# part(1/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_1in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

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

# part(2/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_2in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+2881}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_2881.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(3/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_3in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+5761}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_5762.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(4/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_4in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+8641}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_8641.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(5/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_5in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+11521}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_11521.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(6/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_6in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+14401}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_14401.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(7/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_7in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+17281}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_17281.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(8/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_8in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+20161}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_20161.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(9/10)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_9in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+23041}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_23041.npy")
check_data = np.load(check_route)

print(check_data)
'''

# part(10/10)

dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTrainNorm_10in10.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "trainNorm")
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 2880

for i in range(num_train):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       # ⭐️ ------------------------------------------- ⭐️
       filename = f"SampleTrainNorm_{i+25921}.npy"
       # ⭐️ ------------------------------------------- ⭐️

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "\SampleTrainNorm_25921.npy")
check_data = np.load(check_route)

print(check_data)



# 训练集输出的切片（Part2/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\PFCandT_OutputTrainNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\\trainNorm"
# 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_train = 28800

for i in range(num_train):
       sample_data = dataframe.iloc[i, :].values.copy()

       filename = f"SampleTrainOutNorm_{i+1}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleTrainOutNorm_15.npy")
check_data = np.load(check_route)

print(check_data)
'''


# 验证集输入的切片（Part3/6）----------------------------------------------------------------------------------
# part(1/2)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputValNorm_1in2.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "valNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_val = 1800

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

# part(2/2)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputValNorm_2in2.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "valNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_val = 1800

for i in range(num_val):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       filename = f"SampleValNorm_{i+1801}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleValNorm_1802.npy")
check_data = np.load(check_route)

print(check_data)
'''


# 验证集输出的切片（Part4/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\PFCandT_OutputValNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\\valNorm"      # 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_val = 3600

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
# part(1/2)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTestNorm_1in2.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "testNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_test = 1800

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

# part(2/2)
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\PFCandT_InputTestNorm_2in2.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = ("D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_Input_70dB\\"
           "testNorm")
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_test = 1800

for i in range(num_test):
       sample_data = dataframe.iloc[:, 5*(i):5*(i+1)].values.copy()

       filename = f"SampleTestNorm_{i+1801}.npy"

       save_path = os.path.join(out_dir, filename)

       np.save(save_path, sample_data)

# 检查保存的是否正确
check_route = (out_dir + "/SampleTestNorm_1801.npy")
check_data = np.load(check_route)

print(check_data)
'''


# 测试集输出的切片（Part6/6）----------------------------------------------------------------------------------
'''
dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\PFCandT_OutputTestNorm.xlsx"
# 这里注意断开的时候，别在\处断开。

out_dir = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_caseBase2_update1\DB_out\\testNorm"      # 注意要求该文件夹已存在。
# os.mkdir(out_dir)  # 当 out_dir 表示的文件夹不存在时，用该语句来创建文件夹。

dataframe = pd.read_excel(dir, header=None)
# pd.read_excel 直接得到的是 dataframe类型的文件。

num_test = 3600

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

