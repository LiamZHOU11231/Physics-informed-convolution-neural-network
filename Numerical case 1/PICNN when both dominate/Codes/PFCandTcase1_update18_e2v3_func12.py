# ========================================================================================================
# 功能：生成路径的txt文件索引。
# ========================================================================================================

# Part1/3：生成训练集的索引，包括输入与输出 -------------------------------------------------------------------
'''
file_path = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\TrainNormPathTXT.txt"
num_train = 24000
with (open(file_path, 'w') as file):
    for i in range(num_train):
        filename1 = f"SampleTrainNorm_{i+1}.npy"
        filename2 = f"SampleTrainOutNorm_{i+1}.npy"

        line1 = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\\trainNorm\\" \
                + filename1 + \
                " " + \
                "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_out\\trainNorm\\" \
                + filename2

        file.write(f'{line1}\n')
'''

# Part2/3：生成验证集的索引，包括输入与输出 -------------------------------------------------------------------
'''
file_path = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\ValNormPathTXT.txt"
num_val = 3000
with open(file_path, 'w') as file:
    for i in range(num_val):
        filename1 = f"SampleValNorm_{i+1}.npy"
        filename2 = f"SampleValOutNorm_{i+1}.npy"

        line1 = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\\valNorm\\" \
                + filename1 + \
                " " + \
                "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_out\\valNorm\\" \
                + filename2

        file.write(f'{line1}\n')
'''

# Part3/3：生成测试集的索引，包括输入与输出 -------------------------------------------------------------------
'''
file_path = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\TestNormPathTXT.txt"
num_test = 3000
with open(file_path, 'w') as file:
    for i in range(num_test):
        filename1 = f"SampleTestNorm_{i+1}.npy"
        filename2 = f"SampleTestOutNorm_{i+1}.npy"

        line1 = "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_Input_70dB\\testNorm\\" \
                + filename1 + \
                " " + \
                "D:\RFEMNN_InPyCharm\project1\ForPFCandT_NEW\PFCandT_case1_update18_e2v3\DB_out\\testNorm\\" \
                + filename2

        file.write(f'{line1}\n')
'''

