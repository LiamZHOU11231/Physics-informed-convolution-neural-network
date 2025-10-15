# Physics-informed-convolution-neural-network
An physics-informed convolution neural network for unbalance-related problems


## Abstract
We propose a new framework for unbalance localization and identification of a multi-disk-rotor-bearing-casing system. In a word, compared to pure CNN methods, the proposed PICNN can achieve higher precision with fewer hyperparameters and possesses physical interpretability at the same time. The inputs consist of the pulse signal from the photoelectric transducer and displacement signals from four eddy current sensors. All of the inputs are in the time domain. The outputs include the unbalance amplitudes and phases. Here, the dimension of the input is 1024*5, while that of the output is 1*4 or 1*2. More details can be seen when this research is published. Currently (2025-10-15), this work is under review. The codes and datasets for unbalance identification are provided here. In terms of the PI layer based on the FRF of the dynamic system, its working principle has been elaborated in the paper.

## Introduction
Folder: Numerical case 1. (Taking the subfolder "PICNN when disk 1 dominates" as an example)
1. Datasets. The DB_input folder includes three Excel files (training, validating, testing), which can be processed by the PFCcase1_update12_e2v3_func1.py in the Codes folder. The processing of the Excel files in the DB_output folder is similar to that of the input files.
2. Codes. PFCcase1_update12_e2v3_func1.py and PFCcase1_update12_e2v3_func12.py aim to create slices and indices. The functions of other codes can be looked up in the training code named PFCcase1_update12_e2v3_main_train_e13.py.

Folder: Numerical case 2 (with casing). Similar to the explanation above.

Folder: Experimental case. Similar to the explanation above.






