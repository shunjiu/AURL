# AURL
This is the code for the ICASSP2023 Paper "An Asynchronous Updating Reinforcement Learning Framework for Task-oriented Dialog System". [Link](https://ieeexplore.ieee.org/document/10096940)

## Abstract

Reinforcement learning has been applied to train the dialog systems in many works. Previous approaches divide the dialog system into multiple modules including DST (dialog state tracking) and DP (dialog policy), and train these modules simultaneously. However, different modules influence each other during training. The errors from DST might misguide the dialog policy, and the system action brings extra difficulties for the DST module. To alleviate this problem, we propose **A**synchronous **U**pdating **R**einforcement **L**earning framework (AURL) that updates the DST module and the DP module asynchronously under a cooperative setting. Furthermore, curriculum learning is implemented to address the problem of unbalanced data distribution during reinforcement learning sampling, and multiple user models are introduced to increase the dialog diversity. Results on the public SSD-PHONE dataset show that our method achieves a compelling result with a 31.37% improvement on the dialog success rate.
## Dataset
Download dataset in [Link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=125708), put the SSD_phone dataset into 'data' dir.

## Requirements

- torch
- tqdm
- numpy
- sklearn

## Train

### Pretrain system model

```shell
python run.py --mode=sl_sys --device=cuda:0
```

### Pretrain user model

```shell
python run.py --mode=sl_user --device=cuda:0
```

### RL train
```shell
python run.py --simulator_num=2 --device=cuda:0
```