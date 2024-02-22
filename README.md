# 环境
torch                          1.11.0+cu113
torchvision                    0.12.0
mmcv                           2.0.1
mmdet                          3.2.0
mmengine                       0.10.1
mmsegmentation                 1.2.1

# 性能指标
| 模型    | levir-cd(IoU%) | WHU Building(IoU%) |
|---------|:--------------:|--------------------|
| [changer](https://ieeexplore.ieee.org/document/10129139) | 82.64          | 71.67              |
| [tinycd](https://arxiv.org/abs/2207.13159)  | 83.57          | 84.30              |
| ours    | 83.04          | 85.57              |
