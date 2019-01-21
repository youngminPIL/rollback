## Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification

Official Pytorch implementation of paper:

**[Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification]**(AAAI 2019).
Project page: https://sites.google.com/view/youngmin-ro-vision/home/rollback?authuser=0



## Environment
Python 3.6, Pytorch 0.4.1, Torchvision, tensorboard(optional)


## Train 
Default setting:
- Architecture: ResNet-50
- Dataset: Market-1501
- Batch size: 32
- Image size: 288X144
- Train 4 period. 

### prepare
The dataset path should be changed to your own path.

Market-1501 dataset are available on http://www.liangzheng.org/Project/project_reid.html

```
prepare.py 
```
### train network on the each periods. 

Train model in period 1. This is a baseline of our algorithm. 

The dataset path(data_dir='/home/ro/Reid/Market/pytorch') should be changed to your own path.


```
train_resnet_p1.py
```

Each period should be trained on the results of previous training.
```
train_resnet_p2.py
```

```
train_resnet_p3.py
```

```
train_resnet_p4.py
```

## Test

The test will be done when you complete your trainung up to period 4. 

The dataset path(test_dir='/home/ro/Reid/Market/pytorch') should be changed to your own path.

```
test_resnet.py
```



## Citation

```
@inproceedings{rollback_v1,
	title = {Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification
},
	author = {Youngmin Ro, Jongwon Choi, Dae Ung Jo, Byeongho Heo, Jongin Lim, Jin Young Choi},
	booktitle = {AAAI},
	year = {2019}
}
```
Youngmin Ro, Jongwon Choi, Dae Ung Jo, Byeongho Heo, Jongin Lim, Jin Young Choi, "
Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification", CoRR, 2019. (AAAI at 2019 Feb.)



