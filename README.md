## Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification

Official Pytorch implementation of paper:

**[Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification]**(AAAI 2019).


## Environment
Python 3.6, Pytorch 0.4.1, Torchvision, tensorboard(optional)


## Train 
Train model on the Market-1501 is defult setting

### prepare
```
prepare.py 
```
### train network on the each periods. 

Train resnt-50 in period 1. this is the baseline of our algorithm.
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

The test will be done when you train until period 4. 

```
test_resnet.py
```



## Citation

```
@inproceedings{ABdistill,
	title = {Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification
},
	author = {Youngmin Ro, Jongwon Choi, Dae Ung Jo, Byeongho Heo, Jongin Lim, Jin Young Choi},
	booktitle = {AAAI},
	year = {2019}
}
```
Youngmin Ro, Jongwon Choi, Dae Ung Jo, Byeongho Heo, Jongin Lim, Jin Young Choi, "
Backbone Can Not be Trained at Once: Rolling Back to Pre-trained Network for Person Re-Identification", CoRR, 2019. (AAAI at 2019 Feb.)



