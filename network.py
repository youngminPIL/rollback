import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

# TODO:  modify load_net
def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)

def dup_net1(net_from, net_to, exception):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:3] == exception:
                if not name[0:3] == 'clas':
                    own_state[name].copy_(param)

def dup_net(net_from, net_to):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param)



def dup_net4(net_from, net_to, exception, exception2, exception3, exception4):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        if not name[0:len(exception4)] == exception4:
                            own_state[name].copy_(param)

def dup_net5(net_from, net_to, exception, exception2, exception3, exception4, exception5):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        if not name[0:len(exception4)] == exception4:
                            if not name[0:len(exception5)] == exception5:
                                own_state[name].copy_(param)

def dup_net6(net_from, net_to, exception, exception2, exception3, exception4, exception5, exception6 ):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        if not name[0:len(exception4)] == exception4:
                            if not name[0:len(exception5)] == exception5:
                                if not name[0:len(exception6)] == exception6:

                                    own_state[name].copy_(param)

def dup_net7(net_from, net_to, exception, exception2, exception3, exception4,
             exception5, exception6, exception7):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        if not name[0:len(exception4)] == exception4:
                            if not name[0:len(exception5)] == exception5:
                                if not name[0:len(exception6)] == exception6:
                                    if not name[0:len(exception7)] == exception7:
                                        own_state[name].copy_(param)

def dup_net20(net_from, net_to, exception, exception2, exception3, exception4,
             exception5, exception6, exception7,exception8,exception9,exception10,exception11,exception12,exception13,
              exception14,exception15,exception16,exception17,exception18,exception19,exception20):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        if not name[0:len(exception4)] == exception4:
                            if not name[0:len(exception5)] == exception5:
                                if not name[0:len(exception6)] == exception6:
                                    if not name[0:len(exception7)] == exception7:
                                        if not name[0:len(exception8)] == exception8:
                                            if not name[0:len(exception9)] == exception9:
                                                if not name[0:len(exception10)] == exception10:
                                                    if not name[0:len(exception11)] == exception11:
                                                        if not name[0:len(exception12)] == exception12:
                                                            if not name[0:len(exception13)] == exception13:
                                                                if not name[0:len(exception14)] == exception14:
                                                                    if not name[0:len(exception15)] == exception15:
                                                                        if not name[0:len(exception16)] == exception16:
                                                                            if not name[
                                                                                   0:len(exception17)] == exception17:
                                                                                if not name[0:len(
                                                                                    exception18)] == exception18:
                                                                                    if not name[0:len(
                                                                                            exception19)] == exception19:
                                                                                        if not name[0:len(
                                                                                                exception20)] == exception20:
                                                                                            if not name[0:len(
                                                                                                    exception21)] == exception21:
                                                                                                own_state[name].copy_(
                                                                                                    param)


def dup_net3(net_from, net_to, exception, exception2, exception3):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len(exception)] == exception:
                if not name[0:len(exception2)] == exception2:
                    if not name[0:len(exception3)] == exception3:
                        own_state[name].copy_(param)

def dup_net_target(net_from, net_to, target1, target2, target3, target4):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if name[0:len(target1)] == target1:
                own_state[name].copy_(param)

            if name[0:len(target2)] == target2:
                own_state[name].copy_(param)

            if name[0:len(target3)] == target3:
                own_state[name].copy_(param)

            if name[0:len(target4)] == target4:
                own_state[name].copy_(param)

def dup_net_target7(net_from, net_to, target1, target2, target3, target4, target5, fc, cls):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        #if name in state_dict:
        if name[0:len(target1)] == target1:
            own_state[name].copy_(param)

        if name[0:len(target2)] == target2:
            own_state[name].copy_(param)

        if name[0:len(target3)] == target3:
            own_state[name].copy_(param)

        if name[0:len(target4)] == target4:
            own_state[name].copy_(param)

        if name[0:len(target5)] == target5:
            own_state[name].copy_(param)

        if name[0:len(fc)] == fc:
            own_state[name].copy_(param)

        if name[0:len(cls)] == cls:
            own_state[name].copy_(param)
            name2 = name[:10] + '_1' + name[10:]
            own_state[name2].copy_(param)

def dup_net_target7_4way(net_from, net_to, target1, target2, target3, target4, target5, fc, cls):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        #if name in state_dict:
        if name[0:len(target1)] == target1:
            own_state[name].copy_(param)

        if name[0:len(target2)] == target2:
            own_state[name].copy_(param)

        if name[0:len(target3)] == target3:
            own_state[name].copy_(param)

        if name[0:len(target4)] == target4:
            own_state[name].copy_(param)

        if name[0:len(target5)] == target5:
            own_state[name].copy_(param)

        if name[0:len(fc)] == fc:
            own_state[name].copy_(param)

        if name[0:len(cls)] == cls:
            own_state[name].copy_(param)
            name2 = name[:10] + '_1' + name[10:]
            name3 = name[:10] + '_2' + name[10:]
            name4 = name[:10] + '_3' + name[10:]
            own_state[name2].copy_(param)
            own_state[name3].copy_(param)
            own_state[name4].copy_(param)

def dup_net_concat_copy(net_from, net_to):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            if not name[0:len('fc')] == 'fc':
                if not name[0:len('classifier')] == 'classifier':
                    own_state[name].copy_(param)


def add_A(name):
    name = name[0:5] + 'A' + name[5:]
    return name

def dup_net_concat_copy_1(net_from, net_to):
    state_dict = net_from.state_dict()
    own_state = net_to.state_dict()

    for name, param in state_dict.items():
        name = add_A(name)
        if name in own_state:
            if not name[0:len('fc')] == 'fc':
                if not name[0:len('classifier')] == 'classifier':
                    own_state[name].copy_(param)

def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
