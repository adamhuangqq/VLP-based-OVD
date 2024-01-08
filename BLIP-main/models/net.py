import torch.nn as nn
import torch
import torch.nn.functional as F
class ITC_net(nn.Module):

    def __init__(self, emb_dim, hid_dim):
        super(ITC_net, self).__init__()
        self.l1 = nn.Linear(emb_dim, hid_dim)
        self.l2 = nn.Linear(emb_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        #out = F.normalize(out,dim=-1)
        return out
    
def weights_init(net, init_type='normal', init_gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Linear') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                torch.nn.init.normal_(m.bias.data, 0.0, 0.0)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)