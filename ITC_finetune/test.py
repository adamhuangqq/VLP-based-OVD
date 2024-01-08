from pyexpat import model
import torch
import numpy as np
from net import ITC_net

# a = torch.rand([300,])
# #print(a)
# b = torch.rand([300,])
# with open('a.txt','a') as f:
#     for i in [a,b]:
#         tensor_string = ' '.join([str(item) for item in i.tolist()])
#         f.write(tensor_string+'\n')

# a = np.loadtxt('a.txt')
# a = torch.from_numpy(a)
# print(a.dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ITC_net(256,256)

model_dict      = model.state_dict()
pretrained_dict = torch.load('weights/ep200-loss0.20714.pth', map_location = device)
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)
print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")