# import numpy 
import numpy as np 
import torch
import torch.nn.functional as F
import glob
  
# using numpy.fill_diagonal() method 
# array = np.array([[1, 2, 3], [2, 1, 3]]) 
# np.fill_diagonal(array, 5) 
# print(array)

# a = torch.tensor([[1,1,1],[1,1,1]])
# b = a.clone()
# print(b)

# b = b + 1
# print(b)
# print(a)

# a = torch.randn(4,5)
# b = torch.cat([torch.ones(2,dtype=torch.long),torch.zeros(2,dtype=torch.long)],dim=0)
# print(a.shape, b.shape)
# c = F.cross_entropy(a,b)
# print(c)

if __name__ == '__main__':
    image_path = 'D:/VSCODE/prompt/imgs/'
    image_list = glob.glob(image_path+'000001'+'*.jpg')
    print(image_list)
    
