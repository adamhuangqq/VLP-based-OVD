import os
import numpy as np
import shutil

if __name__=='__main__':
    path = 'F:/car-detect/fwq/'
    save_path = 'F:/car-detect/dataset/'
    for root,dirs,files in os.walk(path):
        # for dir1 in dirs1:
        #     for root2,dirs2,files2 in os.walk(os.path.join(root1, dir1)):
        #         for dir2 in dirs2:
        #             for root3,dirs3,files3 in os.walk(os.path.join(root2, dir2)):
        #                 for dir3 in dirs3:
        #                     if dir3!='split':
        #                         for root4,dirs4,files4 in os.walk(os.path.join(root2, dir2)):

        for file in files:
            path = os.path.join(root,file)
            path_list = path.split('\\')
            if not('split' in path_list):
                if path[-4:] == '.txt':
                    shutil.copy(path,save_path+'txt/')
                elif path[-4:] == 'json':
                    shutil.copy(path,save_path+'json/')
                else:
                    shutil.copy(path,save_path+'images/')
                print(path)
            