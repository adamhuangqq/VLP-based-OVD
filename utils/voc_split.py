import os
import numpy as np
import random
if __name__=="__main__":
    file_name = '2007_train_voc.txt'
    with open(file_name,'r') as f:
        content_list = f.readlines()
    
    bike_list = []
    sofa_list = []
    cow_list = []
    cat_list = []
    rest_list = []

    content_list = [x.split() for x in content_list]
    num = 0
    for i in content_list:
        #num+=1
        for j in i[1:]:
            #print(j.split(','))
            if j.split(',')[-1] == '13' or j.split(',')[-1] == '5' or j.split(',')[-1] == '18':
                if i[0] not in sofa_list:
                    sofa_list.append(i[0])
            if j.split(',')[-1] == '10' or j.split(',')[-1] == '17':
                if i[0] not in bike_list:
                    bike_list.append(i[0])
            if j.split(',')[-1] == '16' or j.split(',')[-1] == '9' or j.split(',')[-1] == '12':
                if i[0] not in cow_list:
                    cow_list.append(i[0])
            if j.split(',')[-1] == '7' or j.split(',')[-1] == '3':
                if i[0] not in cat_list:
                    cat_list.append(i[0])

        if (i[0] not in cow_list) and (i[0] not in bike_list) and (i[0] not in sofa_list) and (i[0] not in cat_list) and (i[0] not in rest_list):
            rest_list.append(i[0])                   
        if num > 100:
            break
    print(len(sofa_list))
    print(len(bike_list))
    print(len(cow_list))
    print(len(cat_list))
    print(len(rest_list))
    save_path = 'voc_split/Main/'
    train_list = [x.split('/')[-1] for x in rest_list]
    sofa_list.extend(bike_list)
    sofa_list.extend(cow_list)
    sofa_list.extend(cat_list)
    test_list = list(set(sofa_list))
    test_list = [x.split('/')[-1] for x in test_list]
    random.shuffle(train_list)
    random.shuffle(test_list)
    print(len(train_list))
    print(len(test_list))


    train = open(save_path+'train.txt','w')
    for i in train_list:
        train.write(i[:-4])
        train.write('\n')
    test = open(save_path+'val.txt', 'w')
    for i in test_list:
        test.write(i[:-4])
        test.write('\n')
    train.close()
    test.close()

