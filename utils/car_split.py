import os
import numpy as np
import random
if __name__=="__main__":
    file_name = '2007_train_car.txt'
    with open(file_name,'r') as f:
        content_list = f.readlines()
    
    bus_list = []
    van_list = []
    suv_list = []
    rest_list = []

    content_list = [x.split() for x in content_list]
    num = 0
    for i in content_list:
        #num+=1
        for j in i[1:]:
            if j[-1] == '1':
                if i[0] not in van_list:
                    van_list.append(i[0])
            if j[-1] == '6':
                if i[0] not in bus_list:
                    bus_list.append(i[0])
            if j[-1] == '3':
                if i[0] not in suv_list:
                    suv_list.append(i[0])

        if (i[0] not in suv_list) and (i[0] not in bus_list) and (i[0] not in van_list) and (i[0] not in rest_list):
            rest_list.append(i[0])                   
        if num > 100:
            break
    print(len(van_list))
    print(len(bus_list))
    print(len(suv_list))
    print(len(rest_list))
    save_path = 'car_split/Main/'
    train_list = [x.split('/')[-1] for x in rest_list]
    van_list.extend(bus_list)
    van_list.extend(suv_list)
    test_list = list(set(van_list))
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

