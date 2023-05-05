#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test 
@File    ：Keras_Model.py
@Author  ：suhai
@Date    ：2023/5/4 22:45 
'''

import os
import numpy as np
import pandas as pd
import time
import math
from keras import Model
import keras.backend as K
from keras.layers import Embedding, Reshape, Input, Dot, Dense, Dropout, concatenate, BatchNormalization
from keras.models import load_model
from keras.utils import plot_model, to_categorical
from keras import regularizers
from keras.constraints import non_neg
from keras import optimizers


def getUI(dsname, dformat):  # 获取全部用户和项目
    st = time.time()
    train = pd.read_csv(dsname + '_train.txt', header=None, names=dformat)
    test = pd.read_csv(dsname + '_test.txt', header=None, names=dformat)
    data = pd.concat([train, test])
    all_user = np.unique(data['user'])
    all_item = np.unique(data['item'])
    train.sort_values(by=['user', 'item'], axis=0, inplace=True)  # 先按时间、再按用户排序
    if os.path.exists('./NN MF'):
        pass
    else:
        os.mkdir('./NN MF')
    train.to_csv('./NN MF/train.txt', index=False, header=0)
    test.to_csv('./NN MF/test.txt', index=False, header=0)
    et = time.time()
    print("get UI complete! cost time:", et - st)
    return all_user, all_item, train, test


def topk(dic, k):
    keys = []
    values = []
    for i in range(0, k):
        key, value = max(dic.items(), key=lambda x: x[1])
        keys.append(key)
        values.append(value)
        dic.pop(key)
    return keys, values


def cal_indicators(rankedlist, testlist):
    HITS_i = 0
    sum_precs = 0
    AP_i = 0
    len_R = 0
    len_T = 0
    MRR_i = 0

    ranked_score = []
    for n in range(len(rankedlist)):
        if rankedlist[n] in testlist:
            HITS_i += 1
            sum_precs += HITS_i / (n + 1.0)
            if MRR_i == 0:
                MRR_i = 1.0 / (rankedlist.index(rankedlist[n]) + 1)

        else:
            ranked_score.append(0)
    if HITS_i > 0:
        AP_i = sum_precs / len(testlist)
    len_R = len(rankedlist)
    len_T = len(testlist)
    return AP_i, len_R, len_T, MRR_i, HITS_i


def Recmand_model(num_user, num_item, d):
    K.clear_session()
    input_uer = Input(shape=[None, ], dtype="int32")
    model_uer = Embedding(num_user, d, input_length=1,
                          embeddings_constraint=non_neg()  # 非负，下同
                          )(input_uer)
    Dropout(0.2)
    model_uer = BatchNormalization()(model_uer)
    model_uer = Reshape((d,))(model_uer)

    input_item = Input(shape=[None, ], dtype="int32")
    model_item = Embedding(num_item, d, input_length=1,
                           embeddings_constraint=non_neg()
                           )(input_item)
    Dropout(0.2)
    model_item = BatchNormalization()(model_item)
    model_item = Reshape((d,))(model_item)

    out = Dot(1)([model_uer, model_item])  # 点积运算
    model = Model(inputs=[input_uer, input_item], outputs=out)
    model.compile(loss='mse', optimizer='sgd')
    model.summary()
    return model


def train(all_user, all_item, train_data, d):
    num_user = max(all_user) + 1
    num_item = max(all_item) + 1
    model = Recmand_model(num_user, num_item, d)
    train_user = train_data['user'].values
    train_item = train_data['item'].values
    train_x = [train_user, train_item]
    #    train_data['rating'] = 1 #不用评分
    train_y = train_data['rating'].values
    model.fit(train_x, train_y, batch_size=128, epochs=8)
    plot_model(model, to_file='./NN MF/NNMF.png', show_shapes=True)  # 输出框架图

    model.save("./NN MF/MFmodel.h5")


def test(train_data, test_data, all_item, k):
    model = load_model('./NN MF/MFmodel.h5')
    PRE = 0
    REC = 0
    MAP = 0
    MRR = 0
    AP = 0
    HITS = 0
    sum_R = 0
    sum_T = 0
    valid_cnt = 0
    stime = time.time()
    test_user = np.unique(test_data['user'])
    for user in test_user:
        #        user = 0

        visited_item = list(train_data[train_data['user'] == user]['item'])
        #        print('访问过的item:',visited_item)
        if len(visited_item) == 0:  # 没有训练数据，跳过
            continue
        per_st = time.time()
        testlist = list(test_data[test_data['user'] == user]['item'].drop_duplicates())  # 去重保留第一个
        testlist = list(set(testlist) - set(testlist).intersection(set(visited_item)))  # 去掉访问过的item

        if len(testlist) == 0:  # 过滤后为空，跳过
            continue
        valid_cnt = valid_cnt + 1  # 有效测试数

        poss = {}
        for item in all_item:
            if item in visited_item:
                continue
            else:
                poss[item] = float(model.predict([[user], [item]]))
        #        print(poss)
        #        print("对用户",user)
        rankedlist, test_score = topk(poss, k)
        #        print("Topk推荐:",rankedlist)
        #        print("实际访问:",testlist)
        #        print("单条推荐耗时:",time.time() - per_st)
        AP_i, len_R, len_T, MRR_i, HITS_i = cal_indicators(rankedlist, testlist)
        AP += AP_i
        sum_R += len_R
        sum_T += len_T
        MRR += MRR_i
        HITS += HITS_i
    #        print(test_score)
    #        print('--------')
    #        break
    etime = time.time()
    PRE = HITS / (sum_R * 1.0)
    REC = HITS / (sum_T * 1.0)
    MAP = AP / (valid_cnt * 1.0)
    MRR = MRR / (valid_cnt * 1.0)
    p_time = (etime - stime) / valid_cnt
    print('评价指标如下:')
    print('PRE@', k, ':', PRE)
    print('REC@', k, ':', REC)
    print('MAP@', k, ':', MAP)
    print('MRR@', k, ':', MRR)
    print('平均每条推荐耗时:', p_time)
    with open('./Basic MF/result_' + dsname + '.txt', 'w') as f:
        f.write('评价指标如下:\n')
        f.write('PRE@' + str(k) + ':' + str(PRE) + '\n')
        f.write('REC@' + str(k) + ':' + str(REC) + '\n')
        f.write('MAP@' + str(k) + ':' + str(MAP) + '\n')
        f.write('MRR@' + str(k) + ':' + str(MRR) + '\n')
        f.write('平均每条推荐耗时@:' + str(k) + ':' + str(p_time) + '\n')


if __name__ == '__main__':
    dsname = 'ML100K'
    dformat = ['user', 'item', 'rating', 'time']
    all_user, all_item, train_data, test_data = getUI(dsname, dformat)  # 第一次使用需取消注释
    d = 60  # 隐因子维度
    steps = 10
    k = 10
    train(all_user, all_item, train_data, d)
    test(train_data, test_data, all_item, k)
