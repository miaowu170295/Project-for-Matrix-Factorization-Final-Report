#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test 
@File    ：Basic_MF.py
@Author  ：suhai
@Date    ：2023/5/4 22:39 
'''
import os
import numpy as np
import pandas as pd
import time
import math


def getUI(dsname, dformat):  # 获取全部用户和项目
    st = time.time()
    train = pd.read_csv(dsname + '_train.txt', header=None, names=dformat)
    test = pd.read_csv(dsname + '_test.txt', header=None, names=dformat)
    data = pd.concat([train, test])
    all_user = np.unique(data['user'])
    all_item = np.unique(data['item'])
    train.sort_values(by=['user', 'item'], axis=0, inplace=True)  # 先按时间、再按用户排序
    num_user = max(all_user) + 1
    num_item = max(all_item) + 1
    rating = np.zeros([num_user, num_item])
    for i in range(0, len(train)):
        user = train.iloc[i]['user']
        item = train.iloc[i]['item']
        score = train.iloc[i]['rating']
        #        score = 1
        rating[user][item] = score
    if os.path.exists('./Basic MF'):
        pass
    else:
        os.mkdir('./Basic MF')
    train.to_csv('./Basic MF/train.txt', index=False, header=0)
    test.to_csv('./Basic MF/test.txt', index=False, header=0)
    np.savetxt('./Basic MF/rating.txt', rating, delimiter=',', newline='\n')
    et = time.time()
    print("get UI complete! cost time:", et - st)


def getData(dformat):
    rating = np.loadtxt('./Basic MF/rating.txt', delimiter=',', dtype=float)
    train = pd.read_csv('./Basic MF/train.txt', header=None, names=dformat)
    test = pd.read_csv('./Basic MF/test.txt', header=None, names=dformat)
    data = pd.concat([train, test])
    all_user = np.unique(data['user'])
    all_item = np.unique(data['item'])
    return rating, train, test, all_user, all_item


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


def matrix_factorization(R, P, Q, d, steps, alpha=0.002, lamda=2e-6):
    Q = Q.T
    sum_st = 0  # 总时长
    e_old = 0  # 上一次的loss
    flag = 1
    for step in range(steps):  # 梯度下降结束条件：1.满足最大迭代次数，跳出
        st = time.time()
        e_new = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i] > 0:
                    eui = R[u][i] - np.dot(P[u, :], Q[:, i])
                    for k in range(d):
                        P[u][k] = P[u][k] + alpha * eui * Q[k][i] - lamda * P[u][k]
                        Q[k][i] = Q[k][i] + alpha * eui * P[u][k] - lamda * Q[k][i]
        cnt = 0
        for u in range(len(R)):
            for i in range(len(R[u])):
                if R[u][i] > 0:
                    cnt = cnt + 1
                    e_new = e_new + pow(R[u][i] - np.dot(P[u, :], Q[:, i]), 2)
        et = time.time()
        print('step', step + 1, 'cost time:', et - st)
        e_new = e_new / cnt
        if step == 0:  # 第一次不算loss之差
            e_old = e_new
            continue
        sum_st = sum_st + (et - st)
        if e_new < 1e-3:  # 梯度下降结束条件：2.loss过小，跳出
            flag = 2
            break
        if e_old - e_new < 1e-10:  # 梯度下降结束条件：3.loss之差过小，梯度消失，跳出
            flag = 3
            break
        else:
            e_old = e_new
    print('---------Summary----------\n',
          'Type of jump out:', flag, '\n',
          'Total steps:', step + 1, '\n',
          'Total time:', sum_st, '\n',
          'Average time:', sum_st / (step + 1.0), '\n',
          "The e is:", e_new)
    return P, Q.T


def train(rating, d, steps):
    R = rating
    N = len(R)  # 用户数
    M = len(R[0])  # 项目数
    P = np.random.normal(loc=0, scale=0.01, size=(N, d))
    Q = np.random.normal(loc=0, scale=0.01, size=(M, d))
    nP, nQ = matrix_factorization(R, P, Q, d, steps)
    #    nR=np.dot(nP,nQ.T)
    np.savetxt('./Basic MF/nP.txt', nP, delimiter=',', newline='\n')
    np.savetxt('./Basic MF/nQ.txt', nQ, delimiter=',', newline='\n')


def test(train_data, test_data, all_item, dsname, k):
    nP = np.loadtxt('./Basic MF/nP.txt', delimiter=',', dtype=float)
    nQ = np.loadtxt('./Basic MF/nQ.txt', delimiter=',', dtype=float)
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
        #        print("对用户",user)
        valid_cnt = valid_cnt + 1  # 有效测试数
        poss = {}
        for item in all_item:
            if item in visited_item:
                continue
            else:
                poss[item] = np.dot(nP[user], nQ[item])
        #        print(poss)
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
    getUI(dsname, dformat)  # 第一次使用后可注释
    rating, train_data, test_data, all_user, all_item = getData(dformat)
    d = 40  # 隐因子维度
    steps = 10
    k = 10
    train(rating, d, steps)
    test(train_data, test_data, all_item, dsname, k)

