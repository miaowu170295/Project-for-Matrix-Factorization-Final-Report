#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test 
@File    ：TensorFlow.py
@Author  ：xiaohitu
@Date    ：2023/5/5 9:37 
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import random
import time
from collections import defaultdict


def getUI(dsname, dformat):  # 获取全部用户和项目
    train = pd.read_csv(dsname + '_train.txt', header=None, names=dformat)
    test = pd.read_csv(dsname + '_test.txt', header=None, names=dformat)
    data = pd.concat([train, test])
    all_user = np.unique(data['user'])
    all_item = np.unique(data['item'])

    if os.path.exists('./NN-BPRMF'):
        pass
    else:
        os.mkdir('./NN-BPRMF')
    train.to_csv('./NN-BPRMF/train.txt', index=False, header=0)
    test.to_csv('./NN-BPRMF/test.txt', index=False, header=0)
    return all_user, all_item, train, test


def load_data(train_data, all_user, all_item):
    user_ratings = defaultdict(set)
    # user_ratings = {}
    max_u_id = max(all_user)
    max_i_id = max(all_item)
    for i in range(0, len(train_data)):
        u = int(train_data.iloc[i]['user'])
        i = int(train_data.iloc[i]['item'])
        # user_ratings[u] = i
        user_ratings[u].add(i)

    print('max_u_id:', max_u_id)
    print('max_i_id:', max_i_id)

    # print(user_ratings)
    return max_u_id, max_i_id, user_ratings


def generate_train_batch(user_ratings, item_count, batch_size):
    """
    构造训练用的三元组
    对于随机抽出的用户u，i可以从user_ratings随机抽出，而j也是从总的电影集中随机抽出，当然j必须保证(u,j)不在user_ratings中
    """
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    return np.asarray(t)


def bpr_mf(user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1),
                                 # initializer = tf.random_uniform_initializer(-1,1)
                                 )
    item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1)
                                 # initializer = tf.random_uniform_initializer(-1,1)
                                 )

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)

    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    regulation_rate = 0.01
    bprloss = tf.reduce_mean(tf.log(tf.sigmoid(x))) + regulation_rate * l2_norm
    train_op = tf.train.GradientDescentOptimizer(0.005).minimize(-bprloss)
    return u, i, j, bprloss, train_op


def topk(dic, k):
    keys = []
    values = []
    for i in range(0, k):
        key, value = min(dic.items(), key=lambda x: x[1])
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


def train(iters, step, batch_size):
    user_count, item_count, user_ratings = load_data(train_data, all_user, all_item)
    L = []
    with tf.Graph().as_default(), tf.Session() as sess:
        u, i, j, bprloss, train_op = bpr_mf(user_count, item_count, dimension)
        sess.run(tf.global_variables_initializer())
        for epoch in range(iters):
            st = time.time()
            print('----------------')
            print("epoch", epoch + 1, 'start')
            _batch_bprloss = 0
            for k in range(step):
                uij = generate_train_batch(user_ratings, item_count, batch_size)
                _bprloss, _train_op = sess.run([bprloss, train_op],
                                               feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
                _batch_bprloss += _bprloss
            print("bpr_loss:", _batch_bprloss)
            L.append(_batch_bprloss)
            print("epoch", epoch + 1, "spend :", time.time() - st, 's')
        variable_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variable_names)
        cnt = 0
        for k, v in zip(variable_names, values):
            print('================')
            print("Variable: ", k)
            print("Shape: ", v.shape)
            if cnt == 0:
                np.savetxt('./NN-BPRMF/userVec.txt', v, delimiter=',', newline='\n')
                cnt = cnt + 1
            else:
                np.savetxt('./NN-BPRMF/itemVec.txt', v, delimiter=',', newline='\n')

    L = np.array(L)


def test(all_user, all_item, train_data, test_data, dimension, k):
    userP = np.loadtxt('./NN-BPRMF/userVec.txt', delimiter=',', dtype=float)
    itemP = np.loadtxt('./NN-BPRMF/itemVec.txt', delimiter=',', dtype=float)
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
                poss[item] = np.dot(userP[user], itemP[item])
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
    print('总耗时：', etime - stime)
    with open('./BPR/result_' + dsname + '.txt', 'w') as f:
        f.write('评价指标如下:\n')
        f.write('PRE@' + str(k) + ':' + str(PRE) + '\n')
        f.write('REC@' + str(k) + ':' + str(REC) + '\n')
        f.write('MAP@' + str(k) + ':' + str(MAP) + '\n')
        f.write('MRR@' + str(k) + ':' + str(MRR) + '\n')
        f.write('平均每条推荐耗时@:' + str(k) + ':' + str(p_time) + '\n')
        f.write('总耗时@:' + str(k) + ':' + str(etime - stime) + 's\n')


if __name__ == '__main__':
    dsname = 'ML100K'
    dformat = ['user', 'item', 'rating', 'time']
    iters = 24
    step = 10000
    dimension = 60
    batch_size = 16

    all_user, all_item, train_data, test_data = getUI(dsname, dformat)
    train(iters, step, batch_size)
    k = 10
    test(all_user, all_item, train_data, test_data, dimension, k)
#    k = 10
#    test(all_user,all_item,train_data,test_data,dimension,k)
#    k = 20
#    test(all_user,all_item,train_data,test_data,dimension,k)

