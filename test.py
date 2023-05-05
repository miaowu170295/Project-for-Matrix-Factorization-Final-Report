#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：test 
@File    ：test.py
@Author  ：xiaohitu
@Date    ：2023/4/24 21:28 
'''
import pandas as pd
import numpy as np
def readML():
    usnames = ['user', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_table("ml-1m/users.dat", sep='::', header=None, names=usnames, engine='python',encoding='ISO-8859-1')
    itnames = ['item', 'title', 'genres']
    item = pd.read_table("ml-1m/movies.dat", sep='::', header=None, names=itnames, engine='python',encoding='ISO-8859-1')
    rtnames = ['user', 'item', 'rating', 'time']
    rating = pd.read_table("ml-1m/ratings.dat", sep='::', header=None, names=rtnames, engine='python',encoding='ISO-8859-1')

    print(user.head())
    print('-----------------------------')
    print(item.head())
    print('-----------------------------')
    print(rating.head())
    print('-----------------------------')

    print('注意：1.用户和项目都是从1开始编号的')
    print("-----------summary--------------")
    ucnt = max(np.unique(rating['user']))
    icnt = max(np.unique(rating['item']))
    rcnt = len(rating)

    print("用户数：", ucnt)
    print("项目数：", icnt)
    print("记录数：", rcnt)
    print("density:", rcnt / float(ucnt * icnt))
    u_ckcnt = rating['user'].value_counts().values
    u_ckcnt_min = min(u_ckcnt)
    u_ckcnt_avg = float(sum(u_ckcnt)) / len(u_ckcnt)
    print("用户最小访问项目次数：", u_ckcnt_min)
    print("用户平均访问项目次数：", u_ckcnt_avg)

    i_ckcnt = rating['item'].value_counts().values
    i_ckcnt_min = min(i_ckcnt)
    i_ckcnt_avg = float(sum(i_ckcnt)) / len(i_ckcnt)
    print("项目最小被访问项目次数：", i_ckcnt_min)
    print("项目平均被访问项目次数：", i_ckcnt_avg)



if __name__ == '__main__':
    readML()
