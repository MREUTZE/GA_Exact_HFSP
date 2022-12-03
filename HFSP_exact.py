#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import re
import xlsxwriter
import pandas as pd
from gurobipy import Model, GRB, tuplelist, tupledict, quicksum, max_, LinExpr


# ## 不考虑时间窗

# In[3]:


def HFSP(N, M, H, P, Q):
    """
    J:订单序列,J=[1,2,..,N]，N:订单总数
    I：I=[1，2，...M],每个stage的机器序列，M：机器总数
    S：stage的序列,S=[1,2,...,H]，H：stage总数

    P：P=[0,0,...,0            第一个是虚拟的stage s0
          0,P11,...P1J,0       前后两列是两个虚拟订单
          0,P21,...P2J,0
          ...
          0,PS1,...PSJ,0],      Psj:订单j在stage s的处理时间

    Q：一个很大的数

    """
    # 构建订单序列
    J = tuplelist([j + 1 for j in range(N)])
    J_j0 = tuplelist([j for j in range(N + 1)])
    J_jN1 = tuplelist([j + 1 for j in range(N + 1)])
    # 构建机器序列
    I = tuplelist([m for m in range(M)])
    # 构建stage序列
    S = tuplelist([s + 1 for s in range(H)])
    S_s0 = tuplelist([s for s in range(H + 1)])
    # 实例化模型
    model = Model('HFSP')
    # 添加变量
    X = model.addVars(N + 2, N + 2, M, H + 1, vtype=GRB.BINARY, name='X[i,j,m,s]')  # N+2是因为有两个虚拟订单
    C = model.addVars(H + 1, N + 2, vtype=GRB.CONTINUOUS, name='C[s,j]')
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    # 约束1：每个作业在每个阶段中只由一台机器处理
    model.addConstrs(quicksum(X[i, j, m, s] for i in J_j0 if i != j for m in I) == 1 for j in J for s in S)

    # 约束2,3：没有一台机器可以同时处理多个作业
    # model.addConstrs( quicksum(quicksum( X[i,j,m,s] for i in J_j0 if i!=j) - quicksum( X[j,k,m,s] for k in J_jN1 if k!=j ) for m in I)==0 for j in J for s in S )
    model.addConstrs(
        quicksum(X[i, j, m, s] for i in J_j0 if i != j for m in I) == quicksum(
            X[j, k, m, s] for k in J_jN1 if k != j for m in I)
        for j in J for s in S)
    # model.addConstrs(
    #     quicksum(X[i, j, m, s] for i in J_j0 if i != j) == quicksum(X[j, k, m, s] for k in J_jN1 if k != j) for m in I
    #     for j in J for s in S)
    model.addConstrs(quicksum(X[0, i, m, s] for i in J) <= 1 for m in I for s in S)

    # 约束4,5：满足优先约束
    model.addConstrs(C[s, j] >= C[s - 1, j] + P[s][j] for j in J for s in S)

    model.addConstrs(C[0, j] == 0 for j in J)

    # 约束6：工作完成时间约束
    model.addConstrs(
        C[s, j] >= C[s, i] + P[s][j] - Q * (1 - quicksum(X[i, j, m, s] for m in I)) for i in J for j in J if i != j for
        s in S)

    #  获得最大完工时间z
    model.addConstrs(C[s, j] <= z for j in J for s in S)

    model.setObjective(z, GRB.MINIMIZE)

    "求解"
    model.Params.LogToConsole = True  # 显示求解过程
    # model.Params.TimeLimit = 300  # 设置求解时间上限
    model.optimize()
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        for s in range(H + 1):
            for m in I:
                for i in range(N + 2):
                    for j in range(N + 2):
                        if i != j:
                            if X[i, j, m, s].x > 0:
                                print("X[{},{},{},{}]=1".format(i, j, m, s))
        for s in range(1, H + 1):
            for j in range(1, N + 1):
                print("C[{},{}]=".format(s, j), C[s, j].x)
        print("obj:{}".format(model.objVal))
    else:
        print("no solution")


# In[4]:


# P = [[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 4, 4, 4, 0]]
# HFSP(3, 2, 2, P, 100)


# ## 考虑时间窗

# In[54]:


def HFSP_time(N, M, H, P, Q, D):
    """
    J:订单序列,J=[1,2,..,N]，N:订单总数
    I：I=[1，2，...M],每个stage的机器序列，M：机器总数
    S：stage的序列,S=[1,2,...,H]，H：stage总数

    P：P=[0,0,...,0            第一个是虚拟的stage s0
          0,P11,...P1J,0       前后两列是两个虚拟订单
          0,P21,...P2J,0
          ...
          0,PS1,...PSJ,0],      Psj:订单j在stage s的处理时间

    Q：一个很大的数
    D: D=[0,d1,d2,...,dJ,0], dj: 订单的到期时间，前后两个是虚拟订单

    """
    # 构建订单序列
    J = tuplelist([j + 1 for j in range(N)])
    J_j0 = tuplelist([j for j in range(N + 1)])
    J_jN1 = tuplelist([j + 1 for j in range(N + 1)])
    # 构建机器序列
    I = tuplelist([m for m in range(M)])
    # 构建stage序列
    S = tuplelist([s + 1 for s in range(H)])
    S_s0 = tuplelist([s for s in range(H + 1)])
    # 实例化模型
    model = Model('HFSP')
    # 添加变量
    X = model.addVars(N + 2, N + 2, M, H + 1, vtype=GRB.BINARY, name='X[i,j,m,s]')  # N+2是因为有两个虚拟订单
    C = model.addVars(H + 1, N + 2, lb=0, vtype=GRB.CONTINUOUS, name='C[s,j]')
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
    d = model.addVars(N + 2, vtype=GRB.CONTINUOUS, name="d[j]")
    # IN = model.addVars(N + 2, vtype=GRB.BINARY, name="iN[j]")
    #     print("d",d[2])
    #     print("D",D[2])
    #     print("C",C[H,2])

    # 约束1：每个作业在每个阶段中只由一台机器处理
    for j in J:
        for s in S:
            lhs = LinExpr(0)
            for i in J_j0:
                if i != j:
                    for m in I:
                        lhs.addTerms(1, X[i, j, m, s])
            model.addConstr(lhs == 1, name="iN[j]")

    # model.addConstrs(quicksum(X[i, j, m, s] for i in J_j0 if i != j for m in I) == 1 for j in J for s in S)
    # print(quicksum(X[i, j, m, s] for i in J_j0 if i != j for m in I) == 1 for j in J for s in S)

    # 约束2,3：没有一台机器可以同时处理多个作业

    # model.addConstrs( quicksum(quicksum( X[i,j,m,s] for i in J_j0 if i!=j) - quicksum( X[j,k,m,s] for k in J_jN1 if k!=j ) for m in I)==0 for j in J for s in S )

    for j in J:
        for s in S:
            for m in I:
                lhs = LinExpr(0)
                # for m in I:
                for i in J_j0:
                    if i != j:
                        lhs.addTerms(1, X[i, j, m, s])
                rhs = LinExpr(0)

                for i in J_jN1:
                    if i != j:
                        rhs.addTerms(1, X[j, i, m, s])
                model.addConstr(lhs == rhs, name="iN[j]")

            # print(lhs)

    # model.addConstrs(
    #     quicksum(X[i, j, m, s] for i in J_j0 if i != j) == quicksum(X[j, k, m, s] for k in J_jN1 if k != j) for m in I
    #     for j in J for s in S)

    # model.addConstrs(
    #     quicksum(X[i, j, m, s] for i in J_j0 if i != j for m in I) == quicksum(X[j, k, m, s] for k in J_jN1 if k != j for m in I)
    #     for j in J for s in S)

    # model.addConstrs(
    #     quicksum(X[i, j, m, s] for i in J_j0 if i != j) == quicksum(X[j, k, m, s] for k in J_jN1 if k != j) for m in I
    #     for j in J for s in S)

    for m in I:
        for s in S:
            lhs = LinExpr(0)
            for i in J:
                lhs.addTerms(1, X[0, i, m, s])
            model.addConstr(lhs <= 1, name="iN[j]")

    # model.addConstrs(quicksum(X[0, i, m, s] for i in J) <= 1 for m in I for s in S)

    # 约束4,5：满足优先约束
    for j in J:
        for s in S:
            model.addConstr(C[s, j] >= C[s - 1, j] + P[s][j], name="iN[j]")

    # model.addConstrs(C[s, j] >= C[s - 1, j] + P[s][j] for j in J for s in S)

    model.addConstrs(C[0, j] == 0 for j in J)
    # 约束6：工作完成时间约束

    for s in S:
        for i in J:
            for j in J:
                if i != j:
                    sum = LinExpr(0)
                    for m in I:
                        sum.addTerms(1, X[i, j, m, s])
                    model.addConstr(C[s, j] >= C[s, i] + P[s][j] - Q * (1 - sum), name="iN[j]")

    # model.addConstrs(
    #     C[s, j] >= C[s, i] + P[s][j] - Q * (1 - quicksum(X[i, j, m, s] for m in I)) for i in J for j in J if i != j for
    #     s in S)

    # 获得最大完工时间z
    for j in J:
        for s in S:
            model.addConstr(C[s, j] <= z, name="iN[j]")

    # model.addConstrs(C[s, j] <= z for j in J for s in S)

    # 获得每个订单违反时间窗的时间
    # M = 1000
    # for j in J:
    #     print(D[j], H)
    #     model.addConstr(d[j] - M * IN[j] <= 0)
    #     # model.addConstr(d[j] == C[H, j] - D[j])

    model.addConstrs(d[j] >= C[H, j] - D[j] for j in J)
    model.addConstrs(d[j] >= 0 for j in J)
    # model.addConstr((C[H, j] - D[j] >= 0) >> (d[j] == C[H, j] - D[j]))

    # print(j, D[j])

    #     model.addConstrs(d2==max_(0,(C[H,j]-D[j]),"maxconstr")for j in J)
    #     for j in J:
    #         model.addGenConstrMax(d[j], [0,C[H,j]-D[j]],0,name="maxconstr")
    ####################################################################################################################

    # 目标
    z2 = 1 * z + 0.2 * quicksum(d[j] for j in J)
    model.setObjective(z2, GRB.MINIMIZE)

    "求解"
    model.Params.LogToConsole = True  # 显示求解过程
    # model.Params.TimeLimit = 300  # 设置求解时间上限
    model.optimize()
    # model.write("E:\清华\课程\高级运筹学\大作业")
    if model.status == GRB.Status.OPTIMAL or model.status == GRB.Status.TIME_LIMIT:
        for s in range(H + 1):
            for m in I:
                for i in range(N + 2):
                    for j in range(N + 2):
                        if i != j:
                            if X[i, j, m, s].x > 0:
                                print("X[{},{},{},{}]=1".format(i, j, m, s))
        for s in range(1, H + 1):
            for j in range(1, N + 1):
                print("C[{},{}]=".format(s, j), C[s, j].x)

        for j in range(1, N + 1):
            print("d[{}]=".format(j), d[j].x)
        print("obj:{}".format(model.objVal))
    else:
        print("no solution")

    return model.objVal,model.Runtime


# In[55]:
num = 5  # 任务数量
k = 1
list_obj = []
filelist=[]
timelist=[]
for filename in os.listdir('.\instance\SMALL'):
    if k <= num:
        print(filename)
        filelist.append((filename))
        path = '.\instance\SMALL\%s' % filename
        f = open(path, 'r')
        lines = f.readlines()
        N = 0  # JOB
        H = 0  # STAGE
        count = 0
        D = []
        D.append(0)
        P = []

        for line in lines:
            if line.isspace():
                continue

            # print(line)
            line = line.replace(" ", ",")
            line = line.split(",")
            if count == 0:
                N = int(line[1])
                H = int(line[0])
                print(N, H)
                P = [([0] * (N + 2)) for p in range(H + 1)]
                print(P)
            else:
                D.append(int(line[H]))
                for i in range(1, H + 1):
                    # print(count)
                    P[i][count] = int(line[i - 1])
                    # print(P[1][count])
                    # P[i][count] = int(line[i-1])
            count += 1



        D.append(0)

        M = 3  # 机器数量
        Q = 10000
        obj,time = HFSP_time(N, M, H, P, Q, D)
        list_obj.append(obj)
        timelist.append(time)

        k += 1


print(list_obj)

work2 = xlsxwriter.Workbook(f'.\Results\EX_{filelist[0]}to{filelist[-1]}_total_result.xlsx')
worksheet = work2.add_worksheet()
worksheet.write(0, 0, 'FileName')
worksheet.write(0, 1, 'MaxEnd')
worksheet.write(0, 2, 'RunTime')
for i in range(len(filelist)):
    worksheet.write(i + 1, 0, filelist[i])
    worksheet.write(i + 1, 1, list_obj[i])
    worksheet.write(i + 1, 2, timelist[i])
work2.close()


