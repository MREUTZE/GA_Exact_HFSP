import random
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import xlsxwriter
from predata import *
import matplotlib.colors as mcolors

class Item:
    def __init__(self):
        self.start = []
        self.end = []
        self._on = []
        self.T = []
        self.last_ot = 0
        self.L = 0

    def update(self, s, e, on, t):
        self.start.append(s)
        self.end.append(e)
        self._on.append(on)
        self.T.append(t)
        self.last_ot = e
        self.L += t

class Scheduling:
    def __init__(self, J_num, Machine, State, PT):
        self.M = Machine
        self.J_num = J_num
        self.State = State
        self.PT = PT
        self.Create_Job()
        self.Create_Machine()
        self.fitness = 0

    def Create_Job(self):
        self.Jobs = []
        for i in range(self.J_num):
            J = Item()
            self.Jobs.append(J)

    def Create_Machine(self):
        self.Machines = []
        for i in range(len(self.M)):  # 突出机器的阶段性，即各阶段有哪些机器
            State_i = []
            for j in range(self.M[i]):
                M = Item()
                State_i.append(M)
            self.Machines.append(State_i)

    # 每个阶段的解码
    def Stage_Decode(self, CHS, Stage):
        for i in CHS:
            last_od = self.Jobs[i].last_ot
            last_Md = [self.Machines[Stage][M_i].last_ot for M_i in range(self.M[Stage])]  # 机器的完成时间
            last_ML = [self.Machines[Stage][M_i].L for M_i in range(self.M[Stage])]  # 机器的负载
            M_time = [self.PT[Stage][M_i][i] for M_i in range(self.M[Stage])]  # 机器对当前工序的加工时间
            O_et = [last_Md[_] + M_time[_] for _ in range(self.M[Stage])]
            if O_et.count(min(O_et)) > 1 and last_ML.count(last_ML) > 1:
                Machine = random.randint(0, self.M[Stage])
            elif O_et.count(min(O_et)) > 1 and last_ML.count(last_ML) < 1:
                Machine = last_ML.index(min(last_ML))
            else:
                Machine = O_et.index(min(O_et))
            if Jobdue[i] < max(last_od, last_Md[Machine]) + M_time[Machine]:
                s, e, f, t = max(last_od, last_Md[Machine]), max(last_od, last_Md[Machine]) + M_time[Machine], max(last_od, last_Md[Machine]) + M_time[Machine] + a * (max(last_od, last_Md[Machine]) + M_time[Machine] - Jobdue[i]), M_time[
                                 Machine]
            else:
                s, e, f, t = max(last_od, last_Md[Machine]), max(last_od, last_Md[Machine]) + M_time[Machine], max(last_od, last_Md[Machine]) + M_time[Machine], M_time[Machine]
            self.Jobs[i].update(s, e, Machine, t)
            self.Machines[Stage][Machine].update(s, e, i, t)
            if objmode == 1:
                if f > self.fitness:
                    self.fitness = f
            elif objmode == 2:
                if e > self.fitness:
                    self.fitness = e
        #print(self.fitness)

    # 解码
    def Decode(self, CHS):
        for i in range(self.State):
            self.Stage_Decode(CHS, i)
            Job_end = [self.Jobs[i].last_ot for i in range(self.J_num)]
            CHS = sorted(range(len(Job_end)), key=lambda k: Job_end[k], reverse=False)

    # 画甘特图
    def Gantt(self):
        fig = plt.figure()
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        # M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
        #      'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
        #      'navajowhite','navy', 'sandybrown']
        colors=list(mcolors.XKCD_COLORS.keys())
        M_num = 0
        MaxEnd = 0
        for i in range(len(self.M)):
            for j in range(self.M[i]):
                for k in range(len(self.Machines[i][j].start)):
                    Start_time = self.Machines[i][j].start[k]
                    End_time = self.Machines[i][j].end[k]
                    Job = self.Machines[i][j]._on[k]
                    plt.barh(M_num, width=End_time - Start_time, height=0.8, left=Start_time, color=colors[Job],edgecolor='black')
                    plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num - 0.2,
                             s=Job + 1, size=12, fontproperties='Times New Roman')
                    if MaxEnd <= End_time:
                        MaxEnd = End_time
                    if i == len(self.M) - 1:
                        # print(f'阶段{i+1}机器{j+1},工序{Job+1},开始时间{Start_time}结束时间{End_time}')
                        if Jobdue[Job] < End_time:
                            print(
                                f'工序{Job + 1}在机器{j + 1}不满足时间约束，结束时间{End_time},时间窗为{Jobdue[Job]},违背时间{End_time - Jobdue[Job]}')
                        else:
                            print(f'工序{Job + 1}在机器{j + 1}满足时间窗结束时间{End_time}')
                M_num += 1
        print(f'最后完工时间{MaxEnd}')
        objlist.append(MaxEnd)
        end=time.perf_counter()
        run_time=end-start
        timelist.append(run_time)
        plt.yticks(np.arange(M_num + 1), np.arange(1, M_num + 2), size=7, fontproperties='Times New Roman')

        plt.ylabel("机器", size=20, fontproperties='SimSun')
        plt.xlabel("时间", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.savefig(f'.\Results\GA_{FileName}_Gantt.png')
        plt.show()

        work = xlsxwriter.Workbook(f'.\Results\GA_{FileName}_result.xlsx')
        worksheet = work.add_worksheet()
        worksheet.write(0, 0, 'MinCost')
        worksheet.write(0, 1, 'Run_Time')
        worksheet.write(1, 1, run_time)
        worksheet.write(1, 0, MaxEnd)
        worksheet.write(2, 0, 'Stage')
        worksheet.write(2, 1, 'Machine ID')
        worksheet.write(2, 2, 'Produce Plan')
        worksheet.write(2, 3, 'Start Time')
        worksheet.write(2, 4, 'End Time')
        worksheet.write(2, 5, 'Violation Time')
        Count = 0
        for i in range(len(self.M)):
            for j in range(self.M[i]):
                for k in range(len(self.Machines[i][j].start)):
                    Start_time = self.Machines[i][j].start[k]
                    End_time = self.Machines[i][j].end[k]
                    Job = self.Machines[i][j]._on[k]
                    worksheet.write(Count + 3, 0, f'Stage{i + 1}')
                    worksheet.write(Count + 3, 1, j + 1)
                    worksheet.write(Count + 3, 2, Job + 1)
                    worksheet.write(Count + 3, 3, Start_time)
                    worksheet.write(Count + 3, 4, End_time)
                    if End_time - Jobdue[Job] > 0:
                        worksheet.write(Count + 3, 5, End_time - Jobdue[Job])
                    else:
                        worksheet.write(Count + 3, 5, 0)
                    Count = Count + 1
        work.close()





class GA:
    def __init__(self,J_num,State,Machine,PT):
        self.State=State
        self.Machine=Machine
        self.PT=PT
        self.J_num=J_num
        self.Pm=0.3
        self.Pc=0.9
        self.Pop_size=100  #种群数量
        self.Gen=10   #迭代次数

    # 随机产生染色体
    def RCH(self):
        Chromo = [i for i in range(self.J_num)]
        random.shuffle(Chromo)
        return Chromo

    # 生成初始种群
    def CHS(self):
        CHS = []
        for i in range(self.Pop_size):
            CHS.append(self.RCH())
        return CHS

    #选择
    def Select(self, Fit_value):
        Fit = []
        for i in range(len(Fit_value)):
            fit = 1 / Fit_value[i]
            Fit.append(fit)
        Fit = np.array(Fit)
        idx = np.random.choice(np.arange(len(Fit_value)), size=len(Fit_value), replace=True,
                               p=(Fit) / (Fit.sum()))
        return idx

    # 交叉
    def Crossover(self, CHS1, CHS2):
        T_r = [j for j in range(self.J_num)]
        r = random.randint(2, self.J_num)  # 在区间[1,T0]内产生一个整数r
        random.shuffle(T_r)
        R = T_r[0:r]  # 按照随机数r产生r个互不相等的整数
        # 将父代的染色体复制到子代中去，保持他们的顺序和位置
        H1=[CHS1[_] for _ in R]
        H2=[CHS2[_] for _ in R]
        C1=[_ for _ in CHS1 if _ not in H2]
        C2=[_ for _ in CHS2 if _ not in H1]
        CHS1,CHS2=[],[]
        k,m=0,0
        for i in range(self.J_num):
            if i not in R:
                CHS1.append(C1[k])
                CHS2.append(C2[k])
                k+=1
            else:
                CHS1.append(H2[m])
                CHS2.append(H1[m])
                m+=1
        return CHS1, CHS2

    # 变异
    def Mutation(self, CHS):
        Tr = [i_num for i_num in range(self.J_num)]
        # 机器选择部分
        r = random.randint(1, self.J_num)  # 在变异染色体中选择r个位置
        random.shuffle(Tr)
        T_r = Tr[0:r]
        K=[]
        for i in T_r:
            K.append(CHS[i])
        random.shuffle(K)
        k=0
        for i in T_r:
            CHS[i]=K[k]
            k+=1
        return CHS

    def main(self):
        BF=[]
        x=[_ for _ in range(self.Gen+1)]
        C=self.CHS()
        Fit=[]
        for C_i in C:
            s=Scheduling(self.J_num,self.Machine,self.State,self.PT)
            s.Decode(C_i)
            Fit.append(s.fitness)
        best_C = None
        best_fit=min(Fit)
        BF.append(best_fit)
        for k in range(self.Gen):
            for i in range(self.Pop_size):
                C_id=self.Select(Fit)
                C=[C[_] for _ in C_id]
                for Ci in range(len(C)):
                    if random.random()<self.Pc:
                        _C=[C[Ci]]
                        CHS1,CHS2=self.Crossover(C[Ci],random.choice(C))
                        _C.extend([CHS1,CHS2])
                        Fi=[]
                        for ic in _C:
                            s = Scheduling(self.J_num, self.Machine, self.State, self.PT)
                            s.Decode(ic)
                            Fi.append(s.fitness)
                        C[Ci]=_C[Fi.index(min(Fi))]
                        Fit.append(min(Fi))
                    elif random.random()<self.Pm:
                        CHS1=self.Mutation(C[Ci])
                        C[Ci]=CHS1
                Fit = []
                Sc=[]
                for C_i in C:
                    s = Scheduling(self.J_num, self.Machine, self.State, self.PT)
                    s.Decode(C_i)
                    Sc.append(s)
                    Fit.append(s.fitness)
                if min(Fit)<=best_fit:
                    best_fit=min(Fit)
                    best_C=Sc[Fit.index(min(Fit))]
            BF.append(best_fit)
            print(f'当前迭代次数{k+1}/{self.Gen},目前最优适应度{best_fit}')
        plt.plot(x,BF)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.xticks(range(0,self.Gen+1,1))
        plt.savefig(f'.\Results\GA_{FileName}_Iter.png')
        plt.show()
        best_C.Gantt()


if __name__=="__main__":
    k=1
    num=1 #任务数量
    a = 0.2  #惩罚系数
    objmode = 1 #objmode=1 则考虑时间窗, objmode=2 则不考虑时间窗
    objlist=[]
    timelist=[]
    filelist=[]
    tasktype = 'BIG'
    for FileName in os.listdir(f'.\instance\{tasktype}'):
        if k <=num:
            Machine = []
            print(FileName)
            filelist.append(FileName)
            readtxtfile(FileName,tasktype)
            templist = readcsv(FileName)
            Job = int(templist[0][1])
            State = int(templist[0][0])
            Jobdue = []
            for i in range(State):
                Machine.append(3)
            for i in range(Job):
                Jobdue.append(int(templist[i + 1][State]))
            PT = Generate(State, Job, Machine, templist)
            start=time.perf_counter()
            g=GA(Job,State,Machine,PT)
            g.main()
            k=k+1

    work2 = xlsxwriter.Workbook(f'.\Results\GA_{filelist[0]}to{filelist[-1]}_total_result.xlsx')
    worksheet = work2.add_worksheet()
    worksheet.write(0, 0, 'FileName')
    worksheet.write(0, 1, 'MaxEnd')
    worksheet.write(0, 2, 'RunTime')
    for i in range(len(filelist)):
        worksheet.write(i + 1, 0, filelist[i])
        worksheet.write(i + 1, 1, objlist[i])
        worksheet.write(i + 1, 2, timelist[i])
    work2.close()