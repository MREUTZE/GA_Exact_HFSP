import csv
import os


def readtxtfile(FileName,tasktype):
    tasktype=tasktype
    dirpath = os.path.dirname(os.path.realpath('__file__'))
    DetailFileName = os.path.join(dirpath, f'instance\{tasktype}', FileName)
    file = open(DetailFileName, 'r', encoding='UTF-8')
    csvFile = open(f'./dataset/{FileName}.csv', 'w', newline='', encoding='UTF-8')
    csvRow = []
    writer = csv.writer(csvFile)
    count = 1
    for line in file:
        csvRow = line.split()
        writer.writerow(csvRow)
    csvFile.close()
    file.close()


def readcsv(FileName):
    with open(f'./dataset/{FileName}.csv','r') as f:
        DetailCsv=csv.reader(f)
        templist=[]
        for row in DetailCsv:
            templist.append(row)
        return templist


def Generate(State,Job,Machine,templist):
    PT = []
    for i in range(State):
        Si=[]       #第i各加工阶段
        for j in range(Machine[i]):
            S0=[]
            for k in range(Job):
                S0.append(int(templist[k+1][i]))
            Si.append(S0)
        PT.append(Si)
    return PT


