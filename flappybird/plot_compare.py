import numpy
from matplotlib import pyplot as plt

file1="./results/file_score_nf_6901837.txt"
file2="./results/file_score_nf_1601829.txt"

Y1=[0]
Y2=[0]

with open(file1, 'r') as the_file:
    score_old=-1
    for line in the_file:
        score=float(line.split('\n')[0])
        if score == score_old:
            continue
        Y1.append(score)
        score_old=score

with open(file2, 'r') as the_file:
    score_old=-1
    for line in the_file:
        score=float(line.split('\n')[0])
        if score == score_old:
            continue
        Y2.append(score)
        score_old=score


len = min(len(Y1),len(Y2))
X=[i*500 for i in range(0,len)]
Y1=Y1[0:len]
Y2=Y2[0:len]


plt.plot(X,Y1,'r')
plt.plot(X,Y2,'g')
plt.show()
