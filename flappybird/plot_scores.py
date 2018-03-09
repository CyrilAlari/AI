import numpy
from matplotlib import pyplot as plt

file_to_read="./results/file_score_nf_3691601.txt"
Y=[0]

with open(file_to_read, 'r') as the_file:
    score_old=-1
    for line in the_file:
        score=float(line.split('\n')[0])

        if score == score_old:
            continue
        Y.append(score)
        score_old=score
X=[i*500 for i in range(0,len(Y))]

plt.xlabel('number of games', fontsize=10)
plt.ylabel('average score before cashing', fontsize=10)

plt.plot(X,Y)
plt.show()
