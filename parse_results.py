import numpy as np
from linear_py import analysis

dataset = 'results/hep_th/'
alg = 'trace_exact/'
file = dataset+alg+'result.txt'

lines = []
with open(file, 'r') as f:
    for l in f:
        lines.append(l)

lines = lines[1:]
for i in range(len(lines)):
    lines[i] = lines[i][1:-2]

for i in range(len(lines)):
    lines[i] = [float(s) for s in lines[i].split(",")]

p_count_dict = {0.1:[], 0.3:[], 0.5:[], 0.7:[], 1.0:[]}
p_time_dict = {0.1:[], 0.3:[], 0.5:[], 0.7:[], 1.0:[]}

for l in lines:
    p_count_dict[l[1]].append(l[2])
    p_time_dict[l[1]].append(l[3])

res = []

for p in [0.1, 0.3, 0.5, 0.7, 1.0]:
    res.append((p, np.mean(p_count_dict[p]), np.mean(p_time_dict[p])))

analysis(res, (res[-1][1], res[-1][2]), dataset+alg+"result.png")