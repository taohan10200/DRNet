import os
import os.path as osp
import json

root = 'T:\CVPR2022'
with open('info.json') as f:
    info = json.load(f)
cat = ['0~50', '50~100', '100~150', '150~200', '200~400']
# for k, v in a.items():
#     data.append(v)
#     if v in range(0,50):
#         number[0]+=1
#     elif v in range(50,100):
#         number[1]+=1
#     elif v in range(100,150):
#         number[2]+=1
#     elif v in range(150, 200):
#         number[3] += 1
#     elif v in range(200, 400):
#         number[4] += 1
with open(osp.join(root, 'new_label.txt'),'r') as f:
    lines = f.readlines()
new_lines = []
for i in lines:
    i = i.rstrip()
    scene_name  = i.split(' ')[0]
    v = info[scene_name]

    if v in range(0,50):
        density_label =0
    elif v in range(50,100):
        density_label = 1
    elif v in range(100,150):
        density_label = 2
    elif v in range(150, 200):
        density_label = 3
    elif v in range(200, 400):
        density_label = 4
    new_i = i+' ' +str(density_label)+ '\n'
    new_lines.append(new_i)
with open(osp.join(root,'scene_label.txt'), 'w') as f:

    f.writelines(new_lines)
