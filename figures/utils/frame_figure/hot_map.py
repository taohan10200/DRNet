import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
def hot_map_P ():

    data = np.load('assign_P.npy', allow_pickle=True)
    data =np.around(data,1)
    id = np.load('id.npy', allow_pickle=True).item()
    h, w = data.shape
    id1, id2 = id['id1'], id['id2']
    idx_row,idx_col  = id1.argsort(), id2.argsort()
    idx_row = np.concatenate([idx_row, np.array([h-1], dtype=np.int)])
    idx_col = np.concatenate([idx_col, np.array([w-1], dtype=np.int)])
    id1.sort()
    id2.sort()
    data = data[idx_row,:]
    data = data[:, idx_col]
    # import pdb
    # pdb.set_trace()
    print()
    f, (ax1) = plt.subplots(figsize=(21, 17), nrows=1)

    # sns.heatmap(data, annot=True, ax=ax1)
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)

    row = list(id1)
    row.append('Out')
    col = list(id2)
    col.append('In')
    print(row)
    data = pd.DataFrame(data, index=row, columns=col)
    sns.heatmap(data, annot=True, linewidths = 0.05, linecolor= 'black',ax=ax1,cmap=cmap, annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'})
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # plt.savefig('cost.png')
    plt.savefig('assign.png')
    plt.show()


def hot_map ():
    import random
    random.seed(40)
    cost = np.load('cost_c_.npy', allow_pickle=True)
    cost =np.around(cost,1)
    id = np.load('id.npy', allow_pickle=True).item()
    h, w = cost.shape
    id1, id2 = id['id1'], id['id2']
    idx_row,idx_col  = id1.argsort(), id2.argsort()

    idx_row = np.concatenate([idx_row, np.array([h-1], dtype=np.int)])
    idx_col = np.concatenate([idx_col, np.array([w-1], dtype=np.int)])
    id1.sort()
    id2.sort()
    cost = cost[idx_row,:]
    cost = cost[:, idx_col]
    cost[:,-1] = 10.2
    cost[-1, :] = 10.2
    sample_num = 7
    idx_row_f = random.sample(range(h-1),sample_num)
    idx_row_f.sort()
    idx_col_f = random.sample(range(w-1), sample_num)
    idx_col_f.sort()
    # idx_row_f = idx_row_f[1:6]
    # idx_col_f = idx_col_f[1:6]
    idx_row_f = [idx_row_f[1],idx_row_f[4],idx_row_f[5]]
    idx_col_f = [idx_col_f[1],idx_col_f[3],idx_col_f[4]]
    cost_f = cost[idx_row_f+[h-1],:] [:,idx_col_f+[w-1]]
    cost_f = torch.from_numpy(cost_f)[None]
    id1, id2 = id1[idx_row_f], id2[idx_col_f]

    # import pdb
    # pdb.set_trace()

    scores = log_optimal_transport(cost_f, iters=100)
    scores = scores.squeeze(0).exp().numpy()
    scores = np.around(scores, 1)
    cost_f = cost_f.squeeze(0).numpy()
    #================================cost_hot_map===========================
    f, (ax1) = plt.subplots(figsize=(sample_num+2, sample_num), nrows=1)
    # sns.heatmap(data, annot=True, ax=ax1)
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    font_size = 40
    row = list(id1)
    row.append('Out')
    col = list(id2)
    col.append('In')
    print(row)
    data = pd.DataFrame(cost_f, index=row, columns=col)
    sns.heatmap(data, annot=True, linewidths = 0.05, linecolor= 'black',ax=ax1,cmap=cmap, annot_kws={'size': font_size,  'color': 'black'})#'weight': 'bold',
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # f.set_facecolor('none')  # 设置图例legend背景透明
    # plt.savefig('cost.png')

    plt.savefig('cost.png',transparent=True,bbox_inches='tight', pad_inches=0.0)
    plt.show()
    plt.close()

    print(scores)
    #================================assignment_matrix_hot_map===========================
    f, (ax2) = plt.subplots(figsize=(sample_num+2, sample_num), nrows=1)

    # sns.heatmap(data, annot=True, ax=ax1)
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)

    row = list(id1)
    row.append('Out')
    col = list(id2)
    col.append('In')
    print(row)
    data = pd.DataFrame(scores, index=row, columns=col)
    sns.heatmap(data, annot=True, linewidths = 0.05, linecolor= 'black',ax=ax2,cmap=cmap, annot_kws={'size': font_size, 'color': 'black'}) #'weight': 'bold'
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # plt.savefig('cost.png')
    plt.savefig('assign.png',transparent=True,bbox_inches='tight', pad_inches=0.0)
    plt.show()


import torch
def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""

    log_u, log_v = torch.zeros_like(log_mu), torch.zeros_like(log_nu) # initialized with the u,v=1, the log(u)=0, log(v)=0
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(Z + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(Z + log_u.unsqueeze(2), dim=1)
    # import pdb
    # pdb.set_trace()
    return Z + log_u.unsqueeze(2) + log_v.unsqueeze(1)


def log_optimal_transport(scores, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    m, n = m-1, n-1
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)



    couplings = scores
    # import numpy as np
    # np.save('cost_c_', couplings.detach().squeeze(0).cpu().numpy(), allow_pickle=True, fix_imports=True)

    norm = - (ms + ns).log() # normalization in the Log-space (log(1/(m+n)))
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    score = Z - norm  # multiply probabilities by M+N
    return score


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # trac
if __name__ == '__main__':
    hot_map()