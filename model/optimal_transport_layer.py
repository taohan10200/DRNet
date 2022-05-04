import torch
import torch.nn as nn

class Optimal_Transport_Layer(nn.Module):
    def __init__(self, config):
        super(Optimal_Transport_Layer, self).__init__()
        self.iters =config['sinkhorn_iterations']
        self.feature_dim = config['feature_dim']
        self.matched_threshold = config['matched_threshold']
        self.bin_score = torch.nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.register_parameter('bin_score', self.bin_score)
    @property
    def loss(self):
        return self.matching_loss, self.hard_pair_loss
    def forward(self,mdesc0, mdesc1, match_gt=None, ignore =False):
        # Compute matching descriptor distance.
        sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)

        scores = sim_matrix / self.feature_dim ** .5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.iters)

        # Get the matches with score above "match_threshold".
        max0 = scores[:, :-1, :-1].max(2)  # the points in a that have matched in b, return b's index,
        max1 = scores[:, :-1, :-1].max(1)  # the points in b that have matched in b, return a's index
        indices0, indices1 = max0.indices, max1.indices

        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

        valid0 = mutual0 & (mscores0 > self.matched_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))


        scores = scores.squeeze(0).exp()

        if match_gt is not None:
            matched_mask = torch.zeros(scores.size()).long().to(scores)

            matched_mask[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]] = 1
            if not ignore: matched_mask[match_gt['un_a'], -1] = 1
            if not ignore: matched_mask[-1, match_gt['un_b']] = 1

            self.matching_loss = -torch.log(scores[matched_mask == 1])

            top2_mask = matched_mask[:-1, :-1]
            scores_ = scores[:-1, :-1]* (1 - top2_mask)
            self.hard_pair_loss = -(torch.log(1- torch.cat([scores_.max(1)[0], scores_.max(0)[0]])))

        return scores, indices0.squeeze(0), indices1.squeeze(0), mscores0.squeeze(0), mscores1.squeeze(0)

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""

    log_u, log_v = torch.zeros_like(log_mu), torch.zeros_like(log_nu) # initialized with the u,v=1, the log(u)=0, log(v)=0
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(Z + log_v.unsqueeze(1), dim=2)
        log_v = log_nu - torch.logsumexp(Z + log_u.unsqueeze(2), dim=1)

    return Z + log_u.unsqueeze(2) + log_v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() # normalization in the Log-space (log(1/(m+n)))
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    score = Z - norm  # multiply probabilities by M+N
    return score


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1