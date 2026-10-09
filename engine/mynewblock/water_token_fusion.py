"""Token-native RWA, GSQWCA and CQWG. No spatial reshaping or convolutions."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RankAwareWaterAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, init_scale=0.01, router_hidden=16, eps=1e-6):
        super().__init__()
        if dim % num_heads or dim // num_heads < 2:
            raise ValueError('RWA requires dim divisible by heads and head_dim >= 2')
        self.num_heads, self.head_dim, self.eps = num_heads, dim // num_heads, eps
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(dim, dim) for _ in range(3))
        self.relation = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.router = nn.Sequential(nn.Linear(2, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        self.out_proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))
        self.save_attention = False
        self.last_attention = self.last_rank_stats = self.last_effective_rank = None
        self.last_relation_strength = None

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def _rank_statistics(self, y):
        centered = y - y.mean(dim=-2, keepdim=True)
        covariance = centered.transpose(-2, -1) @ centered / max(y.shape[-2] - 1, 1)
        diag = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        total_energy = covariance.square().sum(dim=(-2, -1))
        redundancy = ((total_energy - diag.square().sum(-1)) / (total_energy + self.eps)).clamp(0, 1)
        energy = diag.sum(-1, keepdim=True)
        probabilities = diag / (energy + self.eps)
        entropy = -(probabilities * (probabilities + self.eps).log()).sum(-1)
        concentration = (1 - entropy / math.log(self.head_dim)).clamp(0, 1)
        concentration = torch.where(energy.squeeze(-1) > self.eps, concentration, torch.zeros_like(concentration))
        return torch.stack((redundancy, concentration), dim=-1), centered

    def forward(self, water_tokens):
        if water_tokens.ndim != 3 or water_tokens.shape[1] != 3:
            raise ValueError('RWA expects [B,3,Dw] temperature/DO/pH tokens')
        self.last_attention = self.last_rank_stats = self.last_effective_rank = None
        self.last_relation_strength = None
        q, k, v = (self._heads(proj(water_tokens)).float() for proj in (self.q_proj, self.k_proj, self.v_proj))
        with torch.autocast(device_type=water_tokens.device.type, enabled=False):
            q, k = F.elu(q) + 1, F.elu(k) + 1
            n = water_tokens.shape[1]
            kv = k.transpose(-2, -1) @ v / n
            denominator = (q * k.mean(dim=-2, keepdim=True)).sum(-1, keepdim=True)
            global_context = (q @ kv) / denominator.clamp_min(self.eps)
            stats, centered = self._rank_statistics(global_context)
        relation_strength = self.router(stats.detach()).sigmoid()[..., None]
        relation = self._heads(self.relation(water_tokens)).float()
        fused = global_context + relation_strength * relation
        fused = fused.transpose(1, 2).reshape_as(water_tokens).to(water_tokens.dtype)
        delta = self.out_proj(fused)
        result = water_tokens + self.alpha * delta
        if self.save_attention:
            with torch.no_grad(), torch.autocast(device_type=water_tokens.device.type, enabled=False):
                weights = q @ k.transpose(-2, -1)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(self.eps)
                self.last_attention = weights.mean(1).detach().cpu()
                self.last_rank_stats = stats.detach().cpu()
                self.last_relation_strength = relation_strength.detach().cpu()
                eigenvalues = torch.linalg.eigvalsh(centered @ centered.transpose(-2, -1) / (n - 1)).clamp_min(0)
                energy = eigenvalues.sum(-1, keepdim=True)
                p = eigenvalues / energy.clamp_min(self.eps)
                rank = (-(p * p.clamp_min(self.eps).log()).sum(-1)).exp()
                self.last_effective_rank = torch.where(energy.squeeze(-1) > self.eps, rank, torch.zeros_like(rank)).cpu()
        return result


class GlobalSparseQueryWaterCrossAttention(nn.Module):
    """Dense + sparse water contexts, fused by a router for each object query."""
    def __init__(self, query_dim=128, water_dim=64, num_heads=8, sparse_k=2, dropout=0.0):
        super().__init__()
        if num_heads < 1 or query_dim % num_heads:
            raise ValueError('GSQWCA query_dim must be divisible by positive num_heads')
        if isinstance(sparse_k, bool) or not isinstance(sparse_k, int) or sparse_k not in (1, 2):
            raise ValueError('Three water tokens require sparse_k=1 or 2; default is 2')
        self.num_heads, self.head_dim = num_heads, query_dim // num_heads
        self.sparse_k = sparse_k
        self.water_proj = nn.Linear(water_dim, query_dim)
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj, self.v_proj = nn.Linear(query_dim, query_dim), nn.Linear(query_dim, query_dim)
        self.branch_router = nn.Sequential(nn.Linear(3 * query_dim, query_dim), nn.SiLU(), nn.Linear(query_dim, 2))
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)
        self.out_proj, self.dropout = nn.Linear(query_dim, query_dim), nn.Dropout(dropout)
        self.save_attention = False
        self.last_attention = self.last_branch_attention = None
        self.last_global_attention = self.last_sparse_attention = self.last_branch_weights = None
        self.last_sparse_mask = None

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, water_tokens):
        if query.ndim != 3 or water_tokens.ndim != 3 or water_tokens.shape[1] != 3:
            raise ValueError('GSQWCA expects query [B,N,Dq] and water [B,3,Dw]')
        if query.shape[0] != water_tokens.shape[0]:
            raise ValueError('Query and water batch sizes must match')
        self.last_attention = self.last_branch_attention = None
        self.last_global_attention = self.last_sparse_attention = self.last_branch_weights = None
        self.last_sparse_mask = None
        water = self.water_proj(water_tokens)
        q, k, v = self._heads(self.q_proj(query)), self._heads(self.k_proj(water)), self._heads(self.v_proj(water))
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.head_dim)
            global_attn = scores.softmax(-1)
            indices = scores.topk(self.sparse_k, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
            sparse_attn = scores.masked_fill(~mask, float('-inf')).softmax(-1)
            global_context = global_attn @ v.float()
            sparse_context = sparse_attn @ v.float()
        global_context = global_context.transpose(1, 2).reshape_as(query).to(query.dtype)
        sparse_context = sparse_context.transpose(1, 2).reshape_as(query).to(query.dtype)
        router_input = torch.cat((query, global_context, sparse_context), dim=-1)
        logits = self.branch_router(router_input)
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = logits.float().softmax(-1)
            context = weights[..., 0:1] * global_context.float() + weights[..., 1:2] * sparse_context.float()
        context = context.to(query.dtype)
        if self.save_attention:
            attention = weights[:, None, :, 0:1] * global_attn + weights[:, None, :, 1:2] * sparse_attn
            self.last_attention = attention.mean(1).detach().cpu()
            self.last_global_attention = global_attn.mean(1).detach().cpu()
            self.last_sparse_attention = sparse_attn.mean(1).detach().cpu()
            self.last_branch_weights = weights.detach().cpu()
            self.last_sparse_mask = mask.detach().cpu()
            self.last_branch_attention = (global_attn.detach().cpu(), sparse_attn.detach().cpu())
        return self.dropout(self.out_proj(context))


class MultiSparseQueryWaterCrossAttention(GlobalSparseQueryWaterCrossAttention):
    """Legacy constructor name; execution now uses Global + Sparse, not Top-1 + Top-2.

    Old topk=[1,2] configurations select the new default sparse_k=2. Old trained
    branch_logits weights are not equivalent to the query-adaptive router.
    """
    def __init__(self, query_dim=128, water_dim=64, num_heads=8, topk=(1, 2), dropout=0.0, *, sparse_k=2):
        if tuple(topk) != (1, 2):
            raise ValueError('Legacy topk must be [1,2]; use sparse_k=1 or 2 with GSQWCA')
        super().__init__(query_dim, water_dim, num_heads, sparse_k, dropout)


class ComplementaryQueryWaterGate(nn.Module):
    def __init__(self, dim=128, init_scale=0.01):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(4 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.lambda_gate = nn.Parameter(torch.tensor(float(init_scale)))
        self.save_attention = False
        self.last_gate = self.last_gate_mean = None

    def forward(self, query, water_context):
        if query.ndim != 3 or query.shape != water_context.shape:
            raise ValueError('CQWG inputs must have the same [B,N,Dq] shape')
        inputs = torch.cat((query, water_context, query * water_context, (query - water_context).abs()), dim=-1)
        gate = self.gate_net(inputs).sigmoid()
        # Normalize Q_mix-Q inside the residual, preserving identity at lambda=0.
        delta = self.norm(gate * (water_context - query))
        self.last_gate = self.last_gate_mean = None
        if self.save_attention:
            self.last_gate = gate.detach().cpu()
            self.last_gate_mean = gate.detach().float().mean().cpu()
        return query + self.lambda_gate * delta
