"""Three-sensor token encoding and configurable query-water fusion."""
import torch
from torch import nn
from ..water_schema import WATER_FEATURES, WATER_MEANS, WATER_STDS
from .water_token_fusion import RankAwareWaterAttention, GlobalSparseQueryWaterCrossAttention, MultiSparseQueryWaterCrossAttention, ComplementaryQueryWaterGate


def validate_water_features(features=None):
    features = list(WATER_FEATURES if features is None else features)
    if not features or len(set(features)) != len(features):
        raise ValueError('water_features must be nonempty and unique')
    if any(name not in WATER_FEATURES for name in features):
        raise ValueError(f'water_features must be chosen from {WATER_FEATURES}')
    return features


class WaterTokenEncoder(nn.Module):
    def __init__(self, num_params=None, embed_dim=64, num_heads=4,
                 means=WATER_MEANS, stds=WATER_STDS, dropout=0.0,
                 water_features=None, water_self_attn=True, use_rwa=False,
                 water_relation=None, debug=False):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('water_dim must be divisible by water heads')
        self.water_features = validate_water_features(water_features)
        self.num_params = len(self.water_features)
        if num_params is not None and num_params != self.num_params:
            raise ValueError('num_params must equal len(water_features)')
        indices = [WATER_FEATURES.index(name) for name in self.water_features]
        means, stds = torch.as_tensor(means, dtype=torch.float32), torch.as_tensor(stds, dtype=torch.float32)
        if means.shape != (3,) or stds.shape != (3,):
            raise ValueError('water_means/water_stds require three values: temperature, do, ph')
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or (stds <= 0).any():
            raise ValueError('water statistics must be finite, with strictly positive stds')
        if use_rwa and self.water_features != list(WATER_FEATURES):
            raise ValueError('RWA requires all three parameters in temperature, do, ph order')
        self.register_buffer('feature_indices', torch.tensor(indices), persistent=False)
        self.register_buffer('mean', means[indices].clone())
        self.register_buffer('std', stds[indices].clone())
        self.value_encoder = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, embed_dim))
        self.param_embedding = nn.Parameter(torch.empty(self.num_params, embed_dim))
        nn.init.trunc_normal_(self.param_embedding, std=0.02)
        self.use_rwa = bool(use_rwa)
        self.use_water_self_attn = bool(water_self_attn and not use_rwa)
        self.save_attention, self.last_water_self_attn = False, None
        self.debug, self._debug_printed = debug, False
        if use_rwa:
            options = dict(water_relation or {})
            kind = options.pop('type', 'RankAwareWaterAttention')
            if kind != 'RankAwareWaterAttention':
                raise ValueError(f'Unsupported water_relation type: {kind}')
            options.setdefault('num_heads', num_heads)
            self.relation = RankAwareWaterAttention(dim=embed_dim, **options)
        self.norm = nn.LayerNorm(embed_dim) if self.use_water_self_attn else None
        self.water_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) if self.use_water_self_attn else None
        self.delta_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)) if self.use_water_self_attn else None
        self.register_parameter('interaction_scale', nn.Parameter(torch.zeros(1)) if self.use_water_self_attn else None)

    def select_features(self, water):
        if water.ndim != 2 or water.shape[1] != 3:
            raise ValueError('Raw water must have shape [B,3] in temperature, do, ph order')
        return water.index_select(1, self.feature_indices)

    def forward(self, water):
        self.last_water_self_attn = None
        raw_shape = tuple(water.shape)
        values = self.select_features(water).to(self.mean.dtype)
        values = torch.nan_to_num((values - self.mean) / self.std, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5, 5)
        tokens = self.value_encoder(values.unsqueeze(-1)) + self.param_embedding.unsqueeze(0)
        token_shape = tuple(tokens.shape)
        if self.use_rwa:
            self.relation.save_attention = self.save_attention
            tokens = self.relation(tokens)
            self.last_water_self_attn = self.relation.last_attention
        elif self.use_water_self_attn:
            normalized = self.norm(tokens)
            context, weights = self.water_self_attn(normalized, normalized, normalized, need_weights=self.save_attention)
            tokens = tokens + self.interaction_scale * self.delta_proj(context)
            if self.save_attention:
                self.last_water_self_attn = weights.detach().cpu()
        if self.debug and not self._debug_printed:
            print(f'water raw: {raw_shape}; water tokens: {token_shape}; enhanced water tokens: {tuple(tokens.shape)}')
            self._debug_printed = True
        return tokens


class WaterAwareQueryFusion(nn.Module):
    """Each decoder layer consumes the shared, already encoded water tokens."""
    def __init__(self, query_dim=128, water_dim=64, num_heads=8, dropout=0.0,
                 ffn_ratio=4.0, init_scale=0.0, learnable_gate=True,
                 use_msqwca=False, use_cqwg=False, query_water_fusion=None,
                 query_water_gate=None, debug=False):
        super().__init__()
        self.query_norm, self.water_norm = nn.LayerNorm(query_dim), nn.LayerNorm(water_dim)
        self.use_msqwca, self.use_cqwg = bool(use_msqwca), bool(use_cqwg)
        self.learnable_gate = bool(learnable_gate)
        self.save_attention, self.last_query_water_attn = False, None
        self.debug, self._debug_printed = debug, False
        if use_msqwca:
            options = dict(query_water_fusion or {})
            kind = options.pop('type', 'GlobalSparseQueryWaterCrossAttention')
            classes = {'GlobalSparseQueryWaterCrossAttention': GlobalSparseQueryWaterCrossAttention,
                       'MultiSparseQueryWaterCrossAttention': MultiSparseQueryWaterCrossAttention}
            if kind not in classes:
                raise ValueError(f'Unsupported query_water_fusion type: {kind}')
            options.setdefault('num_heads', num_heads)
            options.setdefault('dropout', dropout)
            self.cross_attn = classes[kind](query_dim=query_dim, water_dim=water_dim, **options)
        else:
            self.cross_attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True, kdim=water_dim, vdim=water_dim)
        if use_cqwg:
            options = dict(query_water_gate or {})
            kind = options.pop('type', 'ComplementaryQueryWaterGate')
            if kind != 'ComplementaryQueryWaterGate':
                raise ValueError(f'Unsupported query_water_gate type: {kind}')
            self.gate = ComplementaryQueryWaterGate(dim=query_dim, **options)
        else:
            self.attn_scale = nn.Parameter(torch.tensor([float(init_scale)])) if learnable_gate else 1.0
            self.ffn_scale = nn.Parameter(torch.tensor([float(init_scale)])) if learnable_gate else 1.0
            hidden = int(query_dim * ffn_ratio)
            self.ffn = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, query_dim), nn.Dropout(dropout))

    def forward(self, query, water_tokens, return_attn=False):
        self.last_query_water_attn = None
        q, w = self.query_norm(query), self.water_norm(water_tokens)
        if self.use_msqwca:
            self.cross_attn.save_attention = self.save_attention or return_attn
            context = self.cross_attn(q, w)
            attention = self.cross_attn.last_attention
        else:
            context, attention = self.cross_attn(q, w, w, need_weights=self.save_attention or return_attn)
        if self.save_attention and attention is not None:
            self.last_query_water_attn = attention.detach().cpu()
        if self.use_cqwg:
            self.gate.save_attention = self.save_attention
            result = self.gate(query, context)
        else:
            result = query + self.attn_scale * context
            result = result + self.ffn_scale * self.ffn(result)
        if self.debug and not self._debug_printed:
            print(f'query: {tuple(query.shape)}; water context: {tuple(context.shape)}; updated query: {tuple(result.shape)}')
            self._debug_printed = True
        return (result, attention) if return_attn else result


class WaterAwareQueryCrossAttention(WaterAwareQueryFusion):
    """Standalone legacy API. Decoder layers instead share WaterTokenEncoder."""
    def __init__(self, query_dim=112, water_dim=64, water_heads=4, num_heads=8,
                 dropout=0.0, ffn_ratio=4.0, init_scale=0.0, water_features=None,
                 water_means=WATER_MEANS, water_stds=WATER_STDS, water_self_attn=True,
                 learnable_gate=True, use_rwa=False, use_msqwca=False, use_cqwg=False,
                 water_relation=None, query_water_fusion=None, query_water_gate=None, debug=False):
        super().__init__(query_dim, water_dim, num_heads, dropout, ffn_ratio, init_scale,
                         learnable_gate, use_msqwca, use_cqwg, query_water_fusion, query_water_gate, debug)
        self.water_encoder = WaterTokenEncoder(embed_dim=water_dim, num_heads=water_heads,
            means=water_means, stds=water_stds, dropout=dropout, water_features=water_features,
            water_self_attn=water_self_attn, use_rwa=use_rwa, water_relation=water_relation, debug=debug)

    def forward(self, query, water, return_attn=False):
        self.water_encoder.save_attention = self.save_attention or return_attn
        return super().forward(query, self.water_encoder(water), return_attn)
