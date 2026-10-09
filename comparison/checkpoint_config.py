"""Read explicit or embedded model configuration without historical defaults."""

from pathlib import Path

import yaml


def read_config_tree(path, active=None):
    path = Path(path).resolve()
    active = set() if active is None else active
    if path in active:
        raise ValueError(f'Cyclic config include: {path}')
    active.add(path)
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        if not isinstance(config, dict):
            raise ValueError(f'Model config must contain a YAML mapping: {path}')
        combined = {}
        for include in config.get('__include__', []):
            combined.update(read_config_tree(path.parent / Path(include).expanduser(), active))
        combined.update(config)
        return combined
    finally:
        active.remove(path)


def infer_family(config):
    """Identify architecture from configuration, independently of display labels."""
    model = str(config.get('model', '')).lower()
    if model == 'deim':
        return 'deim'
    if model == 'dfine':
        return 'dfine'
    if model == 'rtdetr':
        return 'rtdetrv2' if 'RTDETRTransformerv2' in config else 'rtdetr'
    if 'DEIMTransformer' in config:
        return 'deim'
    if 'DFINETransformer' in config or 'DFINE' in config:
        return 'dfine'
    if 'RTDETRTransformerv2' in config:
        return 'rtdetrv2'
    if 'RTDETRTransformer' in config:
        return 'rtdetr'
    if isinstance(config.get('backbone'), list) and isinstance(config.get('head'), list):
        return 'yolo'
    return None


def resolve_model_config(spec, out):
    """Prefer user input; recover saved configuration when no file was supplied."""
    if spec.get('config'):
        spec['config_source'] = 'explicit'
        return spec['config']

    import torch
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    containers = [checkpoint] if isinstance(checkpoint, dict) else []
    if containers:
        args = checkpoint.get('args')
        if isinstance(args, dict):
            containers.append(args)
        elif hasattr(args, '__dict__'):
            containers.append(vars(args))
    for container in containers:
        for key in ('model_config', 'yaml_cfg', 'config', 'cfg'):
            value = container.get(key)
            if isinstance(value, dict) and value.get('model'):
                config = dict(value)
                if config.get('__include__'):
                    base = Path(spec['checkpoint']).resolve().parent
                    config['__include__'] = [str((base / path).resolve()) for path in config['__include__']]
                target = Path(out) / 'checkpoint_model_config.yml'
                target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
                spec['config_source'] = f'checkpoint:{key}'
                return str(target)
            if isinstance(value, (str, Path)) and str(value).lower().endswith(('.yaml', '.yml')):
                path = Path(value)
                if not path.is_absolute():
                    path = Path(spec['checkpoint']).resolve().parent / path
                if path.is_file():
                    spec['config_source'] = f'checkpoint:{key}'
                    return str(path.resolve())
    raise ValueError(
        f"{spec['name']}: cannot construct the model from {spec['checkpoint']}: "
        "no usable model configuration was stored in this checkpoint. "
        "Supply config in YAML or --model-config matching these weights."
    )


def validate_deim_config(config, state, name):
    """Explain known structural conflicts before constructing the detector."""
    issues = []
    score_head = state.get('decoder.enc_score_head.weight')
    if score_head is not None:
        saved_classes = int(score_head.shape[0])
        configured_classes = int(config.get('num_classes', 80))
        if configured_classes != saved_classes:
            issues.append(f'num_classes: config={configured_classes}, checkpoint={saved_classes}')
    backbone_name = config.get('DEIM', {}).get('backbone', 'HGNetv2')
    if backbone_name == 'HGNetv2':
        configured_water = bool(config.get('HGNetv2', {}).get('use_water_quality', True))
        saved_water = any(key.startswith('backbone.water_fusion.') for key in state)
        if configured_water != saved_water:
            issues.append(f'HGNetv2.use_water_quality: config={configured_water}, checkpoint={saved_water}')
    if issues:
        raise ValueError(
            f'{name}: config and checkpoint describe different model structures: '
            + '; '.join(issues)
            + '. Select the matching training config (config_used.yml) or an explicitly verified model config. '
            'Parameters will not be skipped or randomly initialized.'
        )
