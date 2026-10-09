"""Grad-CAM from a differentiable detector score and one spatial feature layer."""

import torch
import torch.nn.functional as F


def resolve_layer(model, name, family):
    modules = dict(model.named_modules())
    if name != 'auto':
        if name not in modules:
            raise ValueError(f'CAM layer {name!r} does not exist; run --list-layers')
        return name, modules[name]
    if family == 'yolo':
        head = model.model[-1]
        source = head.f[-1] if isinstance(head.f, (list, tuple)) else head.f
        if source < 0:
            source = len(model.model) - 2
        name = f'model.{source}'
        return name, modules[name]
    candidates = [(name, module) for name, module in modules.items()
                  if name.startswith('encoder.') and isinstance(module, torch.nn.Conv2d)]
    if not candidates:
        candidates = [(name, module) for name, module in modules.items()
                      if name.startswith('backbone.') and isinstance(module, torch.nn.Conv2d)]
    if not candidates:
        raise ValueError('No spatial convolution found; specify cam_layer explicitly')
    return candidates[-1]


def score_target(probabilities, threshold, topk, class_index=None):
    """Select confidence-passing class scores before discrete detection filtering."""
    scores = probabilities[..., class_index] if class_index is not None else probabilities.flatten()
    scores = scores.flatten()
    selected = scores[scores.detach() >= threshold]
    if selected.numel() == 0:
        return None, []
    selected = selected.topk(min(topk, selected.numel())).values
    return selected.sum(), selected.detach().cpu().tolist()


def gradcam(model, layer, pixels, forward_target):
    captured = {}

    def save_activation(_, __, output):
        if not torch.is_tensor(output) or output.ndim != 4:
            raise ValueError('CAM target layer must return a BCHW tensor; choose a spatial layer')
        captured['activation'] = output

    handle = layer.register_forward_hook(save_activation)
    try:
        with torch.enable_grad():
            pixels = pixels.detach().requires_grad_(True)
            target, scores = forward_target(pixels)
            if target is None:
                return torch.zeros(pixels.shape[-2:]), {'status': 'no_scores_above_threshold', 'target_scores': []}
            if 'activation' not in captured:
                raise RuntimeError('Selected layer was not executed during the CAM forward')
            activation = captured['activation']
            gradients = torch.autograd.grad(target, activation, allow_unused=True)[0]
            if gradients is None:
                raise RuntimeError('Selected score has no gradient to CAM layer; select another layer')
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            raw = (weights * activation).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(raw, pixels.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            return cam.detach().cpu(), {
                'status': 'ok' if raw.max().item() > 0 else 'zero_positive_activation',
                'target_scores': scores, 'raw_max': raw.max().item(),
            }
    finally:
        handle.remove()


def yolo_head_target(model, pixels, threshold, topk, class_index=None):
    """Replay the active classification branch with gradients to its input features.

    This also handles end-to-end heads that detach features in their forward.
    It preserves the inference branch choice: one-to-one for end-to-end models.
    """
    head = model.model[-1]
    captured = {}

    def save_inputs(_, inputs):
        captured['features'] = list(inputs[0])

    handle = head.register_forward_pre_hook(save_inputs)
    try:
        model(pixels)
    finally:
        handle.remove()
    use_one2one = bool(getattr(head, 'end2end', False))
    classifier = getattr(head, 'one2one_cv3', None) if use_one2one else getattr(head, 'cv3', None)
    if classifier is None:
        raise RuntimeError('Unsupported YOLO classification head; inspect this Ultralytics version')
    logits = torch.cat([classifier[index](feature).flatten(2)
                        for index, feature in enumerate(captured['features'])], dim=-1)
    probabilities = logits.sigmoid().transpose(1, 2)
    return score_target(probabilities, threshold, topk, class_index)
