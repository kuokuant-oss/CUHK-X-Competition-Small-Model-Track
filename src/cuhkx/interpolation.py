"""Same-origin parameter interpolation with explicitly reset training-side BN statistics."""
# Role: linear interpolation between two state dicts of one network, with the BatchNorm running
# statistics reset, and BatchNorm re-estimation: reset, then a cumulative mean over batches with
# dropout off and no gradients, so the learned weights do not change.
# Used by: training/108_finalize_calibrated.py (member B: interpolate_state, then calibrate_bn) and
# training/124_c_full_student.py (member C: calibrate_bn); training.

import torch
from torch import nn


# Returns a state dict for `model`: learnable parameters become (1 - alpha) * left + alpha * right
# in float32, BatchNorm running statistics are reset (mean 0, variance 1, counter 0) for
# calibrate_bn to re-estimate, and every other buffer must be equal in both inputs and is copied.
# For member B both inputs are the same checkpoint, so this call serves to reset the statistics.
def interpolate_state(model, left, right, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    expected = model.state_dict()
    if set(left) != set(right) or set(left) != set(expected):
        raise ValueError("state keys differ")
    # Names of the learnable parameters, and of the running statistics of every BatchNorm layer.
    parameters = set(dict(model.named_parameters()))
    bn_buffers = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            bn_buffers.update(
                f"{name}.{k}" if name else k
                for k in ("running_mean", "running_var", "num_batches_tracked")
            )
    output = {}
    for key, template in expected.items():
        a, b = left[key], right[key]
        if a.shape != b.shape or a.shape != template.shape:
            raise ValueError(f"shape mismatch: {key}")
        if key in parameters:
            if not a.is_floating_point() or not b.is_floating_point():
                raise ValueError(f"nonfloating learnable parameter: {key}")
            output[key] = a.float() * (1 - alpha) + b.float() * alpha
        elif key in bn_buffers:
            output[key] = (
                torch.ones_like(template)
                if key.endswith("running_var")
                else torch.zeros_like(template)
            )
        else:
            # Non-learned buffers, such as the input normalisation mean and standard deviation.
            if not torch.equal(a, b):
                raise ValueError(f"fixed buffer differs: {key}")
            output[key] = a.clone().to(template.dtype)
    return output


def calibrate_bn(model, loader, *, max_batches=200, device="cpu"):
    """Reset BN then cumulative-average batches; dropout remains off, no parameter updates."""
    if max_batches <= 0:
        raise ValueError("positive max_batches required")
    # The whole model goes to inference mode (dropout off); only the BatchNorm layers are put
    # back into training mode below, so that they collect statistics.
    model.eval()
    modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in modules]
    if not modules:
        raise ValueError("model has no BN modules")
    for module in modules:
        if not module.track_running_stats:
            raise ValueError("BN must track statistics")
        module.reset_running_stats()
        # momentum=None makes PyTorch keep an equal-weight cumulative mean over the batches.
        module.momentum = None
        module.train()
    count = 0
    samples = 0
    try:
        # Forward passes only, outputs discarded; at most max_batches batches (members B and C:
        # 200 batches of 8 training clips with inference-time frames and crops).
        with torch.no_grad():
            for inputs, _ in loader:
                if count == max_batches:
                    break
                # Accept a bare tensor as well as the tuple of tensors that train._collate yields.
                if isinstance(inputs, torch.Tensor):
                    inputs = (inputs,)
                model(*(x.to(device) for x in inputs))
                count += 1
                samples += len(inputs[0])
    finally:
        # Restore the original momenta and inference mode, also if a batch fails.
        for module, momentum in zip(modules, momenta, strict=True):
            module.momentum = momentum
        model.eval()
    if count == 0:
        raise ValueError("empty calibration")
    # Summary that the calling scripts write to a JSON file beside the calibrated checkpoint.
    return dict(
        batches=count,
        samples=samples,
        method="reset+cumulative batch average",
        dropout=False,
        gradients=False,
    )
