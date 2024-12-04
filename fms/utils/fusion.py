import torch
import torch.nn as nn


def _maybe_unfuse_weights(module: nn.Module):
    if hasattr(module, "unfuse_weights"):
        if isinstance(module.weight, torch.Tensor) and module.weight.is_distributed():
            module.weight = module.weight.redistribute()
        result = module.unfuse_weights()
        del module
    else:
        result = module
    return result


def apply_unfuse_weights(module: nn.Module) -> nn.Module:
    """When applied to a module, will unfuse modules that support the unfuse_weights method

    Parameters
    ----------
    module: nn.Module
        the module to unfuse

    Returns
    -------
    nn.Module
        the original module unfused
    """
    with torch.no_grad():
        wrapped = _maybe_unfuse_weights(module)
        if wrapped is not module:
            if wrapped.weight.is_distributed():
                    wrapped.weight = wrapped.weight.redistribute()
            return wrapped

        for name, layer in module.named_children():
            unfused_layer = apply_unfuse_weights(layer)
            setattr(module, name, unfused_layer)
        
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if module.weight.is_distributed():
                module.weight = module.weight.redistribute()
        return module
