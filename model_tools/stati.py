from concurrent.futures import ThreadPoolExecutor

import torch


def freeze_module(module: torch.nn.Module):
    """
    Freeze all the parameters of a given PyTorch module, setting requires_grad to False.
    
    Args:
        module (torch.nn.Module): The model to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module):
    """
    Unfreeze all the parameters of a given PyTorch module, setting requires_grad to True.
    
    Args:
        module (torch.nn.Module): The model to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True


@torch.no_grad()
def get_size(t: torch.Tensor) -> int:
    """
    Get the size of a given tensor in bytes.
    
    Args:
        t (torch.Tensor): The tensor whose size is to be calculated.
    
    Returns:
        int: The size of the tensor in bytes.
    """
    return t.nelement() * t.element_size()


def stati_model(model: torch.nn.Module, unit: str = "bytes") -> dict:
    """
    Return the model's parameter count and memory size, both for gradient and non-gradient parameters.
    
    Args:
        model (torch.nn.Module): A PyTorch model.
        unit (str): The unit for memory size, options are 'bytes', 'kb', 'mb', or 'gb'.
    
    Returns:
        dict: A dictionary with parameter counts and memory size.
    """
    param_size = 0
    buffer_size = 0
    param_count_with_grad = 0
    param_count_without_grad = 0

    with ThreadPoolExecutor() as executor:
        param_futures = [executor.submit(get_size, param) for param in model.parameters()]
        param_size = sum(future.result() for future in param_futures)

        for param in model.parameters():
            if param.requires_grad:
                param_count_with_grad += param.numel()
            else:
                param_count_without_grad += param.numel()

        buffer_futures = [executor.submit(get_size, buffer) for buffer in model.buffers()]
        buffer_size = sum(future.result() for future in buffer_futures)

    total_size = param_size + buffer_size

    unit_dict = {
        "bytes": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3
    }

    total_size_in_unit = total_size / unit_dict.get(unit.lower(), 1)

    return {
        "param_count_with_grad": param_count_with_grad,
        "param_count_without_grad": param_count_without_grad,
        "model_size": total_size_in_unit
    }
