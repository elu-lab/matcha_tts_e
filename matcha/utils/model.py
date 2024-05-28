## Github[matcha.utils.model.py]: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/utils/model.py
## from https://github.com/jaywalnut310/glow-tts 

import numpy as np
import torch

def sequence_mask(length, max_length = None): 
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype = length.dtype, device = length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def fix_len_compatibility(length, num_downsamplings_in_unet =2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length/factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length
    
def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device
    b, t_x, t_y = mask.shape

    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype = mask.dtype).to(device = device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = (torch.sum(logw - logw_) ** 2) / torch.sum(lengths)
    return loss


# Remove OUTLIER
def remove_outlier(values):
    # values: np.array
    if np.isnan(values).any():
        p25 = np.nanpercentile(values, 25)
        p75 = np.nanpercentile(values, 75)
    else:   
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        
    # p25 = np.percentile(values, 25)
    # p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    # normal_indices = np.logical_and(values > lower, values < upper)
    # return values[normal_indices]
    
    values = np.where(values >= lower, values, p25)
    values = np.where(values <= upper, values, p75)
    return values 


def normalize(data, mu, std):
    # this is added to avoid NaN things.
    data_device = data.device
    data_dtype = data.dtype
    # Outlier Removed
    data_new = torch.tensor(remove_outlier(data.numpy()))
    
    if not isinstance(mu, (float, int)):
        if isinstance(mu, list):
            # mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            mu = torch.tensor(mu, dtype=data_dtype, device=data_device)
        elif isinstance(mu, torch.Tensor):
            # mu = mu.to(data.device)
            mu = mu.to(data_device)
        elif isinstance(mu, np.ndarray):
            # mu = torch.from_numpy(mu).to(data.device)
            mu = torch.from_numpy(mu).to(data_device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, (float, int)):
        if isinstance(std, list):
            # std = torch.tensor(std, dtype=data.dtype, device=data.device)
            std = torch.tensor(std, dtype=data_dtype, device=data_device)
        elif isinstance(std, torch.Tensor):
            # std = std.to(data.device)
            std = std.to(data_device)
        elif isinstance(std, np.ndarray):
            # std = torch.from_numpy(std).to(data.device)
            std = torch.from_numpy(std).to(data_device)
        std = std.unsqueeze(-1)

    return (data_new - mu) / std

# def normalize(data, mu, std):
#     if not isinstance(mu, (float, int)):
#         if isinstance(mu, list):
#             mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
#         elif isinstance(mu, torch.Tensor):
#             mu = mu.to(data.device)
#         elif isinstance(mu, np.ndarray):
#             mu = torch.from_numpy(mu).to(data.device)
#         mu = mu.unsqueeze(-1)

#     if not isinstance(std, (float, int)):
#         if isinstance(std, list):
#             std = torch.tensor(std, dtype=data.dtype, device=data.device)
#         elif isinstance(std, torch.Tensor):
#             std = std.to(data.device)
#         elif isinstance(std, np.ndarray):
#             std = torch.from_numpy(std).to(data.device)
#         std = std.unsqueeze(-1)

#     return (data - mu) / std


def denormalize(data, mu, std):
    if not isinstance(mu, float):
        if isinstance(mu, list):
            mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
        elif isinstance(mu, torch.Tensor):
            mu = mu.to(data.device)
        elif isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu).to(data.device)
        mu = mu.unsqueeze(-1)

    if not isinstance(std, float):
        if isinstance(std, list):
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(data.device)
        elif isinstance(std, np.ndarray):
            std = torch.from_numpy(std).to(data.device)
        std = std.unsqueeze(-1)

    return data * std + mu
