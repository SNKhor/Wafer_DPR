# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

"""
Rewrite the code for deformable convolutional network with PyTorch
Reference: https://github.com/Junliangwangdhu/WaferMap/blob/master/deform_conv.py
"""
import numpy as np
import torch

#Map the input array to new coordinates by interpolation
from scipy.ndimage import map_coordinates as sp_map_coordinates

#flatten tensor
def torch_flatten(a):
    return a.view(-1)

#PyTorch version of np.repeat for 1D
def torch_repeat(a, repeats, axis=0):
    assert a.dim() == 1
    a = a.unsqueeze(-1)
    a = a.repeat(1, repeats)
    a = torch_flatten(a)
    return a

#PyTorch version of np.repeat for 2D
def torch_repeat_2d(a, repeats):
    assert a.dim() == 2
    a = a.unsqueeze(0)
    a = a.repeat(repeats, 1, 1)
    return a

'''
Parameters:
input: tf.Tensor. shape = (s, s)
coords: tf.Tensor. shape = (n_points, 2)
coords_lt -- left-top of coordinates
coords_rb -- right-bottom of coordinates
coords_lb -- left-bottom of coordinates
coords_rt -- right-top of coordinates 

for mapped_vals is calculated by bilinear interpolation

'''

def torch_map_coordinates(input, coords, order=1):
    assert order == 1  # '1' means linear interpolation

    coords_lt = coords.floor().to(torch.int32)
    coords_rb = coords.ceil().to(torch.int32)
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], dim=1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], dim=1)

    vals_lt = input[coords_lt[:, 0], coords_lt[:, 1]]
    vals_rb = input[coords_rb[:, 0], coords_rb[:, 1]]
    vals_lb = input[coords_lb[:, 0], coords_lb[:, 1]]
    vals_rt = input[coords_rt[:, 0], coords_rt[:, 1]]

    coords_offset_lt = coords - coords_lt.to(torch.float32)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals

# Scipy batch version of map_coordinates
def sp_batch_map_coordinates(inputs, coords):
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([sp_map_coordinates(input, coord.T, mode = 'nearest', order = 1) 
                            for input, coord in zip(inputs, coords)])

    return mapped_vals.numpy()

#Batch version of torch_map_coordinates
def torch_batch_map_coordinates(input, coords, order=1):
    batch_size, input_size, _ = input.shape
    n_coords = coords.shape[1]

    coords = coords.clamp(0, input_size - 1)
    coords_lt = coords.floor().to(torch.int32)
    coords_rb = coords.ceil().to(torch.int32)
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], dim=-1)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], dim=-1)

    idx = torch_repeat(torch.arange(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([idx, torch_flatten(coords[..., 0]), torch_flatten(coords[..., 1])], dim=-1)
        vals = input[indices[:, 0], indices[:, 1], indices[:, 2]]
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    #bilinear interpolation
    coords_offset_lt = coords - coords_lt.to(torch.float32)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals

# SciPy batch version of sp_batch_map_offsets
def np_batch_map_offsets(input, offsets):
    batch_size = input.shape[0]
    input_size = input.shape[1]
    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)
    
    mapped_vals = np_batch_map_coordinates(input, coords)

    return mapped_vals

# PyTorch batch version of torch_batch_map_offsets
def torch_batch_map_offsets(input, offsets, order=1):
    batch_size, input_size, _ = input.shape

    offsets = offsets.view(batch_size, -1, 2)
    grid = torch.meshgrid(torch.arange(input_size), torch.arange(input_size))
    grid = torch.stack(grid, dim=-1)
    grid = grid.to(torch.float32)
    grid = grid.view(-1, 2)
    grid = torch_repeat_2d(grid, batch_size)
    coords = grid + offsets

    mapped_vals = torch_batch_map_coordinates(input, coords)

    return mapped_vals
