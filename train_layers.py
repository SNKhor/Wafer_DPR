import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from .deform_conv import torch_batch_map_offsets

class ConvOffset2D_train(nn.Conv2d):
    '''
    Convolutional layer responsible for learning the 2D offsets and output the deformed
    feature map using bilinear interpolation

    Note that this layer does not perform convolution on the deformed feature map

    '''

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        '''
        Parameters:
        filters: int
        Number of channels of the input feature map
        init_normal_stddev: float
        Normal kernel initialization
        **kwargs:
        pass to superclass. see the Conv2D layer in PyTorch
        '''
        self.filters = filters
        super(ConvOffset2D_train, self).__init__(self.filters * 2, filters, kernel_size=3, padding='same', bias=False)
        normal_(self.weight, mean=0, std=init_normal_stddev)

    def forward(self, x):
        '''
        return the deformed featured map
        '''
        x_shape = x.shape
        offsets = super(ConvOffset2D_train, self).forward(x)

        # offsets: (b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)
        
        # X_offset: (b*c, h, w)
        x_offset = torch_batch_map_offsets(x, offsets)
        
        # x_offset: (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        '''
        Output shape is the same as input shape
        Because this layer only does the deformation part
        '''
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        '''
        (b, c, h, 2w)->(bc, h, w, 2)
        '''
        x = x.view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        '''
        (b, c, h, w)->(bc, h, w)
        '''
        x = x.view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        '''
        (b*c, h, w)->(b, c, h, w)
        '''
        x = x.view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x
