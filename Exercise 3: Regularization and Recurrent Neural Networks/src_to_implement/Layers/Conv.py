import numpy as np
from Layers import Base
import copy

class Conv(Base.BaseLayer):

    #Constructor with stride shape, conv shape and number of kernels           
    def __init__(self, stride_shape, conv_shape, num_kernels):

        super().__init__()
        self.trainable = True  # this layer has learnable parameters

        #stride shape normalisation
        if isinstance(stride_shape, int):
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.sy, self.sx = stride_shape            # (stride_y, stride_x)

        #detect 1-D vs 2-D kernel
        # 2-D: conv_shape = (C, kH, kW)
        # 1-D: conv_shape = (C, kL) -> treat as (C, kL, 1)
        self.is_2d = len(conv_shape) == 3
        if self.is_2d:
            self.conv_shape = conv_shape
        else:
            self.conv_shape = (*conv_shape, 1)

        C, kH, kW = self.conv_shape
        self.K    = num_kernels                     # #kernels / output-channels

        #weight & bias initialisation
        self.weights = np.random.uniform(0, 1, (self.K, C, kH, kW))  # (K,C,kH,kW)
        self.bias    = np.random.uniform(0, 1, (self.K,))            # (K,)

        # placeholders for grads
        self.gradient_weights = None
        self.gradient_bias    = None

        # optimiser container
        self._opt = None

        # cache variables (set in forward)
        self.inp   = None     # input after possible promotion to 4-D
        self.pad_y = 0
        self.pad_x = 0
        self.squeeze = False  # flag for 1-D promotion

    #Optimizer property (two copies: one for weight, one for bias)
    @property
    def optimizer(self):
        return self._opt

    @optimizer.setter
    def optimizer(self, opt):
        self._opt        = opt
        self._opt.weight = copy.deepcopy(opt)  # independent copy for weights
        self._opt.bias   = copy.deepcopy(opt)  # independent copy for bias

    #Forward pass 
    def forward(self, x):

        #Promote 1-D to 2-D by adding a dummy width dimension
        self.squeeze = False
        if x.ndim == 3:                                        # (B,C,L)
            x = x[:, :, :, np.newaxis]                         # (B,C,L,1)
            self.squeeze = True

        self.inp = x                                           # cache input
        B, C, H, W = x.shape
        _, kH, kW  = self.conv_shape

        #SAME-padding amounts 
        self.pad_y = (kH - 1) // 2
        self.pad_x = (kW - 1) // 2
        pad_post_y = kH - 1 - self.pad_y
        pad_post_x = kW - 1 - self.pad_x

        # pad: ((before, after) per dim)
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (self.pad_y, pad_post_y), (self.pad_x, pad_post_x)),
            mode="constant"
        )

        #output spatial size
        out_H = (H + self.pad_y + pad_post_y - kH) // self.sy + 1
        out_W = (W + self.pad_x + pad_post_x - kW) // self.sx + 1
        out   = np.zeros((B, self.K, out_H, out_W))

        #Convolution by explicit loops
        for b in range(B):
            for k in range(self.K):
                for i in range(out_H):
                    y0 = i * self.sy
                    for j in range(out_W):
                        x0 = j * self.sx
                        # slice receptive field → (C,kH,kW)
                        region = x_pad[b, :, y0:y0+kH, x0:x0+kW]
                        out[b, k, i, j] = np.sum(region * self.weights[k]) + self.bias[k]

        #squeeze back to 1-D if needed
        if self.squeeze:
            out = out.squeeze(axis=3)   # (B,K,L_out)
        return out

    #Backward pass   
    def backward(self, err):

        #re-expand 1-D error to 4-D if necessary
        if self.squeeze:
            err = err[:, :, :, np.newaxis]         # (B,K,L_out,1)

        B, K, out_H, out_W = err.shape
        _, C, H, W         = self.inp.shape
        _, kH, kW          = self.conv_shape

        #allocate gradient tensors
        self.gradient_bias    = np.zeros_like(self.bias)
        self.gradient_weights = np.zeros_like(self.weights)
        grad_in               = np.zeros_like(self.inp)

        #Precompute SAME padding parameters again
        pad_post_y = kH - 1 - self.pad_y
        pad_post_x = kW - 1 - self.pad_x

        #Padded input (so weight-grad loop can slice without bounds checks
        x_pad = np.pad(
            self.inp,
            ((0, 0), (0, 0), (self.pad_y, pad_post_y), (self.pad_x, pad_post_x)),
            mode="constant"
        )

        #Main nested loops (exact analytic gradients)
        for b in range(B):
            for k in range(K):
                for i in range(out_H):
                    y0 = i * self.sy
                    for j in range(out_W):
                        x0   = j * self.sx
                        delta = err[b, k, i, j]        # scalar dL/d pre-activation

                        # bias gradient
                        self.gradient_bias[k] += delta

                        for c in range(C):
                            #  window from padded input (C axis iterated)
                            for yy in range(kH):
                                in_y = y0 + yy
                                if in_y < self.pad_y or in_y >= self.pad_y + H:
                                    continue  # outside valid input after padding
                                for xx in range(kW):
                                    in_x = x0 + xx
                                    if in_x < self.pad_x or in_x >= self.pad_x + W:
                                        continue

                                    # index in original input tensor
                                    src_y = in_y - self.pad_y
                                    src_x = in_x - self.pad_x

                                    # weight gradient
                                    self.gradient_weights[k, c, yy, xx] += (
                                        self.inp[b, c, src_y, src_x] * delta
                                    )

                                    # input gradient
                                    grad_in[b, c, src_y, src_x] += (
                                        self.weights[k, c, yy, xx] * delta
                                    )

        #Optimiser update (if attached)
        if self._opt:
            self.weights = self._opt.weight.calculate_update(
                self.weights, self.gradient_weights)
            self.bias = self._opt.bias.calculate_update(
                self.bias, self.gradient_bias)

        #squeeze back to 1-D shape if needed
        if self.squeeze:
            grad_in = grad_in.squeeze(axis=3)      # (B,C,L)
        return grad_in

    #Initializer hook (called by NeuralNetwork.append_layer)
    def initialize(self, w_init, b_init):
        fan_in  = np.prod(self.conv_shape)
        fan_out = self.K * np.prod(self.conv_shape[1:])
        self.weights = w_init.initialize(self.weights.shape, fan_in, fan_out)
        self.bias    = b_init.initialize(self.bias.shape, 1, self.K)
