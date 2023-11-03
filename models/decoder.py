import torch.nn as nn
from torch.nn import Conv2d, Linear, Sequential, ReLU
import torch

class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 kernel_size,
                 decoder_layers):
                 
        super(Decoder, self).__init__()

        layer = []
        temp = hidden_size + 2 + 3*3*2

        for i in range(len(decoder_layers)):
            layer.append(Linear(in_features=temp,
                                out_features=decoder_layers[i]))
            layer.append(ReLU())
            temp = decoder_layers[i]

        layer.append(Linear(in_features=temp,
                     out_features=3))

        self.seq = Sequential(*layer)

            


    def forward(self, x, interspace_coord, cell):
        '''
        x <- [[B, C, 3, 3], [B, C, 3, 3],[B, C, 3, 3],[B, C, 3, 3]]
        '''

        BxnumPixelPts, _, _,  = x.shape

        pred = 0

        x = [x[...,0], x[..., 1], x[..., 2], x[..., 3]]
        interspace_coord = [interspace_coord[..., :3, :3],
                            interspace_coord[..., :3, 3:],
                            interspace_coord[..., 3:,  :3],
                            interspace_coord[..., 3:, 3:]
                           ]



        for each_corner, each_interspace in zip(x, interspace_coord):

            pred += self.seq(torch.cat([each_corner.reshape(BxnumPixelPts, -1),
                                        each_interspace.reshape(BxnumPixelPts,
                                                                -1),
                                        cell],
                                        dim=-1))

        return pred
