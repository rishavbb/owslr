import torch
from torch.nn import Conv2d, ReLU, Linear, Sigmoid, Sequential, Dropout, Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"

class EarlyEmbed(torch.nn.Module):
    def __init__(self,
                 #kernel_size=3,
                 hidden_size):
  
        super(EarlyEmbed, self,).__init__()
  
        #self.hidden_size = hidden_size
        self.conv =  Conv2d(in_channels=3,
                            out_channels=hidden_size,
                            kernel_size=1,
                            padding=0
                            )
        self.act = ReLU()
      
    def forward(self, x):
  
        B, C, H, W = x.shape
  
        early_embeddings = self.act(self.conv(x))
  
        return early_embeddings


class FetchBlock(torch.nn.Module):
    def __init__(self,
                 kernel_size,
                 hidden_size,
                 img_length,
                 dropout=.1):
        super(FetchBlock, self).__init__()

        self.kernel_size = kernel_size
        self.conv1 = Conv2d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=self.kernel_size,
                          padding=self.kernel_size//2
                          )

        self.conv2 = Conv2d(in_channels=hidden_size,
                          out_channels=hidden_size,
                          kernel_size=self.kernel_size,
                          padding=self.kernel_size//2
                          )
        self.relu = ReLU()

        self.solver = Linear(in_features=self.kernel_size*self.kernel_size*hidden_size,
                             out_features=hidden_size)
        self.sigmoid = Sigmoid()

        self.dropout = Dropout(dropout)


    def forward(self,
                x,
                early_embed):

        '''
        x : [B, C, H, W]
        early_embed: [B, C, H, W]
        '''

        B, C, H, W = x.shape
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x + early_embed





class SharedAttention(torch.nn.Module):
    def __init__(self,
                 img_length,
                 hidden_size,
                 ):
        super(SharedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.img_length = img_length
        self.conv1 = Conv2d(in_channels=self.hidden_size*2, #*self.kernel_size*self.kernel_size,
                            out_channels=self.hidden_size,
                            kernel_size=2, #self.kernel_size,
                            padding=2//2,
                            )

        self.conv2 = Conv2d(in_channels=self.hidden_size*2, #*self.kernel_size*self.kernel_size,
                           out_channels=self.hidden_size,
                           kernel_size=2, #self.kernel_size,
                           padding=2//2, #self.kernel_size//2
                           )

        self.conv3 = Conv2d(in_channels=self.hidden_size,
                            out_channels=self.hidden_size,
                            kernel_size=2, #self.kernel_size,
                            padding=0,#2//2, #self.kernel_size//2
                            )

        self.seq = Sequential(Linear(in_features=self.hidden_size*4*self.img_length*2,
                                     out_features=self.hidden_size),
                              Linear(in_features=self.hidden_size,
                                     out_features=2*2*self.hidden_size)
                                     )


    def assemble(self, x):

        x = torch.sin(self.conv1(x)) + torch.cos(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward(self, x,):

        B, C, H, W = x.shape

        #x = self.assemble(x)
        self.border_index = torch.tensor([0, 1, H-2, H-1]).to(device) 

        k = torch.cat([torch.index_select(x, -1, self.border_index).view(B, -1),
                       torch.index_select(x, -2, self.border_index).view(B, -1)],
                       dim=-1)
        k = self.seq(k).reshape(B, self.hidden_size, 2, 2)

        x = x[..., 2:4, 2:4]

        x = torch.cat([x, k], dim=-3)

        x = self.assemble(x).reshape(B, C, -1).permute(0, 2, 1)
        x = [x[:, 0,:], x[:, 1,:], x[:, 2,:], x[:, 3,:]]

        return x


class OverlappingWindow(torch.nn.Module):
    def __init__(self,
                 #window_size,
                 starting_window_size,
                 ending_window_size,
                 kernel_size,
                 hidden_size,
                 distance_decay_rate=0.1):
        super(OverlappingWindow, self).__init__()

        #self.window_size = window_size
        self.starting_window_size = starting_window_size
        self.ending_window_size = ending_window_size
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size + 4

        # self.distance_matrix = self.get_distance_matrix(distance_decay_rate) #.to(device)
        self.starting_window_size = starting_window_size
        self.ending_window_size = ending_window_size

        self.conv1 = Conv2d(in_channels=self.hidden_size, #*self.kernel_size*self.kernel_size,
                            out_channels=self.hidden_size,
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size//2
                            )

        #self.conv2 = Conv2d(in_channels=self.hidden_size, #*self.kernel_size*self.kernel_size,
        #                    out_channels=self.hidden_size,
        #                    kernel_size=self.kernel_size,
        #                    padding=self.kernel_size//2
        #                    )

        self.conv3 = Conv2d(in_channels=self.hidden_size,
                            out_channels=self.hidden_size,
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size//2
                            )

        self.linear5 = Linear(in_features=self.hidden_size*5*5,
                              out_features=3*3*(self.hidden_size-4))#.to(device)
        self.linear4 = Linear(in_features=self.hidden_size*4*4,
                              out_features=3*3*(self.hidden_size-4))#.to(device)
        self.linear3 = Linear(in_features=self.hidden_size*3*3,
                              out_features=3*3*(self.hidden_size-4))#.to(device)

        self.final = Linear(in_features=3*3*(self.hidden_size-4),
                            out_features=self.hidden_size-4)

        self.relu = ReLU()


    def get_distance_matrix(self, distance_decay_rate):

        distance_matrix = torch.Tensor([[1.0, 1.0],
                          [1.0, 1.0],])
        distance_decay = 1 - distance_decay_rate
        for _ in range(2):
            distance_matrix = torch.nn.functional.pad(distance_matrix,
                                                      (1,1,1,1),
                                                      mode='constant',
                                                      value=distance_decay)
            distance_decay -= distance_decay_rate
        
        distance_matrix = Parameter(distance_matrix)
        final = {}
        for window_size in range(self.starting_window_size, self.ending_window_size -1, -1):
            final[window_size] = self.get_four_corner_windows(distance_matrix,
                                                              window_size)


        return final #self.get_four_corner_windows(distance_matrix)



    def get_four_corner_windows(self, x, window_size):

        H, W = x.shape[-2:]  #[H, W]

        left_top = x[..., :window_size, :window_size]

        right_top = x[..., :window_size, W-(window_size):]

        bottom_left = x[..., H-(window_size):, :window_size]

        bottom_right = x[..., H-(window_size):, W-(window_size):]

        return torch.stack([left_top,
                            right_top,
                            bottom_left,
                            bottom_right], dim=0).to(device)


    def assemble(self, x, window_size):
        '''
        Multiple x with the distance matrix and rearrange the shape
        and transfer features for the next window size
        x-> Shape:[96000, 68, window_size, window_size]
        window_size-> (int)

        Returns:
        x-> Shape:[4, 24000, -1]


        '''
        
        Bx4, _, _, _ = x.shape

        x = torch.sin(x) + torch.cos(x)
        x = self.conv1(x)
        x = self.relu(x)



        x = torch.sin(x) + torch.cos(x)
        x = self.conv3(x).reshape(4, Bx4//4, self.hidden_size, window_size, window_size) 

        # d_mat = self.distance_matrix[window_size]
        # for i in range(4):
        #     x[i] = x[i]*d_mat[i] #.clone()

        x = self.relu(x)
        x = x.reshape(4, Bx4//4, -1) 

        return x



    def join_feat(self, x):
        '''
        Concatenating the four corner features
        x-> Shape:[4, 24000, 64, 3, 3]

        Returns:
        x-> Shape:[4, 24000, 64, 6, 6]
        '''

        m = torch.cat([x[0], x[1]], dim=-1)
        n = torch.cat([x[2], x[3]], dim=-1)
        x = torch.cat([m,n], dim=-2)
        return x

    def forward(self, x, cell, interspace_coord):

        B, C, H, W = x.shape

        new_cell = cell.clone()
        new_cell = new_cell.reshape(B, -1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1, 6, 6)
        


        for window_size in range(self.starting_window_size, self.ending_window_size - 1, -1):

            x = torch.cat([x, new_cell, interspace_coord], dim=-3)  
            #x-> Shape:(B-24000, 68, semiLocal_length-6, semiLocal_length-6)

            x = self.get_four_corner_windows(x, window_size).\
                                            reshape(-1,
                                                    self.hidden_size,
                                                    window_size,
                                                    window_size)
            #x-> Shape:(24000*4-96000, 68, semiLocal_length-6, semiLocal_length-6)

            x = self.assemble(x, window_size) 

            if window_size==5:
                x = self.relu(self.linear5(x)).reshape(4, B, self.hidden_size - 4, 3, 3)
            elif window_size==4:
                x = self.relu(self.linear4(x)).reshape(4, B, self.hidden_size - 4, 3, 3)
            elif window_size==3:
                x = self.relu(self.linear3(x)).reshape(4, B, self.hidden_size - 4, 3, 3)

            else:
                raise Exception("Window Size has to be >=3 to <=5")
            x = self.join_feat(x)

        
        corners = self.get_four_corner_windows(x, self.ending_window_size) 
        _, B_new, _, _, _ = corners.shape
        corners = corners.reshape(B_new*4, -1)


        corners = self.final(corners).reshape(B_new, self.hidden_size-4, 2, 2)

        x = x[:, :,
              self.ending_window_size-1:self.ending_window_size+1,
              self.ending_window_size-1:self.ending_window_size+1]

        x = (x*corners).reshape(B_new, self.hidden_size-4, -1)

        return x



class Encoder(torch.nn.Module):
  def __init__(self,
               num_layers=4,
               kernel_size=3,
               hidden_size=128, #TODO yet to be decided
               img_length=6,
               num_heads=4,
               distance_decay_rate=0.1
               ):
    super(Encoder, self).__init__()

    layers = []
    for _ in range(num_layers):

      layers.append(FetchBlock(kernel_size=kernel_size,
                                       hidden_size=hidden_size,
                                       img_length=img_length,
                                       dropout=.1))
    self.deep_intricate_block = Sequential(*layers)


  def forward(self, x, early_embed):

      for each_layer in self.deep_intricate_block:
          x = each_layer(x, early_embed)

      return x


if __name__ == "__main__":
    dummy_model = Encoder(num_layers=1, 
               initial_window_size=3,
               kernel_size=3,
               hidden_size=128, #TODO yet to be decided
               img_length=18,
               num_heads=4,
               distance_decay_rate=0.1).to(device)
        

    dummy_model.eval()


    dummy_x = torch.randn(1, 6, 18, 18).to(device)
    print(dummy_model(dummy_x))


    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, dummy_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
