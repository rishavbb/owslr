import torch
import torch.nn.functional as F
from models.overlap import OverlappingWindow 
from models.decoder import Decoder
import models
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OWSLR(torch.nn.Module):

    def __init__(self,
                 encoder_spec,
                 starting_window_size,
                 ending_window_size,
                 lr_height_width,
                 kernel_size,
                 hidden_size,
                 semiLocal_length,
                 distance_decay_rate,
                 decoder_layers,
                 ):

        super(OWSLR, self).__init__()

        self.semiLocal_length = semiLocal_length   
        self.lr_height_width = lr_height_width
        self.hidden_size = hidden_size
        self.semiLocal_y, self.semiLocal_x = torch.meshgrid(torch.linspace(start=-self.semiLocal_length//2,
                                                                           end=self.semiLocal_length//2,
                                                                           steps=self.semiLocal_length),
                                                            torch.linspace(start=-self.semiLocal_length//2,
                                                                           end=self.semiLocal_length//2,
                                                                           steps=self.semiLocal_length),
                                                            indexing='ij')
        self.semiLocal_y = self.semiLocal_y.to(device)
        self.semiLocal_x = self.semiLocal_x.to(device)
        
        self.encoder = models.make(encoder_spec)

        self.decoder = Decoder(hidden_size=self.hidden_size,
                               kernel_size=kernel_size,
                               decoder_layers=decoder_layers,
                               )

        self.overlapping_window = OverlappingWindow(starting_window_size=starting_window_size,
                                                    ending_window_size=ending_window_size,
                                                    hidden_size=hidden_size,
                                                    kernel_size=kernel_size,
                                                    distance_decay_rate=distance_decay_rate)
        

    def img_preprocess(self, img):
        img = (img - 0.5) / 0.5
        return img

    def img_postprocess(self, img):
        img = img * 0.5 + 0.5
        return img


    def create_semiLocal_area(self, hr_cs, lr_unit_width, lr_unit_height):
        '''
        hr_cs -> high resolution coordinates - Shape:(B-16, sample_size-1500, xy-2)
        lr_unit_width -> unit width of feature map from edsr  (float)
        lr_unit_height -> unit height of feature map from edsr  (float)
        semiLocal_length -> length of the semilocal region  (int)


        Returns:
        hr_coordinates: semi-local region coordinates around all 1500 (sample_size)
                        points wrt to HR Image 
                        - Shape:(B-16, sample_size-1500, semiLocal_length-6, semiLocal_length-6, xy-2)
        interspace_coord: distance between the query point and the other 
                          36 (semiLocal_length*semiLocal_length) semilocal points 
                          - Shape:(B-16, sample_size-1500, semiLocal_length-6, semiLocal_length-6, xy-2)
         '''

        m = hr_cs[..., 0].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                           1,
                                                           self.semiLocal_length,
                                                           self.semiLocal_length)
        x = (m + self.semiLocal_x * lr_unit_width).clamp_(-1 + 1e-6, 1 - 1e-6)

        n = hr_cs[..., 1].unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                             1,
                                                             self.semiLocal_length,
                                                             self.semiLocal_length)
        y = (n + self.semiLocal_y * lr_unit_height).clamp_(-1 + 1e-6, 1 - 1e-6)

        return torch.stack((x, y), dim=-1).to(device), torch.stack(((m-x), (n-y)), dim=-1).to(device)



    def make_feat(self, x):
        x = self.encoder(x)
        return x


    def forward(self, lr_img, hr_training_coordinates, cell):
        x = self.make_feat(lr_img)
        x = self.find_rgb(x, hr_training_coordinates, cell)
        return x


    def find_rgb(self, x, hr_training_coordinates, cell):

        lr_unit_width = lr_unit_height = 1/x.shape[-1]  

        cell[..., 0] *= x.shape[-2]
        cell[..., 1] *= x.shape[-1]

        hr_training_coordinates, interspace_coord = \
            self.create_semiLocal_area(hr_cs=hr_training_coordinates.flip(-1),
                                       lr_unit_width=lr_unit_width,
                                       lr_unit_height=lr_unit_height) 
        #hr_training_coordinates: Shape:(B-16,
        #                                sample_size-1500,
        #                                semiLocal_length-6,
        #                                semiLocal_length-6, xy-2)
        #interspace_coord: Shape:(B-16,
        #                         sample_size-1500,
        #                         semiLocal_length-6,
        #                         semiLocal_length-6, xy-2)

        interspace_coord = interspace_coord.reshape(-1,
                                                    self.semiLocal_length,
                                                    self.semiLocal_length,
                                                    2).permute(0,3,1,2)
        
        B, num_pixel_points, _, _, _ = hr_training_coordinates.shape 


        x = F.grid_sample(x,
                          hr_training_coordinates.reshape(B,-1,2).unsqueeze(1),
                          mode='nearest',
                          align_corners=False)[:, :, 0, :].\
                          reshape(B,
                                  self.hidden_size,
                                  num_pixel_points,
                                  self.semiLocal_length*self.semiLocal_length).\
                          permute(0,2,1,3).\
                          reshape(B,
                                  num_pixel_points,
                                  self.hidden_size,
                                  self.semiLocal_length,
                                  self.semiLocal_length)

        x = x.reshape(-1,
                      self.hidden_size,
                      self.semiLocal_length,
                      self.semiLocal_length)  
        #x - Shape:({B-16 * sample_size-1500}-24000,
        #           hidden_size-64,
        #           semiLocal_length-6,
        #           semiLocal_length-6)

        x = self.overlapping_window(x, cell, interspace_coord)

        cell = cell.reshape(-1, 2)

        x = self.decoder(x, interspace_coord, cell)
        
        return x
