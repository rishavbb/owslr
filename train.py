from functools import partial
import math
import datasets
import torch 
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import utils
from models.model import OWSLR 
import os
import yaml
import datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

  
def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'],
                            args={'dataset': dataset})

    loader = DataLoader(dataset,
                        batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'),
                        num_workers=8, pin_memory=True)

    return loader


def eval_psnr(loader,
              model,
              data_norm=None,
              eval_type=None,
              eval_bsize=None,
              verbose=False):

    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr,
                            dataset='div2k',
                            scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr,
                            dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError


    val_res = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        with torch.no_grad():
            pred = model(inp,
                            batch['coord'],
                            batch['cell'])

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        else:
            B, N_pxl_pts, _ = batch['gt'].shape
            pred = pred.reshape(B, N_pxl_pts, 3)

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])


        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    model = OWSLR(**config["model"]["args"]
                    ).to(device)
    
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of Parameters", params)

    training_data = make_data_loader(config.get('train_dataset'), tag='train')
    validation_data = make_data_loader(config.get('val_dataset'), tag='val')
  
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["optimizer"]["lr"])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config["multi_step_lr"]["milestones"],
                                                     gamma=config["multi_step_lr"]["gamma"])

    # torch.autograd.set_detect_anomaly(True)

    t = config['data_norm']["inp"]
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()

    t = config['data_norm']["gt"]
    gt_sub = torch.FloatTensor(t['sub']).view(1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1).cuda()
    
    for epoch in range(config["epochs"]):
        tot = 0
        model.train()


        for batch in tqdm(training_data, leave=False, desc="Epoch: "+ str(epoch)):
            for k, v in batch.items():
                batch[k] = v.cuda()

            

            inp = (batch['inp'] - inp_sub) / inp_div
            pred_rgb = model(inp, batch["coord"] , batch["cell"])

            gt = (batch['gt'] - gt_sub) / gt_div

            B, N_pxl_pts, _ = gt.shape 
            pred_rgb = pred_rgb.reshape(B, N_pxl_pts, 3)

            loss = loss_fn(pred_rgb, gt)

            tot+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
                    'model': model_spec,
                    'optimizer': optimizer_spec,
                    'epoch': epoch
                  }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        val_res = eval_psnr(validation_data, model_,
                            data_norm=config['data_norm'],
                            eval_bsize=config.get('eval_bsize'))
            

        print("Epoch:", epoch, "  Average Loss:", tot/len(training_data))
        print('val: psnr={:.4f}'.format(val_res))
        tot = 0
        scheduler.step()

        if epoch % config["save_every"]==0:
            torch.save(sv_file, config["save"]+str(epoch)+".pth")

if __name__ == "__main__":
    train()
