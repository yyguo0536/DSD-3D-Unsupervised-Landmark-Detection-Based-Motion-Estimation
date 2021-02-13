from tqdm import trange
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from logger import Logger
from modules.model_3d import GeneratorFullModel, DiscriminatorFullModel
from modules.util_3d import AntiAliasInterpolation3d, make_coordinate_grid
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater
from niidata import *
import random
from itertools import combinations
import pickle

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")



class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 3, 4])).to(device0)#.cuda()
        #noise[:,:,2] = 0
        #noise[:,2,:2] = 0
        self.theta = noise + torch.eye(3, 4).view(1, 3, 4).to(device0)# + 
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps'], kwargs['points_tps']), type=noise.type()).to(device0)#.cuda()
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 3])).to(device0)#.cuda()
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).to(device0)#.cuda()
        grid = grid.view(1, frame.shape[2] * frame.shape[3] * frame.shape[4], 3)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], frame.shape[4], 3)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :3], coordinates.unsqueeze(-1)) #+ theta[:, :, :, 3:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 3) - control_points.view(1, 1, -1, 3)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed







def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    pkl_dir = 'load the reference landmarks *.pickle'
    
    
    with open(pkl_dir, 'rb') as outfile:
        dict_data = pickle.load(outfile)
    

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params, config['model_params']['common_params'], dict_data)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params, dict_data)



    img_list = []
    for index_patient, series_img in enumerate(dataset):
        series_img_list = []
        for k in range(5):
            series_img_list.append(series_img+'/image_data/t'+str(k+1)+'.nii')
        img_list = img_list + series_img_list


    trainning_list = Temporal_cardiac_all(img_list, job='seg', rotate=False)
    train_dataloader = DataLoader(trainning_list, batch_size=train_params['batch_size'], \
        shuffle=True, num_workers=6, drop_last=True)


    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):

            for index_num, x in enumerate(train_dataloader):

                index_tmp = np.random.randint(2)

                transform = Transform(x['driving'].shape[0], **config['train_params']['augmentation_params'])
                if index_tmp == 0:
                    tmp = x['source']
                    x['source'] = transform.transform_frame(x['driving'].to(device0)).cpu()
                    x['driving'] = transform.transform_frame(tmp.to(device0)).cpu()
                else:
                    x['driving'] = transform.transform_frame(x['driving'].to(device0)).cpu()
                    x['source'] = transform.transform_frame(x['source'].to(device0)).cpu()
                
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                print(epoch, index_num, losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'pointdisc': pointdisc,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
