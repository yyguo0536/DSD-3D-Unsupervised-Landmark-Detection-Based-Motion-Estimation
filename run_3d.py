import matplotlib

matplotlib.use('Agg')

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

#from frames_dataset import FramesDataset

from modules.generator_3d import OcclusionAwareGenerator
from modules.discriminator_3d import MultiScaleDiscriminator, Discriminator
from modules.keypoint_detector_3d import KPDetector

import torch

from train_3d import train
from reconstruction import reconstruction
from animate import animate
from atlas_transfer import atlas
from transfer import transfer
from pretext import pretext
from pretext_vis import pretext_vis
from niidata import *
import pandas as pd
from dense_evaluate import eval_deform


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")



data = pd.read_csv('load your 4D dataset')

image_l = data['img']
mask_l = data['msk']
patient_name = []

for i in range(len(image_l)):
    patient_name.append(image_l[i].split('/')[-1])
    base_dir = '/'.join(image_l[i].split('/')[:-1]) 

patient_name = set(patient_name)
patient_name = list(patient_name)

patient_name.sort()
for k in range(len(patient_name)):
    patient_name[k] = '/'.join([base_dir, patient_name[k]])




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "pretext", "train_graph", \
        "reconstruction", "animate", "deform_eval", "transfer", "motion_inter", "eval_iter"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--crossvalid", default='all', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--object", default="cardiac", help="dataset")
    parser.add_argument("--loadpre", action="store_true")
    parser.add_argument("--loadnum", default=100)
    parser.set_defaults(verbose=False)


    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += opt.crossvalid
        #log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    if opt.mode == 'pretext':
        pretext(config, log_dir, patient_name)
    else:
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

        if torch.cuda.is_available():
            generator.to(device0)
        if opt.verbose:
            print(generator)

        discriminator = Discriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
        if torch.cuda.is_available():
            discriminator.to(device0)
        if opt.verbose:
            print(discriminator)

        if torch.cuda.is_available():
            pointdisc.to(device0)
        if opt.verbose:
            print(pointdisc)

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

        if torch.cuda.is_available():
            kp_detector.to(device1)

        if opt.mode == 'train':
            if opt.loadpre:
                pretext_dir = os.path.join(log_dir, 'pretext')
                pretext_dir = pretext_dir + '/epoch' + opt.loadnum
                tmp = torch.load(pretext_dir)
                model_dict = kp_detector.predictor.encoder.state_dict()
                pretrained_dict = {k: v for k, v in tmp.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                kp_detector.predictor.encoder.load_state_dict(model_dict)

        if opt.verbose:
            print(kp_detector)


        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)
   

        
        #dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    

        if opt.mode == 'train':
            print("Training...")
            if opt.object == 'cardiac':
                train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, patient_name, opt.device_ids)
            else:
                train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, temporal_list, opt.device_ids)
        elif opt.mode == 'reconstruction':
            print("Reconstruction...")
            #reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, temporal_list)
            reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, patient_name)
        elif opt.mode == 'animate':
            print("Animate...")
            animate(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, patient_name)
        elif opt.mode == 'atlas':
            print("Atlas...")
            atlas(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, patient_name)
        elif opt.mode == 'transfer':
            print("Transfer...")
            transfer(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, patient_name)
        elif opt.mode == 'deform_eval':
            print('Eval...')
            with torch.no_grad():
                eval_deform(config, generator, kp_detector, opt.checkpoint, log_dir, patient_name)
        '''elif opt.mode == 'motion_inter':
            print('Interpolation')
            motion_inter(config, generator, kp_detector, opt.checkpoint, log_dir, patient_name)
        elif opt.mode == 'train_graph':
            print('CreatGraph')
            train_graph(config, log_dir)
        elif opt.mode == 'eval_iter':
            print('Evaluation iteratively')
            eval_deform_iter(config, generator, kp_detector, opt.checkpoint, log_dir, patient_name)'''
