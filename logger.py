import numpy as np
import torch
import torch.nn.functional as F
import imageio
from modules.util_3d import make_coordinate_grid
import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
import SimpleITK as sitk

class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image, name_list = self.visualizer.visualize(inp['driving'], inp['source'], out)

        for img, img_name in zip(image, name_list):
            if type(img) == tuple:
                sitk.WriteImage(img[0], '/'.join([self.visualizations_dir, img_name[0]]))
                sitk.WriteImage(img[1], '/'.join([self.visualizations_dir, img_name[1]]))
            else:
                sitk.WriteImage(img, '/'.join([self.visualizations_dir, img_name]))
        #imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer_generator=None, optimizer_discriminator=None, \
                 optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.scale = 1
        self.colormap = plt.get_cmap(colormap)

    def heatmap_with_kp(self, heatmap):
        """
        Transform a keypoint into gaussian like representation
        """
        #mean = kp['value']#.to(device0)

        num_chanl = heatmap.shape[-1]

        #heatmap[heatmap>0.3] = 1
        heatmap = np.where(heatmap > 0.5, 1, 0)

        for i in range(num_chanl):
            heatmap[:,:,:,:,i] = heatmap[:,:,:,:,i] * (i + 1)

        heatmap = np.sum(heatmap, axis=4)
        heatmap = heatmap[:,:,:,:,np.newaxis]

        return heatmap.astype(np.int32)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        landmark = np.zeros_like(image)
        kp_array = kp_array[0,:]
        spatial_size = np.array(image.shape[1:4][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        #kp_array = kp_array * self.scale
        
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            kp_x = int(np.round(kp[0]))
            kp_y = int(np.round(kp[1]))
            kp_z = int(np.round(kp[2]))
            kp_x_low = (kp_x - 2) if (kp_x - 2)>0 else 0
            kp_x_up = (kp_x + 3) if (kp_x + 3)<image.shape[-2] else image.shape[-2]

            kp_y_low = (kp_y - 2) if (kp_y - 2)>0 else 0
            kp_y_up = (kp_y + 3) if (kp_y + 3)<image.shape[-3] else image.shape[-3]

            kp_z_low = (kp_z - 2) if (kp_z - 2)>0 else 0
            kp_z_up = (kp_z + 3) if (kp_z + 3)<image.shape[-4] else image.shape[-4]
            #rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            landmark[:,kp_z_low:kp_z_up, kp_y_low:kp_y_up, kp_x_low:kp_x_up, :] = kp_ind+1
        return image, landmark

    def create_image_column_with_kp(self, images, kp):
        landmark_array = self.heatmap_with_kp(kp) #for v, k in zip(images, kp)
        return (self.create_image_column(images), self.create_image_column(landmark_array))

    def create_image_column(self, images):
        #if self.draw_border:
        images = np.copy(images)
        return sitk.GetImageFromArray(images[0,:,:,:,:])

    def create_image_grid(self, args, args_name):
        out = []
        out_name = []
        for arg, arg_name in zip(args, args_name):
        #for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
                out_name.append(arg_name)
            else:
                out.append(self.create_image_column(arg))
                out_name.append(arg_name)
        return out, out_name

    def visualize(self, driving, source, out):
        images = []
        images_name = []

        # Source image with keypoints
        _, _, d, h, w = source.shape
        source = source.data.cpu()
        kp_source = F.interpolate(out['kp_source']['heatmap'], size=(d, h, w), mode='trilinear').data.cpu().numpy()
        #kp_source = out['kp_source']['heatmap'].data.cpu().numpy()
        kp_source = np.transpose(kp_source, [0, 2, 3, 4, 1])
        source = np.transpose(source, [0, 2, 3, 4, 1])
        images.append((source, kp_source))
        images_name.append(('source_img.nii', 'source_img_kp.nii'))


        source_pred = F.interpolate(out['kp_source']['predicted_map'], size=(d, h, w), mode='trilinear').data.cpu().numpy()
        source_pred = np.transpose(kp_source, [0, 2, 3, 4, 1])


        # Equivariance visualization
        if 'transformed_frame' in out:
            _, _, d, h, w = out['transformed_frame'].shape
            transformed = out['transformed_frame'].data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 4, 1])
            transformed_kp = F.interpolate(out['transformed_kp']['heatmap'], size=(d, h, w), mode='trilinear').data.cpu().numpy()
            #transformed_kp = out['transformed_kp']['heatmap'].data.cpu().numpy()
            transformed_kp = np.transpose(transformed_kp, [0, 2, 3, 4, 1])
            images.append((transformed, transformed_kp))
            images_name.append(('transformed_frame.nii', 'transformed_kp.nii'))

        # Driving image with keypoints
        _, _, d, h, w = driving.shape
        kp_driving = F.interpolate(out['kp_driving']['heatmap'], size=(d, h, w), mode='trilinear').data.cpu().numpy()
        #kp_driving = out['kp_driving']['heatmap'].data.cpu().numpy()
        kp_driving = np.transpose(kp_driving, [0, 2, 3, 4, 1])
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 4, 1])
        images.append((driving, kp_driving))
        images_name.append(('driving_img.nii', 'driving_img_kp.nii'))

        

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 4, 1])
            images.append(deformed)
            images_name.append('deformed_img.nii')
        if 'deformation' in out:
            deformation = out['deformation'].data.cpu().numpy()
            deformation = np.transpose(deformation, [0,2,3,4,1])
            images.append(deformation)
            images_name.append('deformation.nii')

        # TPS initial deformed image
        if 'initial_deformed' in out:
            deformed = out['initial_deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 4, 1])
            images.append(deformed)
            images_name.append('initial_deformed_img.nii')

        if 'ref_hp' in out:
            #calibrated_hp = out['calibrated_hp'].data.cpu().numpy()
            calibrated_hp = F.interpolate(out['ref_hp'], size=(d, h, w), mode='trilinear').data.cpu().numpy()
            calibrated_hp = np.transpose(calibrated_hp, [0, 2, 3, 4, 1])
            images.append((source, calibrated_hp))
            images_name.append(('source_img.nii', 'ref_kp.nii'))

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 4, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].data.cpu().numpy()
            images.append((prediction, kp_norm))
            images_name.append(('prediction_img.nii', 'prediction_img_kp.nii'))
        images.append(prediction)
        images_name.append('prediction_img.nii')


        ## Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = out['occlusion_map'].data.cpu()#.repeat(1, 3, 1, 1)
            self.scale = source.shape[-2] / occlusion_map.shape[-1]
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:4]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 4, 1])
            images.append(occlusion_map)
            images_name.append('occlusion_map_img.nii')


        image, name_list = self.create_image_grid(images, images_name)
        #image = (255 * image).astype(np.uint8)
        return image, name_list
