# - *- coding: utf- 8 - *-
import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#import scipy.io as sio
import SimpleITK as sitk
#import torchvision.transforms as tr
import random
import numpy as np
from itertools import combinations
import random


class Temporal_cardiac(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.t_dim = [0,0,0]

        list_order = [0,1,2,3,4]
        self.list_order = list(combinations(list_order,2))
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        self.list_weights = []

        for i in range(len(self.list_order)):
            self.list_weights.append(np.abs(self.list_order[i][0]-self.list_order[i][1]))

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            print(rotation_angle)
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([16,16,0])
        crop_filter.SetUpperBoundaryCropSize([16,16,0])
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)

            if self.rotate:
                img_size = imgdata.GetSize()
                origin = imgdata.GetOrigin()
                t_dim_tmp = self.t_dim
                t_dim_tmp[dim_index[0]] = 1
                t_dim_tmp = tuple(t_dim_tmp)
                transform = sitk.VersorTransform(t_dim_tmp, rotation_angle[0])
                transform.SetCenter((img_size[0]//2, img_size[1]//2, img_size[2]//2))
                imgdata = sitk.Resample(imgdata, imgdata.GetSize(),transform, sitk.sitkLinear, origin, imgdata.GetSpacing(), imgdata.GetDirection())
                imgdata.SetOrigin(origin)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -500.0, 600.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        num_list = self.list_order[index]
        weights_tmp = self.list_weights[index]

        source_img = self.img_list[num_list[0]].reshape((1,) + self.img_list[num_list[0]].shape)
        driving_img = self.img_list[num_list[1]].reshape((1,) + self.img_list[num_list[1]].shape)
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        out['driving'] = driving_img
        out['source'] = source_img
        out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out


class Temporal_read_new(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.t_dim = [0,0,0]

        list_order = [0,1,2,3,4]
        self.list_order = list(combinations(list_order,2))

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            print(rotation_angle)
        
        # slices
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            if self.rotate:
                img_size = imgdata.GetSize()
                origin = imgdata.GetOrigin()
                t_dim_tmp = self.t_dim
                t_dim_tmp[dim_index[i]] = 1
                t_dim_tmp = tuple(t_dim_tmp)
                transform = sitk.VersorTransform(t_dim_tmp, rotation_angle[i])
                transform.SetCenter((img_size[0]//2, img_size[1]//2, img_size[2]//2))
                imgdata = sitk.Resample(imgdata, imgdata.GetSize(),transform, sitk.sitkLinear, origin, imgdata.GetSpacing(), imgdata.GetDirection())
                imgdata.SetOrigin(origin)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            imgdata = np.clip(imgdata, -1000.0, 400.0)

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        num_list = self.list_order[index]

        source_img = self.img_list[num_list[0]].reshape((1,) + self.img_list[num_list[0]].shape)
        driving_img = self.img_list[num_list[1]].reshape((1,) + self.img_list[num_list[1]].shape)
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        out['driving'] = driving_img
        out['source'] = source_img

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out




class Temporal_cardiac_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.t_dim = [0,0,0]

        self.list_order = []
        self.list_order_new = []

        for i in range(int(len(image_list)/5)):
            list_order = [int(i*5), int(i*5+1), int(i*5+2), int(i*5+3), int(i*5+4)]
            list_order = [[int(i*5), int(i*5+2)], [int(i*5), int(i*5+3)], [int(i*5), int(i*5+4)], \
                [int(i*5+1), int(i*5+3)], [int(i*5), int(i*5+4)], [int(i*5+2), int(i*5+4)]]#list(combinations(list_order, 2))
            self.list_order = self.list_order + list_order
            list_order_new = [[float(1), float(2)], [float(1), float(2.5)], [float(1), float(3)], \
                [float(1.5), float(2.5)], [float(1.5), float(3)], [float(2), float(3)]]
            self.list_order_new = self.list_order_new + list_order_new
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        #print(self.list_order)
        self.list_weights = []

        for i in range(len(self.list_order)):
            tmp = 5 - np.abs(self.list_order[i][0]-self.list_order[i][1])
            tmp = (tmp / 8) + 1.0
            #tmp = (tmp / 4) + 1.0
            #self.list_weights.append()
            self.list_weights.append(tmp)

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            print(rotation_angle)
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)

            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[160.0,160.0,160.0]
            factorSize = np.asarray([160,160,160], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize([160,160,160])
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)

            if self.rotate:
                img_size = imgdata.GetSize()
                origin = imgdata.GetOrigin()
                t_dim_tmp = self.t_dim
                t_dim_tmp[dim_index[0]] = 1
                t_dim_tmp = tuple(t_dim_tmp)
                transform = sitk.VersorTransform(t_dim_tmp, rotation_angle[0])
                transform.SetCenter((img_size[0]//2, img_size[1]//2, img_size[2]//2))
                imgdata = sitk.Resample(imgdata, imgdata.GetSize(),transform, sitk.sitkLinear, origin, imgdata.GetSpacing(), imgdata.GetDirection())
                imgdata.SetOrigin(origin)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            if (imgdata.max() - imgdata.min())>100:
                imgdata = np.clip(imgdata, -500.0, 600.0)
                imgdata = (imgdata - imgdata.mean())/imgdata.std()

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())
            
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = self.list_order[index]
        weights_tmp = self.list_weights[index]

        source_img = self.img_list[num_list[0]].reshape((1,) + self.img_list[num_list[0]].shape)
        driving_img = self.img_list[num_list[1]].reshape((1,) + self.img_list[num_list[1]].shape)
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        out['driving'] = driving_img
        out['source'] = source_img
        out['driving_value'] = np.array(self.list_order_new[index][1]).astype(np.float32)
        out['source_value'] = np.array(self.list_order_new[index][0]).astype(np.float32)
        out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out


class Temporal_lung_all(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.t_dim = [0,0,0]

        self.list_order = []
        self.list_order_new = []

        for i in range(int(len(image_list)/5)):
            list_order = [int(i*5), int(i*5+1), int(i*5+2), int(i*5+3), int(i*5+4)]
            list_order = list(combinations(list_order, 2))
            self.list_order = self.list_order + list_order
        #self.list_order = [[0,1],[1,2],[2,3],[3,4]]
        #print(self.list_order)
        self.list_weights = []

        for i in range(len(self.list_order)):
            self.list_weights.append(5-np.abs(self.list_order[i][0]-self.list_order[i][1]))

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            print(rotation_angle)
        
        # slices
        
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -1000.0, 100.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = self.list_order[index]
        weights_tmp = self.list_weights[index]

        source_img = self.img_list[num_list[0]].reshape((1,) + self.img_list[num_list[0]].shape)
        driving_img = self.img_list[num_list[1]].reshape((1,) + self.img_list[num_list[1]].shape)
        
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_img = torch.from_numpy(source_img.astype(np.float32))
        driving_img = torch.from_numpy(driving_img.astype(np.float32))

        out['driving'] = driving_img
        out['source'] = source_img
        out['deformed_weights'] = weights_tmp.astype(np.float32)

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out




class Temporal_cardiac_test(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, job, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.img_norm_list = []
        self.img_body = []
        self.t_dim = [0,0,0]

        self.list_order = image_list

        if self.rotate:
            rotation_angle = np.random.rand(5)*30.0*np.pi/180.0
            dim_index = np.random.randint(3,size=5)
            #self.t_dim[dim_index] = 1
            #self.t_dim = tuple(self.t_dim)
            print(rotation_angle)
        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)

            spacing = np.asarray(imgdata.GetSpacing())*np.asarray(imgdata.GetSize(), dtype=float)/[160.0,160.0,160.0]
            factorSize = np.asarray([160,160,160], dtype=int)
            T = sitk.AffineTransform(3)
            T.SetMatrix(imgdata.GetDirection())
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgdata)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize([160,160,160])
            resampler.SetInterpolator(sitk.sitkBSpline)
            imgdata = resampler.Execute(imgdata)

            if self.rotate:
                img_size = imgdata.GetSize()
                origin = imgdata.GetOrigin()
                t_dim_tmp = self.t_dim
                t_dim_tmp[dim_index[0]] = 1
                t_dim_tmp = tuple(t_dim_tmp)
                transform = sitk.VersorTransform(t_dim_tmp, rotation_angle[0])
                transform.SetCenter((img_size[0]//2, img_size[1]//2, img_size[2]//2))
                imgdata = sitk.Resample(imgdata, imgdata.GetSize(),transform, sitk.sitkLinear, origin, imgdata.GetSpacing(), imgdata.GetDirection())
                imgdata.SetOrigin(origin)
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            if norm:
                imgdata = imgdata / 10
            imgdata = np.clip(imgdata, -500.0, 600.0)

            #print(imgdata.max(), imgdata.min(), imgdata.mean(), imgdata.std())

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            #print(imgdata.std())
            #print(image_list[i])
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.list_order)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        #print(self.list_order[index])
        num_list = index

        source_real_img = self.img_list[num_list].reshape((1,) + self.img_list[num_list].shape)
        driving_real_img = self.img_list[num_list].reshape((1,) + self.img_list[num_list].shape)

        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        out = {}
        source_real_img = torch.from_numpy(source_real_img.astype(np.float32))
        driving_real_img = torch.from_numpy(driving_real_img.astype(np.float32))

        out['driving'] = driving_real_img
        out['source'] = source_real_img

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out


class Temporal_Graph(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        #assert job in self.suitableJobs, 'not suitable jobs'
        #self.job = job
        self.rotate = rotate
        self.image_list = image_list

        
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        dic_tmp = self.image_list[index]

        with open(dic_tmp, 'rb') as outfile:
            dict_data = pickle.load(outfile)

        vertex = np.array(dict_data['vertext'])
        edge = np.array(dict_data['edge'])

        
        out = {}
        vertex = torch.from_numpy(vertex.astype(np.float32))
        edge = torch.from_numpy(edge.astype(np.float32))

        out['vertex'] = vertex
        out['edge'] = edge

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out







class Temporal_Graph_value(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, \
            spacing=None, crop=None, ratio=None, rotate=True, \
            include_slices=None, norm=False):
        #assert job in self.suitableJobs, 'not suitable jobs'
        #self.job = job
        self.rotate = rotate
        self.image_list = image_list

        
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        dic_tmp = self.image_list[index]

        with open(dic_tmp, 'rb') as outfile:
            dict_data = pickle.load(outfile)

        vertex = np.array(dict_data['vertext'])
        edge = np.array(dict_data['edge'])

        
        out = {}
        vertex = torch.from_numpy(vertex.astype(np.float32))
        edge = torch.from_numpy(edge.astype(np.float32))

        out['vertex'] = vertex
        out['edge'] = edge

        #index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return out




class Temporal_read_pre(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image_list, num_l, job, \
            spacing=None, crop=None, ratio=None, rotate=False, \
            include_slices=None):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        self.rotate = rotate
        self.img_list = []
        self.num_l = num_l
        self.t_dim = [0,0,0]

        
        # slices
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([16,16,0])
        crop_filter.SetUpperBoundaryCropSize([16,16,0])
        for i in range(len(image_list)):
            imgdata = sitk.ReadImage(image_list[i])
            imgdata = crop_filter.Execute(imgdata)
            
            imgdata = sitk.GetArrayFromImage(imgdata)
            imgdata = imgdata.astype(np.float64)
            imgdata = np.clip(imgdata, -500.0, 600.0)

            imgdata = (imgdata - imgdata.mean())/imgdata.std()
            self.img_list.append(imgdata)

        
        

    def __len__(self):
        return len(self.num_l)

    def __getitem__(self, index):
        # image
        #image_data0 = self.img_list[0]
        #image_data9 = self.img_list[-1]
        num_list = self.num_l[index]
        

        

        imgs_list = [self.img_list[img_num].reshape((1,) + self.img_list[img_num].shape) for img_num in num_list]

        #img_list = [img.reshape((1,) + img.shape) for img in self.img_list]

        imgs_list = np.concatenate(imgs_list,0)
        
        #size = np.array(image_data.shape)
        # one channel image
        #image_data = image_data.reshape((1,) + image_data.shape)
        image_data = torch.from_numpy(imgs_list.astype(np.float32))
        index_list = torch.from_numpy(num_list.astype(np.int64))
        # label
        
        return image_data, index_list
