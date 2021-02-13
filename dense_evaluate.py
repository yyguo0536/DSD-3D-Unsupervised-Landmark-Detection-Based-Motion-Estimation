import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback
from niidata import *
from create_graph import *
import pandas as pd
import json
import pickle
import torch.nn.functional as F
from modules.util_3d import AntiAliasInterpolation3d, make_coordinate_grid, kp2gaussian
import SimpleITK as sitk
from quantify_eva import analyse_deform, dice_eva
from eval_data import *

from demons_refine import refine_def
#from BSpline_refine import bsplineRefine

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

def dice_eva(msk1, msk2):
    smooth = 1
    iflat = msk1.reshape(-1)
    tflat = msk2.reshape(-1)
    intersection = (iflat * tflat).sum()

    A_sum = np.sum(iflat * iflat)
    B_sum = np.sum(tflat * tflat)

    return ((2. * intersection+smooth)/(A_sum+B_sum+smooth))

def deform_input(inp, deformation, vec_def):
    _, d_old, h_old, w_old, _ = deformation.shape
    _, _, d, h, w = inp.shape
    if h_old != h or w_old != w or d_old != d:
        deformation = deformation.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
        deformation = deformation.permute(0, 2, 3, 4, 1)
        vec_def = vec_def.permute(0, 4, 1, 2, 3)
        vec_def = F.interpolate(vec_def, size=(d, h, w), mode='trilinear')
        vec_def = vec_def.permute(0, 2, 3, 4, 1)
    return F.grid_sample(inp, deformation), vec_def


def save_kp_img(img_dir, img, kp_map):
    spatial_size = img.shape[2:]
    kp_map = kp2gaussian(kp_map, spatial_size=spatial_size, kp_variance=0.01)
    #kp_map = F.interpolate(kp_map, size=img.shape[2:], mode='trilinear')
    kp_map = np.transpose(kp_map.cpu().data.numpy(), [0, 2, 3, 4, 1])
    img = np.transpose(img.cpu().data.numpy(), [0, 2, 3, 4, 1])
    #kp_map = kp_map[0,:,:,:,:].cpu().data.numpy()

    num_chanl = kp_map.shape[-1]

    #heatmap[heatmap>0.3] = 1
    kp_map = np.where(kp_map > 0.7, 1, 0)

    for i in range(num_chanl):
        kp_map[:,:,:,:,i] = kp_map[:,:,:,:,i] * (i + 1)

    kp_map = np.sum(kp_map, axis=4)
    kp_map = kp_map[:,:,:,:,np.newaxis]

    kp_map = kp_map.astype(np.int32)

    kp_img = sitk.GetImageFromArray(kp_map[0,:,:,:,:])
    img = sitk.GetImageFromArray(img[0,:,:,:,:])

    kp_name = img_dir+'landmark.nii'
    img_name = img_dir+'img.nii'

    sitk.WriteImage(img, img_name)
    sitk.WriteImage(kp_img, kp_name)

def distance(p_lists):
    dis_val = []
    edge_val = []
    for i in range(len(p_lists)):
        tmp_edge = []

        ed = p_lists - p_lists[i,:]
        ed = torch.pow(ed, 2).sum(dim=-1)
        values, indices = torch.sort(ed)
        for k in range(3):
            tmp_edge.append([i, indices[k+1].cpu().data.numpy().tolist()])
            dis_val.append(values[k+1].cpu().data.numpy().tolist())

        edge_val.append(tmp_edge)

    return dis_val, edge_val

def distance_group(p_lists):
    dis_val = []
    edge_val = []
    for k in range(p_lists[0].shape[0]):
        tmp_edge = []
        for i in range(len(p_lists)):
            if i == 0:
                ed = p_lists[i] - p_lists[i][k,:]
            else:
                ed = ed + p_lists[i] - p_lists[i][k,:]

        ed = torch.pow(ed, 2).sum(dim=-1)
        values, indices = torch.sort(ed)
        for kk in range(3):
            tmp_edge.append([k, indices[kk+1].cpu().data.numpy().tolist()])
            dis_val.append(values[kk+1].cpu().data.numpy().tolist())

        edge_val.append(tmp_edge)

    return edge_val





def save_kp_img(img_dir, img, kp_map):
    kp_map = F.interpolate(kp_map, size=img.shape[2:], mode='trilinear')
    kp_map = np.transpose(kp_map.cpu().data.numpy(), [0, 2, 3, 4, 1])
    img = np.transpose(img.cpu().data.numpy(), [0, 2, 3, 4, 1])
    #kp_map = kp_map[0,:,:,:,:].cpu().data.numpy()

    num_chanl = kp_map.shape[-1]

    #heatmap[heatmap>0.3] = 1
    kp_map = np.where(kp_map > 0.7, 1, 0)

    for i in range(num_chanl):
        kp_map[:,:,:,:,i] = kp_map[:,:,:,:,i] * (i + 1)

    kp_map = np.sum(kp_map, axis=4)
    kp_map = kp_map[:,:,:,:,np.newaxis]

    kp_map = kp_map.astype(np.int32)

    kp_img = sitk.GetImageFromArray(kp_map[0,:,:,:,:])
    img = sitk.GetImageFromArray(img[0,:,:,:,:])

    kp_name = img_dir+'landmark.nii'
    img_name = img_dir+'img.nii'

    sitk.WriteImage(img, img_name)
    sitk.WriteImage(kp_img, kp_name)



def eval_deform(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'dense_eval/png')

    kp_dir = os.path.join(log_dir, 'kp_img/')
    graph_dir = os.path.join(log_dir, 'graph_file/')
    pkl_dir = os.path.join(log_dir, 'graph_json/')

    log_dir_new = os.path.join(log_dir, 'dense_eval')
    maxPool = torch.nn.MaxPool3d(5,1,2).cuda()

    dir_img = ['/image_data/t1.nii', '/image_data/t2.nii', '/image_data/t3.nii', '/image_data/t4.nii', '/image_data/t5.nii']
    dir_msk = ['/label_data/t1_label.nii', '/label_data/t2_label.nii', \
        '/label_data/t3_label.nii', '/label_data/t4_label.nii', '/label_data/t5_label.nii']

    graph_list = pd.read_csv(\
        log_dir + '/graph.csv')
    base_dir = '/'.join(graph_list['graph'].tolist()[0].split('/')[:-1])

    gap_num = 2

    valid_data = []
    valid_img = []
    patient_list = []

    valid_list = []

    for k in range(len(valid_list)):
        valid_data.append(base_dir+'/'+str(valid_list[k])+'.pickle')
        valid_img.append(dataset[valid_list[k]])

        tmp_patient = dataset[valid_list[k]].split('/')[-1]
        patient_list.append(tmp_patient)



    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='eval'.")
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir_new):
        os.makedirs(log_dir_new)
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    mse_list = []
    psnr_list = []
    ssim_list = []
    nrmse_list = []
    dice_list = []
    refined_dice = []
    hd_list_refined = []

    amsd_list = []
    mmsd_list = []
    rmsd_list = []
    hd_list = []

    #dataset_loader = Temporal_Graph(valid_data)
    #dataloader = DataLoader(dataset_loader, batch_size=1, shuffle=False, num_workers=1)


    for it, valid_patient in enumerate(valid_data):
        with open(valid_patient, 'rb') as outfile:
            dict_data = pickle.load(outfile)

        vertex = np.array(dict_data['vertext'])
        edge = np.array(dict_data['edge'])

        
        out = {}
        vertex = torch.from_numpy(vertex.astype(np.float32))
        edge = torch.from_numpy(edge.astype(np.float32))

        out['vertex'] = vertex
        out['edge'] = edge
            

        #valid_dataset = Temporal_Graph_value(graph_list['graph'][-1])
        #valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

        patient_name = patient_list[it]
        img_dir = valid_img[it] + dir_img[0]


        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize([8,8,0])
        crop_filter.SetUpperBoundaryCropSize([8,8,0])

        imgdata = sitk.ReadImage(img_dir)
        imgdata = sitk.GetArrayFromImage(imgdata)
        imgdata = sitk.GetImageFromArray(imgdata)

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

        imgdata = sitk.GetArrayFromImage(imgdata)
        imgdata = imgdata.astype(np.float32)
        imgdata_or = imgdata
        imgdata = np.clip(imgdata, -500.0, 600.0)


        
        seg_tmp = valid_img[it] + dir_msk[0]

        initial_msk = sitk.ReadImage(seg_tmp)
        initial_msk = sitk.GetArrayFromImage(initial_msk)
        initial_msk = sitk.GetImageFromArray(initial_msk)

        initial_msk = crop_filter.Execute(initial_msk)
        resampler.SetInterpolator(sitk.sitkLabelGaussian)
        initial_msk = resampler.Execute(initial_msk)

        initial_msk = sitk.GetArrayFromImage(initial_msk)
        initial_msk = initial_msk.astype(np.float64)
        initial_msk = torch.from_numpy(initial_msk.astype(np.float32)).to(device0)
        initial_msk = initial_msk.unsqueeze(0).unsqueeze(0)

        imgdata = (imgdata - imgdata.mean())/imgdata.std()
        imgdata = torch.from_numpy(imgdata.astype(np.float32)).to(device0)
        imgdata = imgdata.unsqueeze(0).unsqueeze(0)

        kp_source = {}
        kp_driving = {}
        kp_source['value'] = out['vertex'][0,:,:].unsqueeze(0).to(device0)
        


        for i in range(4):
            seg_tmp = valid_img[it] + dir_msk[i+1]
            target_msk = sitk.ReadImage(seg_tmp)
            target_msk = sitk.GetArrayFromImage(target_msk)
            target_msk = sitk.GetImageFromArray(target_msk)

            target_img = sitk.ReadImage(valid_img[it] + dir_img[i+1])
            target_img = sitk.GetArrayFromImage(target_img)
            target_img = sitk.GetImageFromArray(target_img)


            target_msk = crop_filter.Execute(target_msk)
            target_img = crop_filter.Execute(target_img)

            resampler.SetInterpolator(sitk.sitkLabelGaussian)
            target_msk = resampler.Execute(target_msk)

            resampler.SetInterpolator(sitk.sitkBSpline)
            target_img = resampler.Execute(target_img)
            target_img = sitk.GetArrayFromImage(target_img)
            target_img = target_img.astype(np.float32)
            target_img = np.clip(target_img, -500.0, 600.0)
            target_img_or = target_img
            target_img = (target_img - target_img.mean())/target_img.std()

            target_msk = sitk.GetArrayFromImage(target_msk)
            target_msk = target_msk.astype(np.float64)
            #kp_driving['value'] = kp_out[i][:,:,:3].to(device0) + out['vertex'][0,:,:].unsqueeze(0).cuda()
            kp_driving['value'] = out['vertex'][i+1,:,:].unsqueeze(0).to(device0)

            outdict = generator(imgdata, kp_source=kp_source, kp_driving=kp_driving)
            deformed = outdict['deformed']
            deformation = outdict['deformation']
            vec_def = outdict['vec_deformation']
            deformed_msk, vec_def = deform_input(initial_msk, deformation, vec_def)
            deformed_msk = deformed_msk.cpu().data

            refine_demons = True

            if refine_demons:
                fixed = sitk.GetImageFromArray(target_img_or)
                moving = sitk.GetImageFromArray(imgdata_or)
                vec_def = vec_def[0,:,:,:,:].cpu().data.numpy() * vec_def.shape[3] / 2
                vec_def_ini = sitk.GetImageFromArray(vec_def.astype(np.float64))
                outTx_ini = sitk.DisplacementFieldTransform(vec_def_ini)
                vec_def = sitk.GetImageFromArray(vec_def.astype(np.float64))

                #outTx = bsplineRefine(fixed, moving, vec_def)
                outTx = refine_def(fixed, moving, vec_def, i+1)
                
                resampler_def = sitk.ResampleImageFilter()
                resampler_def.SetReferenceImage(fixed)
                resampler_def.SetInterpolator(sitk.sitkLinear)
                resampler_def.SetTransform(outTx)
                moved_img = resampler_def.Execute(moving)
                moving_msk = sitk.GetImageFromArray(initial_msk[0,0,:,:,:].cpu().data.numpy())
                resampler_def.SetInterpolator(sitk.sitkLabelGaussian)
                deformed_msk_refine = resampler_def.Execute(moving_msk)

                resampler_def.SetTransform(outTx_ini)
                resampler_def.SetInterpolator(sitk.sitkLinear)
                moved_img_ini = resampler_def.Execute(moving)
                #moving_msk_array = sitk.GetArrayFromImage(moving_msk)
                deformed_msk_array = sitk.GetArrayFromImage(deformed_msk_refine)
                dice_tmp = 2*(target_msk*deformed_msk_array).sum()/(target_msk.sum()+deformed_msk_array.sum())
                refined_dice.append(dice_tmp)

                hd_eval = Surface(deformed_msk_array.astype(np.int32), \
                    target_msk.astype(np.int32))
                hd_list_refined.append(hd_eval.hausdorff())

                moved_array=sitk.GetArrayFromImage(moved_img)
                moved_array = (moved_array - moved_array.mean())/moved_array.std()
                moved_img = sitk.GetImageFromArray(moved_array)

                moved_array=sitk.GetArrayFromImage(moved_img_ini)
                moved_array = (moved_array - moved_array.mean())/moved_array.std()
                moved_img_ini = sitk.GetImageFromArray(moved_array)


                print(dice_tmp)
        

            deformed_msk_img = sitk.GetImageFromArray(deformed_msk[0,0,:,:,:].numpy())

            sitk.WriteImage(deformed_msk_img, png_dir+'/'+patient_name+'_0to'+str(i+1)+'_msk.nii')

            deformed_msk = torch.where(deformed_msk>0.6, torch.Tensor([1.0]), torch.Tensor([0.0]))

            deformed_msk_img_binary = sitk.GetImageFromArray(deformed_msk[0,0,:,:,:].numpy())

            sitk.WriteImage(deformed_msk_img_binary, png_dir+'/'+patient_name+'_0to'+str(i+1)+'_bimsk.nii')

            target_msk_img_binary = sitk.GetImageFromArray(target_msk)

            sitk.WriteImage(target_msk_img_binary, png_dir+'/'+patient_name+'_'+str(i+1)+'_bimsk.nii')

            dice_list.append(dice_eva(deformed_msk[0,0,:,:,:].numpy(), target_msk))


            ground_spacing = [1.0,1.0,1.0]

            amsd_eval = Surface(deformed_msk[0,0,:,:,:].numpy().astype(np.int32), \
                target_msk.astype(np.int32), \
                physical_voxel_spacing=ground_spacing)
            mmsd_eval = Surface(deformed_msk[0,0,:,:,:].numpy().astype(np.int32), \
                target_msk.astype(np.int32), \
                physical_voxel_spacing=ground_spacing)
            rmsd_eval = Surface(deformed_msk[0,0,:,:,:].numpy().astype(np.int32), \
                target_msk.astype(np.int32), \
                physical_voxel_spacing=ground_spacing)
            hd_eval = Surface(deformed_msk[0,0,:,:,:].numpy().astype(np.int32), \
                target_msk.astype(np.int32), \
                physical_voxel_spacing=ground_spacing)
            hd_list.append(hd_eval.hausdorff())

            amsd_list.append(amsd_eval.get_average_symmetric_surface_distance())
            mmsd_list.append(mmsd_eval.get_maximum_symmetric_surface_distance())
            rmsd_list.append(rmsd_eval.get_root_mean_square_symmetric_surface_distance())

            #ground_spacing = [test_info.GetSpacing()[2], test_info.GetSpacing()[0], test_info.GetSpacing()[1]]

            #amsd_eval = Surface(pred_numpy, groundtruth_numpy, physical_voxel_spacing=ground_spacing)
            #mmsd_eval = Surface(pred_numpy, groundtruth_numpy, physical_voxel_spacing=ground_spacing)
            #rmsd_eval = Surface(pred_numpy, groundtruth_numpy, physical_voxel_spacing=ground_spacing)
            #hausdorff_eval = Surface(pred_numpy, groundtruth_numpy)
                
            #amsd_data.append(amsd_eval.get_average_symmetric_surface_distance())
            #mmsd_data.append(mmsd_eval.get_maximum_symmetric_surface_distance())
            #rmsd_data.append(rmsd_eval.get_root_mean_square_symmetric_surface_distance())


            image_data = sitk.ReadImage(valid_img[it]+dir_img[i+1])
            image_data = sitk.GetArrayFromImage(image_data)
            image_data = sitk.GetImageFromArray(image_data)
            image_data = crop_filter.Execute(image_data)
            resampler.SetInterpolator(sitk.sitkBSpline)
            image_data = resampler.Execute(image_data)


            image_data = sitk.GetArrayFromImage(image_data)
            image_data = image_data.astype(np.float64)
            image_data = np.clip(image_data, -500.0, 600.0)
            image_data = (image_data - image_data.mean())/image_data.std()
            image_data = image_data.astype(np.float32)

            deformed_img = sitk.GetImageFromArray(deformed[0,0,:,:,:].cpu().data.numpy())
            real_img = sitk.GetImageFromArray(image_data)
            moving_img = sitk.GetImageFromArray(imgdata[0,0,:,:,:].cpu().data.numpy())
            sitk.WriteImage(real_img, png_dir+'/'+patient_name+'_'+str(i+1)+'.nii')
            sitk.WriteImage(deformed_img, png_dir+'/'+patient_name+'_0to'+str(i+1)+'.nii')
            sitk.WriteImage(moved_img, png_dir+'/'+patient_name+'_0to'+str(i+1)+'_refined_new.nii')
            #sitk.WriteImage(moved_img_ini, png_dir+'/'+patient_name+'_0to'+str(i+1)+'_ini.nii')
            sitk.WriteImage(moving_img, png_dir+'/'+patient_name+'_0.nii')

    s_dice_list = []
    #s_mse_list = []
    #s_psnr_list = []
    #s_ssim_list = []
    #s_nrmse_list = []

    for k in range(4):
        #tmp_mse = []
        #tmp_psnr = []
        #tmp_ssim = []
        #tmp_nrmse = []
        tmp_dice = []
        for i in range(2):
            #tmp_mse.append(mse_list[k + i * 4])
            #tmp_psnr.append(psnr_list[k + i * 4])
            #tmp_ssim.append(ssim_list[k + i * 4])
            #tmp_nrmse.append(nrmse_list[k + i * 4])
            tmp_dice.append(dice_list[k + i * 4])

        s_dice_list.append(tmp_dice)
        #s_mse_list.append(tmp_mse)
        #s_psnr_list.append(tmp_psnr)
        #s_ssim_list.append(tmp_ssim)
        #s_nrmse_list.append(tmp_nrmse)

    #df = pd.DataFrame({'mse':mse_list, 'psnr':psnr_list,'ssim':ssim_list,'nrmse':nrmse_list,'dice':dice_list})
    #df.to_csv(log_dir_new+"/deform.csv", index=False)
    print(hd_list)
    print(np.array(hd_list).mean())
    df = pd.DataFrame({'dice':dice_list, 'hd':hd_list, 'dice_refine':refined_dice, 'hd_refine':hd_list_refined})
    df.to_csv(log_dir_new+"/deform_new.csv", index=False)







