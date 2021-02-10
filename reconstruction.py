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

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

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
    kp_map = np.where(kp_map > 0.5, 1, 0)

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



def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')

    kp_dir = os.path.join(log_dir, 'kp_img/')
    graph_dir = os.path.join(log_dir, 'graph_file/')
    pkl_dir = os.path.join(log_dir, 'graph_json/')

    log_dir_old = log_dir

    log_dir = os.path.join(log_dir, 'reconstruction')
    maxPool = torch.nn.MaxPool3d(5,1,2).cuda()


    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(kp_dir):
        os.makedirs(kp_dir)


    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)


    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)

    loss_list = []
    '''if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)'''

    generator.eval()
    kp_detector.eval()
    graph_list = []

    for index_patient, series_img in enumerate(dataset):

        dic_patient = {}

        vertex_list = []
        edge_list = []

        num_lists = np.arange(0,5)
        num_shuffled = []
        for num in range(1):
            np.random.shuffle(num_lists)
            num_shuffled.append(num_lists.copy())

        if config['dataset_params']['object'] == 'cardiac':
            series_img_list = []
            for k in range(5):
                series_img_list.append(series_img+'/image_data/t'+str(k+1)+'.nii')

            patient_name = series_img.split('/')[-1]
            
            trainning_list = Temporal_cardiac_test(\
                series_img_list[:5], job='seg', rotate=rotate_tmp, norm=norm_tmp)

        else:
            trainning_list = Temporal_lung_all_graph(\
                series_img, job='seg')

        train_dataloader = DataLoader(trainning_list, batch_size=1, shuffle=False, num_workers=6, drop_last=True)

        for it, x in tqdm(enumerate(train_dataloader)):

            with torch.no_grad():
                predictions = []
                visualizations = []
                if torch.cuda.is_available():
                    x['driving'] = x['driving'].cuda()
                

                kp_source = kp_detector(x['driving'].to(device1))

                img_dir = kp_dir + str(index_patient) + '_' + str(it) + '_'
                save_kp_img(img=x['driving'], img_dir=img_dir, kp_map=kp_source['heatmap'])

                #kp_source = kp_detector(x['driving'])
                vertex_list.append(kp_source['value'][0,:,:].cpu().data.numpy().tolist())

                ed_dis, edge_connect = distance(kp_source['value'][0,:,:])

                edge_list.append(edge_connect)

                patient_name_dir = graph_dir + str(index_patient) + '_' + str(it)
                create_graph(patient_name_dir, kp_source['value'].cpu().data.numpy()[0,:,:], edge_connect)
                print('done')

        dic_patient['vertext'] = vertex_list
        dic_patient['edge'] = edge_connect



        graph_pkl_dir = pkl_dir + str(index_patient) + '.pickle'
        with open(graph_pkl_dir, 'wb') as outfile:
            pickle.dump(dic_patient, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            #outfile.write('\n')

        with open(graph_pkl_dir, 'rb') as outfile:
            data = pickle.load(outfile)

        graph_list.append(graph_pkl_dir)


        print(data)

    all_vertex = []
    all_vertex_list = []

    for kk in range(len(graph_list)):
        with open(graph_list[kk], 'rb') as outfile:
            dict_data = pickle.load(outfile)

        vertex = np.array(dict_data['vertext'])
        edge = np.array(dict_data['edge'])

        all_vertex.append(torch.from_numpy(vertex[0,:,:]))
        all_vertex_list.append(vertex)

    all_edge = distance_group(all_vertex)

    for i in range(len(graph_list)):
        dic_p = {}
        temp_vertex = []
        for j in range(vertex.shape[0]):
            patient_name_dir = graph_dir + str(i) + '_' + str(j)
            create_graph(patient_name_dir, all_vertex_list[i][j,:,:], all_edge)
            temp_vertex.append(all_vertex_list[i][j,:,:].tolist())

        dic_p['vertext'] = temp_vertex
        dic_p['edge'] = all_edge

        graph_pkl_dir = pkl_dir + str(i) + '.pickle'
        with open(graph_pkl_dir, 'wb') as outfile:
            pickle.dump(dic_p, outfile, protocol=pickle.HIGHEST_PROTOCOL)



    df = pd.DataFrame({'graph':graph_list})
    df.to_csv(log_dir_old + "/graph.csv", index=False)











