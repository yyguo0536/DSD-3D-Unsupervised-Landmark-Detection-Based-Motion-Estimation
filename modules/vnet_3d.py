import torch
import torch.nn as nn
import torch.nn.functional as F
from .pyr_lap import dec_lap_pyr, syn_lap_pyr

class NetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad = 1):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(NetConvBlock, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv3d( \
                in_channels, out_channels, kernel_size=kernel_sz, padding=pad))
        self.bns.append(nn.BatchNorm3d(out_channels))
        #self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=kernel_sz, padding=pad))
            self.bns.append(nn.BatchNorm3d(out_channels))
            #self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            #out = self.afs[i](out)
        return out

class NetInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers=1, k_size=3, pad_size=1):
        super(NetInBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.convb = NetConvBlock(in_channels, out_channels, layers=layers, kernel_sz=k_size, pad=pad_size)

    def forward(self, x):
        out = self.bn(x)
        out = self.convb(x)
        #out = torch.add(out, x)
        return out

class NetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetDownBlock, self).__init__()
        self.down = nn.Conv3d( \
                in_channels, out_channels, kernel_size=2, stride=2)
        #self.af= nn.PReLU(out_channels)
        self.bn = nn.BatchNorm3d(out_channels)
        self.convb = NetConvBlock(out_channels, out_channels, layers=layers)

    def forward(self, x):
        down = self.down(x)
        down = self.bn(down)
        #down = self.af(down)
        out = self.convb(down)
        out = torch.add(out, down)
        return out

class NetUpBlock(nn.Module):
    def __init__(self, in_channels, br_channels, out_channels, layers):
        super(NetUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        #self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn(up)
        #up = self.af(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        return out

class NetJustUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers):
        super(NetJustUpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        #self.af= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels, out_channels, layers=layers)

    def forward(self, x):
        up = self.up(x)
        up = self.bn(up)
        #up = self.af(up)
        out = self.convb(up)
        #out = torch.add(out, up)
        return out


class NetOutBlock(nn.Module):
    def __init__(self, \
            in_channels, br_channels, out_channels, classes, layers=1):
        super(NetOutBlock, self).__init__()
        self.up = nn.ConvTranspose2d(\
                in_channels, out_channels, kernel_size=2, stride=2)
        self.bn_up = nn.BatchNorm2d(out_channels)
        #self.af_up= nn.PReLU(out_channels)
        self.convb = NetConvBlock( \
                out_channels+br_channels, out_channels, layers=layers)
        self.conv = nn.Conv2d(out_channels, classes, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(classes)
        #self.af_out= nn.PReLU(classes)

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.bn_up(up)
        #up = self.af_up(up)
        out = torch.cat([up, bridge], 1)
        out = self.convb(out)
        out = torch.add(out, up)
        out = self.conv(out)
        out = self.bn_out(out)
        #out = self.af_out(out)
        return out




class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class down_conv(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2, kernel_sz=3, pad = 1):
        '''
        ConvBlock = consistent convs
        for each conv, conv(5x5) -> BN -> activation(PReLU)
        params:
        in/out channels: output/input channels
        layers: number of convolution layers
        '''
        super(down_conv, self).__init__()
        self.layers = layers
        self.afs = torch.nn.ModuleList() # activation functions
        self.convs = torch.nn.ModuleList() # convolutions
        self.bns = torch.nn.ModuleList()
        # first conv
        self.convs.append(nn.Conv3d( \
                in_channels, out_channels, kernel_size=kernel_sz, stride = (1,2,2), padding=pad))
        self.bns.append(nn.BatchNorm3d(out_channels))
        #self.afs.append(nn.PReLU(out_channels))
        #self.afs.append(nn.ELU())
        for i in range(self.layers-1):
            self.convs.append(nn.Conv3d( \
                    out_channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm3d(out_channels))
            #self.afs.append(nn.PReLU(out_channels))

    def forward(self, x):
        out = x
        for i in range(self.layers):
            out = self.convs[i](out)
            out = self.bns[i](out)
            #out = self.afs[i](out)
        return out






class Temporal_Encoder(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder, self).__init__()
        self.syn_ll = syn_lap_pyr
        self.dec_ll = dec_lap_pyr
        self.in_block_new = NetConvBlock(in_channels, 30)
        self.down_block1 = NetDownBlock(30, 48, 2)
        self.down_block2 = NetDownBlock(48, 80, 2)
        self.down_block3 = NetDownBlock(80, 160, 2)
        self.down_block4 = NetDownBlock(160, 256, 2)

        

    def forward(self, image):
        #img = self.dec_ll(image, 2)
        #img = self.syn_ll(img)
        fm0 = self.in_block_new(image)
        fm1 = self.down_block1(fm0)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)


        return [fm, fm3, fm2, fm1, fm0]



class Temporal_Decoder(nn.Module):
    def __init__(self, out_chal = 3):
        classes = out_chal
        super(Temporal_Decoder, self).__init__()
        self.up_block2 = NetUpBlock(256, 160, 180, 2)
        self.up_block3 = NetUpBlock(180, 80, 100, 2)
        self.up_block4 = NetUpBlock(100, 48, 64, 2)
        self.up_block5 = NetJustUpBlock(64, classes, 2)
        

    def forward(self, feature_list):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)

        up5 = self.up_block2(feature_list[0], feature_list[1])
        #up4 = self.up_block2(up5, feature_list[2])
        up3 = self.up_block3(up5, feature_list[2])
        up2 = self.up_block4(up3, feature_list[3])#, feature_list[3])
        up1 = self.up_block5(up2)

        return up1



class Temporal_Encoder_G(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_G, self).__init__()

        self.down_block2 = NetDownBlock(in_channels, 32, 2)
        self.down_block3 = NetDownBlock(32, 48, 2)
        self.down_block4 = NetDownBlock(48, 96, 2)
        self.down_block5 = NetDownBlock(96, 160, 1)

        

    def forward(self, image):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)
        #fm = self.in_block(image)
        #fm1 = self.down_block1(fm)
        fm2 = self.down_block2(image)
        fm3 = self.down_block3(fm2)
        fm4 = self.down_block4(fm3)
        fm = self.down_block5(fm4)


        return [fm, fm4, fm3, fm2]




class Temporal_Decoder_G(nn.Module):
    def __init__(self, out_chal = 1):
        classes = out_chal
        super(Temporal_Decoder_G, self).__init__()
        self.up_block1 = NetUpBlock(160, 96, 64, 2)
        self.up_block2 = NetUpBlock(64, 48, 32, 2)
        self.up_block3 = NetUpBlock(32, 32, 16, 2)
        self.up_block4 = NetJustUpBlock(16, out_chal, 2)

        

    def forward(self, features):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)

        up = self.up_block1(features[0], features[1])
        up = self.up_block2(up, features[2])
        up = self.up_block3(up, features[3])
        up = self.up_block4(up)

        return up


class Temporal_VGG(nn.Module):
    def __init__(self, classes_num = 1):
        classes = classes_num
        super(Temporal_VGG, self).__init__()
        self.in_block = down_conv(1, 12, 2, (3,5,5), pad = (1,2,2))#96*128*128
        self.down_block1 = NetDownBlock(12, 24, 2)#48*64*64
        self.down_block2 = NetDownBlock(24, 48, 2)#24*32*32
        self.down_block3 = NetDownBlock(48, 96, 2)#12*16*16
        self.down_block4 = NetDownBlock(96, 192, 2)#6*8*8
        

        

    def forward(self, image):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)
        fm = self.in_block(image)
        fm1 = self.down_block1(fm)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)
        #fm = self.down_block5(fm4)

        return fm.view(1,-1)



class Temporal_Encoder_D(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_D, self).__init__()
        self.syn_ll = syn_lap_pyr
        self.dec_ll = dec_lap_pyr
        self.in_block_new = NetConvBlock(in_channels, in_channels)#
        self.down_block1 = NetDownBlock(in_channels, 48, 2)#
        self.down_block2 = NetDownBlock(48, 96, 2)#
        self.down_block3 = NetDownBlock(96, 160, 2)#
        self.down_block4 = NetDownBlock(160, 256, 2)#

        

    def forward(self, image):
        #img = self.dec_ll(image, 2)
        #img = self.syn_ll(img)
        fm0 = self.in_block_new(image)
        fm1 = self.down_block1(fm0)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)
        #fm = self.down_block5(fm4)


        return [fm, fm3, fm2, fm1, fm0]



class Temporal_Decoder_D(nn.Module):
    def __init__(self, out_chal = 3):
        classes = out_chal
        super(Temporal_Decoder_D, self).__init__()
        self.up_block2 = NetUpBlock(256, 160, 160, 2)
        self.up_block3 = NetUpBlock(160, 96, 128, 2)
        self.up_block4 = NetUpBlock(128, 48, 64, 2)
        self.up_block5 = NetJustUpBlock(64, classes, 2)


        

    def forward(self, feature_list):
        up5 = self.up_block2(feature_list[0], feature_list[1])
        up4 = self.up_block3(up5, feature_list[2])
        up2 = self.up_block4(up4, feature_list[3])#, feature_list[3])
        up1 = self.up_block5(up2)

        return up1





class Temporal_Encoder_ND(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_ND, self).__init__()
        self.syn_ll = syn_lap_pyr
        self.dec_ll = dec_lap_pyr
        self.in_block_new1 = NetConvBlock(in_channels, 48)
        self.down_block1 = NetDownBlock(48, 48, 2)
        self.down_block2 = NetDownBlock(48, 80, 2)
        self.down_block3 = NetDownBlock(80, 160, 2)
        self.down_block4 = NetDownBlock(160, 256, 2)

        

    def forward(self, field):
        #img = self.dec_ll(image, 2)
        #img = self.syn_ll(img)
        fm0 = self.in_block_new1(field)

        fm1 = self.down_block1(fm0)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)
        #fm = self.down_block5(fm4)


        return [fm, fm3, fm2, fm1, fm0]



class Temporal_Decoder_ND(nn.Module):
    def __init__(self, out_chal = 3):
        classes = out_chal
        super(Temporal_Decoder_ND, self).__init__()
        self.up_block2 = NetUpBlock(256, 160, 180, 2)
        self.up_block3 = NetUpBlock(180, 80, 100, 2)
        self.up_block4 = NetUpBlock(100, 48, 64, 2)
        self.up_block5 = NetUpBlock(64, 48, classes, 2)
        

    def forward(self, feature_list):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)

        up5 = self.up_block2(feature_list[0], feature_list[1])
        #up4 = self.up_block2(up5, feature_list[2])
        up3 = self.up_block3(up5, feature_list[2])
        up2 = self.up_block4(up3, feature_list[3])
        up1 = self.up_block5(up2, feature_list[4])

        return up1




class Temporal_Encoder_Pre(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_Pre, self).__init__()
        self.in_block = NetConvBlock(in_channels, 30)
        self.down_block1 = NetDownBlock(30, 48, 2)
        self.down_block2 = NetDownBlock(48, 80, 2)
        self.down_block3 = NetDownBlock(80, 160, 2)
        self.down_block4 = NetDownBlock(160, 256, 2)
        
        self.fc_layer1 = nn.Linear(32000, 4096)
        self.reLu1 = nn.ReLU()
        self.dropLayer1 = nn.Dropout(p=0.1)
        self.fc_layer2 = nn.Linear(4096,512)
        self.reLu2 = nn.PReLU()
        self.dropLayer2 = nn.Dropout(p=0.1)

        #self.fc_layer2 = nn.Linear(4096,1024)
        

        self.BN1d_layer1 = nn.BatchNorm1d(2)
        self.conv1d_layer_concat1 = nn.Conv1d(2, 1, 5, stride=1)
        self.PR1d_layer1 = nn.PReLU()

        self.BN1d_layer2 = nn.BatchNorm1d(4)
        self.conv1d_layer_concat2 = nn.Conv1d(4, 1, 5, stride=1)
        self.PR1d_layer2 = nn.PReLU()

        self.fc_layer3 = nn.Linear(504,5)
        self.reLu3 = nn.PReLU()

        

    def forward_branch(self, image):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)
        #image = F.interpolate(image, size=(80,80,80), mode='trilinear')
        fm = self.in_block(image)
        fm1 = self.down_block1(fm)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)
        #fm = self.down_block5(fm4)

        #fm = self.down_block6(fm)

        fc = fm.view(1, -1)
        fc = self.fc_layer1(fc)
        fc = self.reLu1(fc)
        fc = self.dropLayer1(fc)
        fc = self.fc_layer2(fc)
        fc = self.reLu2(fc)
        fc = self.dropLayer2(fc)
        #fc = self.fc_layer3(fc)
        #fc = self.reLu3(fc)

        return fc.view(1,1,512), [fm, fm3, fm2, fm1]


    def forward_branch2(self, img):
        img_out = self.BN1d_layer1(img)
        img_out = self.conv1d_layer_concat1(img_out)
        img_out = self.PR1d_layer1(img_out)

        return img_out

    def forward_branch3(self, img):
        img_out = self.BN1d_layer2(img)
        img_out = self.conv1d_layer_concat2(img_out)
        img_out = self.PR1d_layer2(img_out)

        fc_out = img_out.view(1,-1)
        fc_out = self.fc_layer3(fc_out)
        fc_out = self.reLu3(fc_out)

        return fc_out.view(1,5,1)


    def forward_concat(self, f_list):
        all_list = []
        for i in range(len(f_list)):
            num_list = [0,1,2,3,4]
            num_list.remove(i)
            tmp_cat = [torch.cat((f_list[i], f_list[j]), 1) for j in num_list]
            tmp_series = [self.forward_branch2(tmp_cat[k]) for k in range(len(tmp_cat))]
            img_cat = torch.cat([li for li in tmp_series],1)
            all_list.append(self.forward_branch3(img_cat))

        return all_list


    def forward(self, img_list):

        img_series = [self.forward_branch(torch.unsqueeze(img_list[:,i,:,:,:], 1)) for i in range(img_list.shape[1])]

        all_list = []
        for i in range(len(img_series)):
            num_list = [0,1,2,3,4]
            num_list.remove(i)
            tmp_cat = [torch.cat((img_series[i][0], img_series[j][0]), 1) for j in num_list]
            tmp_series = [self.forward_branch2(tmp_cat[k]) for k in range(len(tmp_cat))]
            img_cat = torch.cat([li for li in tmp_series],1)
            all_list.append(self.forward_branch3(img_cat))

        img_cat = torch.cat([li for li in all_list],2)

        #img_cat = torch.unsqueeze(img_cat, 0)

        #fc_cat = self.conv1d_layer1(img_cat)
        #fc_cat = self.BN1d_layer1(fc_cat)
        #fc_cat = self.PR1d_layer1(fc_cat)


        return img_cat, img_series#F.log_softmax(fc_cat, dim=1)



class Temporal_Encoder_Pre_New(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_Pre_New, self).__init__()
        self.in_block = NetConvBlock(in_channels, 30)
        self.down_block1 = NetDownBlock(30, 48, 2)
        self.down_block2 = NetDownBlock(48, 80, 2)
        self.down_block3 = NetDownBlock(80, 160, 2)
        self.down_block4 = NetDownBlock(160, 256, 2)
        self.down_block5 = NetDownBlock(256, 384, 2)
        self.down_block6 = nn.AdaptiveMaxPool3d((1,1,1))
        
        self.fc_layer1 = nn.Linear(384, 384)
        self.reLu1 = nn.PReLU()
        #self.fc_layer2 = nn.Linear(4096,1024)
        

        self.BN1d_layer1 = nn.BatchNorm1d(2)
        self.conv1d_layer_concat1 = nn.Conv1d(2, 1, 5, stride=1)
        self.PR1d_layer1 = nn.PReLU()

        self.BN1d_layer2 = nn.BatchNorm1d(4)
        self.conv1d_layer_concat2 = nn.Conv1d(4, 1, 5, stride=1)
        self.PR1d_layer2 = nn.PReLU()

        self.fc_layer3 = nn.Linear(376,5)
        self.reLu3 = nn.PReLU()

        

    def forward_branch(self, image):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)
        #image = F.interpolate(image, size=(80,80,80), mode='trilinear')
        fm = self.in_block(image)
        fm1 = self.down_block1(fm)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm4 = self.down_block4(fm3)
        fm5 = self.down_block5(fm4)

        fm = self.down_block6(fm5)
        print(fm.shape)

        fc = fm.view(1, -1)
        fc = self.fc_layer1(fc)
        fc = self.reLu1(fc)

        return fc.view(1,1,384), [fm, fm3, fm2, fm1]


    def forward_branch2(self, img):
        img_out = self.BN1d_layer1(img)
        img_out = self.conv1d_layer_concat1(img_out)
        img_out = self.PR1d_layer1(img_out)

        return img_out

    def forward_branch3(self, img):
        img_out = self.BN1d_layer2(img)
        img_out = self.conv1d_layer_concat2(img_out)
        img_out = self.PR1d_layer2(img_out)

        fc_out = img_out.view(1,-1)
        fc_out = self.fc_layer3(fc_out)
        fc_out = self.reLu3(fc_out)

        return fc_out.view(1,5,1)


    def forward_concat(self, f_list):
        all_list = []
        for i in range(len(f_list)):
            num_list = [0,1,2,3,4]
            num_list.remove(i)
            tmp_cat = [torch.cat((f_list[i], f_list[j]), 1) for j in num_list]
            tmp_series = [self.forward_branch2(tmp_cat[k]) for k in range(len(tmp_cat))]
            img_cat = torch.cat([li for li in tmp_series],1)
            all_list.append(self.forward_branch3(img_cat))

        return all_list


    def forward(self, img_list):

        img_series = [self.forward_branch(torch.unsqueeze(img_list[:,i,:,:,:], 1)) for i in range(img_list.shape[1])]

        all_list = []
        for i in range(len(img_series)):
            num_list = [0,1,2,3,4]
            num_list.remove(i)
            tmp_cat = [torch.cat((img_series[i][0], img_series[j][0]), 1) for j in num_list]
            tmp_series = [self.forward_branch2(tmp_cat[k]) for k in range(len(tmp_cat))]
            img_cat = torch.cat([li for li in tmp_series],1)
            all_list.append(self.forward_branch3(img_cat))

        img_cat = torch.cat([li for li in all_list],2)



        return img_cat, img_series#F.log_softmax(fc_cat, dim=1)



class Temporal_Encoder_Pre_Eval(nn.Module):
    def __init__(self, in_channels = 1):
        #classes = classes_num
        super(Temporal_Encoder_Pre_Eval, self).__init__()
        self.in_block = NetConvBlock(in_channels, 30)
        self.down_block1 = NetDownBlock(30, 48, 2)
        self.down_block2 = NetDownBlock(48, 80, 2)
        self.down_block3 = NetDownBlock(80, 160, 2)
        self.down_block4 = NetDownBlock(160, 256, 2)
        
        self.fc_layer1 = nn.Linear(32000, 4096)
        self.reLu1 = nn.ReLU()
        self.dropLayer1 = nn.Dropout(p=0.1)
        self.fc_layer2 = nn.Linear(4096,512)
        #self.fc_layer3 = nn.Linear(1024,10)
        self.reLu2 = nn.PReLU()
        self.dropLayer2 = nn.Dropout(p=0.1)

        #self.fc_layer2 = nn.Linear(4096,1024)
        

        self.BN1d_layer1 = nn.BatchNorm1d(2)
        self.conv1d_layer_concat1 = nn.Conv1d(2, 1, 5, stride=1)
        self.PR1d_layer1 = nn.PReLU()

        self.BN1d_layer2 = nn.BatchNorm1d(4)
        self.conv1d_layer_concat2 = nn.Conv1d(4, 1, 5, stride=1)
        self.PR1d_layer2 = nn.PReLU()

        self.fc_layer3 = nn.Linear(504,5)
        self.reLu3 = nn.PReLU()

        #self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')#Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        #self.soft_max = nn.LogSoftmax()

        

    def forward_branch(self, image):
        #img = self.dec_ll(image, 4)
        #img = self.syn_ll(img)
        #image = F.interpolate(image, size=(80,80,80), mode='trilinear')
        fm = self.in_block(image)
        fm1 = self.down_block1(fm)
        fm2 = self.down_block2(fm1)
        fm3 = self.down_block3(fm2)
        fm = self.down_block4(fm3)
        print(fm.shape)
        #fm = self.down_block5(fm4)

        #fm = self.down_block6(fm)

        fc = fm.view(1, -1)
        fc = self.fc_layer1(fc)
        fc = self.reLu1(fc)
        fc = self.dropLayer1(fc)
        fc = self.fc_layer2(fc)
        fc = self.reLu2(fc)
        fc = self.dropLayer2(fc)
        #fc = self.fc_layer3(fc)
        #fc = self.reLu3(fc)

        return fc, [fm, fm3, fm2, fm1]


    def forward_branch2(self, img):
        img_out = self.BN1d_layer1(img)
        img_out = self.conv1d_layer_concat1(img_out)
        img_out = self.PR1d_layer1(img_out)

        return img_out

    def forward_branch3(self, img):
        img_out = self.BN1d_layer2(img)
        img_out = self.conv1d_layer_concat2(img_out)
        img_out = self.PR1d_layer2(img_out)

        fc_out = img_out.view(1,-1)
        fc_out = self.fc_layer3(fc_out)
        fc_out = self.reLu3(fc_out)

        return fc_out.view(1,5,1)


    def forward_concat(self, f_list):
        all_list = []
        for i in range(len(f_list)):
            num_list = [0,1,2,3,4]
            num_list.remove(i)
            tmp_cat = [torch.cat((f_list[i], f_list[j]), 1) for j in num_list]
            tmp_series = [self.forward_branch2(tmp_cat[k]) for k in range(len(tmp_cat))]
            img_cat = torch.cat([li for li in tmp_series],1)
            all_list.append(self.forward_branch3(img_cat))

        return all_list


    def forward(self, img_list):

        #img_series = [self.forward_branch(torch.unsqueeze(img_list[:,i,:,:,:], 1)) for i in range(img_list.shape[1])]

        img_series = self.forward_branch(img_list)

        #img_cat = torch.unsqueeze(img_cat, 0)

        #fc_cat = self.conv1d_layer1(img_cat)
        #fc_cat = self.BN1d_layer1(fc_cat)
        #fc_cat = self.PR1d_layer1(fc_cat)


        return img_series#F.log_softmax(fc_cat, dim=1)