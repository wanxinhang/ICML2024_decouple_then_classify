import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
class Encoder(nn.Module):
    def __init__(self,input_dim,hid_dimen1):
        super(Encoder,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(input_dim,hid_dimen1),
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        return encode
class Decoder(nn.Module):
    def __init__(self,input_dim,hid_dimen1):
        super(Decoder,self).__init__()
        self.decoder  =  nn.Sequential(
            nn.Linear(input_dim,hid_dimen1),
            nn.Tanh()
        )

    def forward(self, x):
        decode = self.decoder(x)
        return decode

class MLP(nn.Module):
    def __init__(self,input_dim,hid_dimen1,hid_dimen2,hid_dimen3):
        super(MLP,self).__init__()
        self.Mlp =  nn.Sequential(
        nn.Linear(input_dim,hid_dimen1),
        nn.Tanh(),
        nn.Linear(hid_dimen1, hid_dimen2),
        nn.Tanh(),
        nn.Linear(hid_dimen2, hid_dimen3),
        nn.Tanh(),
        )

    def forward(self, x):
        mlp = self.Mlp(x)
        return mlp

class ASMVL(nn.Module):
    def __init__(self,  num_classes,  num_views,dimension,hid_d, device):
        super(ASMVL, self).__init__()
        self.device = device
        self.dimension = dimension
        self.num_views = num_views
        self.num_classes=num_classes
        self.mv_module = nn.ModuleList()
        self.hid_dimen1=hid_d[0]
        self.hid_dimen2= hid_d[1]
        total_dimen = 0
        for i in range(self.num_views):      #specific feature encoder for each view

            self.mv_module.append(Encoder(dimension[i],self.hid_dimen1))
            total_dimen=total_dimen+dimension[i]

        self.mv_module.append(Encoder(total_dimen, self.hid_dimen1))#shared feature encoder for each view

        self.mv_module.append(Encoder(self.hid_dimen1, self.hid_dimen2))#shared layer encoder

        self.mv_module.append(Decoder(self.hid_dimen2,self.hid_dimen1))#shared layer decoder

        for i in range(self.num_views):
            self.mv_module.append(Decoder(self.hid_dimen1,dimension[i]))


        self.view_class_dimension1= hid_d[2]
        self.view_class_dimension2 = hid_d[3]
        self.classifier_view=MLP(self.hid_dimen2,self.view_class_dimension1,self.view_class_dimension2,self.num_views)
        self.label_class_dimension1 = hid_d[4]
        self.label_class_dimension2 = hid_d[5]
        self.share_classifier_label=MLP(self.hid_dimen2,self.label_class_dimension1,self.label_class_dimension2,self.num_classes)
        self.specific_classifier_label=MLP(self.hid_dimen2,self.label_class_dimension1,self.label_class_dimension2,self.num_classes)
        # print(hidden_dim)
        # exit()

    def forward(self, fea):
        #encoder beign
        specific_fea_en_lay1 = []
        specific_fea_en_lay2 = []
        fea_con = fea[0]
        # print(fea_con.shape)
        time1=time.time()
        for i in range(self.num_views):
            tmp = self.mv_module[i](fea[i])
            specific_fea_en_lay1.append(tmp)
            if i == 0:
                continue
            fea_con = torch.cat((fea_con, fea[i]), 1)
        share_fea_en_lay1=self.mv_module[self.num_views](fea_con)
        time2 = time.time()
        # print(time2 - time1)
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views+1](specific_fea_en_lay1[i])
            specific_fea_en_lay2.append(tmp)
        share_fea_en_lay2 = self.mv_module[self.num_views+1](share_fea_en_lay1)
        #encoder end
        #decoder begin
        specific_fea_de_lay1 = []
        specific_fea_de_lay2 = []
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views+ 2](specific_fea_en_lay2[i]+ share_fea_en_lay2)
            specific_fea_de_lay1.append(tmp)
        for i in range(self.num_views):
            tmp = self.mv_module[self.num_views + 3+i](specific_fea_de_lay1[i])
            specific_fea_de_lay2.append(tmp)
        #decoder end
        #view classfier begin
        view_class_specific_res=[]
        for i in range(self.num_views):
            tmp = self.classifier_view(specific_fea_en_lay2[i])
            view_class_specific_res.append(tmp)
        view_class_share_res = self.classifier_view(share_fea_en_lay2)
        #label classfier begin
        specific_con=specific_fea_en_lay2[0]
        for i in range(self.num_views):
            if i == 0:
                continue
            specific_con =  specific_con+specific_fea_en_lay2[i]
        label_class_specific_res = self.specific_classifier_label(specific_con)
        label_class_share_res=self.share_classifier_label(share_fea_en_lay2)
        return specific_fea_de_lay2,view_class_specific_res, view_class_share_res, label_class_specific_res, label_class_share_res,specific_con,share_fea_en_lay2