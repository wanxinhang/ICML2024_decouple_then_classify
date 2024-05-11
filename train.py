import time

import numpy as np
import torch
from util import get_data
from model import  ASMVL
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
import heapq
import math
def train(args, device):
    file = open('D:\wxhwork/active_smvl_idea/my_icml/res/' + args.dataset + '_doc.txt', 'w')
    repeat_num=args.n_repeated
    file.truncate(0)
    fea,labels,num_view,dimension=get_data(args.dataset, device)
    num_classes = len(np.unique(labels))
    # sample_num=
    sample_num=labels.shape[0]
    labels = labels.to(device)
    hid_d=[args.d1,args.d2,args.d3,args.d4,args.d5,args.d6]
    real_label_ratio=args.label_ratio
    if round(sample_num*real_label_ratio)<num_classes:
        real_label_ratio=real_label_ratio+args.select_each_ratio
    if round(sample_num*real_label_ratio)<num_classes:
        real_label_ratio=real_label_ratio+args.select_each_ratio
    if round(sample_num*real_label_ratio)<num_classes:
        real_label_ratio=real_label_ratio+args.select_each_ratio
    if round(sample_num*real_label_ratio)<num_classes:
        real_label_ratio=real_label_ratio+args.select_each_ratio
    sss = StratifiedShuffleSplit(n_splits=repeat_num, test_size=real_label_ratio,
                                 random_state=1)
    acc_total=[]
    each_fea = []
    for v in range(num_view):
        each_fea.append(fea[v])
    doc1_acc = np.zeros((25, repeat_num))
    doc2_acc = np.zeros((25, repeat_num))
    doc3_acc = np.zeros((25, repeat_num))
    avg1_acc = np.zeros((25,))
    avg2_acc = np.zeros((25,))
    avg3_acc = np.zeros((25,))
    std1_acc = np.zeros((25,))
    std2_acc = np.zeros((25,))
    std3_acc = np.zeros((25,))
    com_acc1=np.zeros((5,repeat_num))
    com_acc2 = np.zeros((5, repeat_num))
    com_acc3 = np.zeros((5, repeat_num))
    com_avg1=np.zeros((5,))
    com_avg2 = np.zeros((5,))
    com_avg3 = np.zeros((5, ))
    com_std1 = np.zeros((5, ))
    com_std2 = np.zeros((5, ))
    com_std3 = np.zeros((5, ))

    iter=-1
    for unlabel_index, label_index in sss.split(fea[0], labels.cpu().numpy()):
        iter=iter+1
        real_unlabel_index = unlabel_index
        this_ratio = real_label_ratio
        train_label = labels.cpu().numpy()
        train_label = torch.tensor(train_label)
        train_label = train_label.to(device)
        this_ratio=real_label_ratio
        model = ASMVL(num_classes, num_view, dimension, hid_d, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        epoch_max=args.num_epoch
        top_ratio = args.top_ratio
        for cnttt in range(200):
            if this_ratio>0.51:
                cnttt=cnttt-1
                break
            loss_doc=[]
            with tqdm(total=epoch_max, desc="Training") as pbar:
                for epoch in range(epoch_max):
                    # print(epoch)
                    loss_total=0
                    model.train()
                    optimizer.zero_grad()
                    time1=time.time()
                    specific_fea_de_lay2,view_class_specific_res, view_class_share_res, label_class_specific_res, label_class_share_res,_,_ = model(each_fea)
                    #encoder_decoder_loss
                    loss1 = nn.MSELoss()
                    each_fea1 = torch.cat(each_fea, dim=1)
                    specific_fea_de_lay2 = torch.cat(specific_fea_de_lay2, dim=1)
                    ed_loss = loss1(each_fea1, specific_fea_de_lay2)
                    del each_fea1,specific_fea_de_lay2
                    #classfy_view_loss
                    view_loss=0
                    for v in range(num_view):
                        tmp_label= torch.zeros((sample_num,num_view), dtype=torch.float, device='cuda:0')
                        tmp_label[:,v]=1.0
                        view_loss=view_loss+F.kl_div(view_class_specific_res[v].softmax(dim=-1).log(), tmp_label.softmax(dim=-1), reduction='sum')
                    tmp_label=torch.ones((sample_num,num_view), dtype=torch.float, device='cuda:0')/num_view
                    view_loss=view_loss+F.kl_div(view_class_share_res.softmax(dim=-1).log(), tmp_label.softmax(dim=-1), reduction='sum')
                    #label_loss
                    label_loss=0
                    # class_res1 = torch.max(label_class_specific_res[label_index].softmax(dim=-1), label_class_share_res[label_index].softmax(dim=-1))
                    real_label = torch.zeros((label_index.shape[0],num_classes), dtype=torch.float, device='cuda:0')
                    for num in range(label_index.shape[0]):
                        real_label[num,train_label[label_index[num]]] = 1

                    label_loss = F.kl_div(label_class_specific_res[label_index].softmax(dim=-1).log(), real_label, reduction='sum') + F.kl_div( label_class_share_res[label_index].softmax(dim=-1).log(), real_label, reduction='sum')
                    loss_total=ed_loss+args.lambda1*view_loss+args.lambda2*label_loss
                    # loss_total=loss_total+loss
                    # print(loss_total)
                    # loss_tatal
                    loss_total.backward()
                    optimizer.step()
                    loss_doc.append(loss_total.item())
                    with torch.no_grad():
                        model.eval()
                        cnt=0
                        acc=0
                        acc1=0
                        acc2=0
                        # print(type(unlabel_index))
                        each_fea1 = []
                        for v in range(num_view):
                            each_fea1.append(fea[v][real_unlabel_index])
                            # each_fea1.append(fea[v])
                        _,_, _, label_class_specific_res, label_class_share_res,_,_ = model(each_fea1)
                        class_res=torch.max(label_class_specific_res.softmax(dim=-1), label_class_share_res.softmax(dim=-1))
                        # # print(class_res.shape)
                        which=torch.argmax(class_res)

                        for num in range(real_unlabel_index.shape[0]):
                            which = torch.argmax(class_res[num])
                            # print(which)
                            which1=torch.argmax(label_class_specific_res[num])
                            which2=torch.argmax(label_class_share_res[num])
                        # which1=
                            if which==labels[real_unlabel_index[num]]:
                                acc=acc+1
                            if which1 == labels[real_unlabel_index[num]]:
                                acc1 = acc1 + 1
                            if which2 == labels[real_unlabel_index[num]]:
                                acc2 = acc2 + 1
                        # # acc_total.append(acc/cnt)
                        print(1-unlabel_index.shape[0]/sample_num,1-real_unlabel_index.shape[0]/sample_num,acc,acc1,acc2,real_unlabel_index.shape[0])
                        tmppp_app=[]
                        tmppp_app.append(this_ratio)
                        tmppp_app.append(acc/real_unlabel_index.shape[0])
                        tmppp_app.append(acc1/real_unlabel_index.shape[0])
                        tmppp_app.append(acc2/real_unlabel_index.shape[0])
                        file.write('test:')
                        file.write(str(tmppp_app))
                        file.write('\n')
                        pbar.update(1)
                    #
                    # del output, hidden_list, w_list
                    # torch.cuda.empty_cache()
            file.write('loss:')
            file.write(str(loss_doc))
            file.write('\n')
            # print(train_label)
            # print(labels)
            with torch.no_grad():
                model.eval()
                acc = 0
                acc1 = 0
                acc2 = 0
                # print(type(unlabel_index))
                each_fea1 = []
                for v in range(num_view):
                    each_fea1.append(fea[v][real_unlabel_index])
                    # each_fea1.append(fea[v])
                _, _, _, label_class_specific_res, label_class_share_res, _, _ = model(each_fea1)
                class_res = torch.max(label_class_specific_res.softmax(dim=-1), label_class_share_res.softmax(dim=-1))
                # # print(class_res.shape)
                which = torch.argmax(class_res)

                for num in range(real_unlabel_index.shape[0]):
                    which = torch.argmax(class_res[num])
                    # print(which)
                    which1 = torch.argmax(label_class_specific_res[num])
                    which2 = torch.argmax(label_class_share_res[num])
                    # which1=
                    if which == labels[real_unlabel_index[num]]:
                        acc = acc + 1
                    if which1 == labels[real_unlabel_index[num]]:
                        acc1 = acc1 + 1
                    if which2 == labels[real_unlabel_index[num]]:
                        acc2 = acc2 + 1
                doc1_acc[cnttt,iter]=acc/real_unlabel_index.shape[0]
                doc2_acc[cnttt,iter]= acc1 / real_unlabel_index.shape[0]
                doc3_acc[cnttt,iter] = acc2/ real_unlabel_index.shape[0]
                if  this_ratio>=0.085 and this_ratio<=0.115:
                    # print(iter)
                    com_acc1[0,iter]=acc/real_unlabel_index.shape[0]
                    com_acc2[0,iter] = acc1 / real_unlabel_index.shape[0]
                    com_acc3[0,iter] = acc2 / real_unlabel_index.shape[0]
                if this_ratio >= 0.185 and this_ratio <= 0.215:
                    com_acc1[1, iter] = acc / real_unlabel_index.shape[0]
                    com_acc2[1, iter] = acc1 / real_unlabel_index.shape[0]
                    com_acc3[1, iter] = acc2 / real_unlabel_index.shape[0]
                if  this_ratio>=0.285 and this_ratio<=0.315:
                    com_acc1[2,iter]=acc/real_unlabel_index.shape[0]
                    com_acc2[2,iter] = acc1 / real_unlabel_index.shape[0]
                    com_acc3[2,iter] = acc2 / real_unlabel_index.shape[0]
                if  this_ratio>=0.385 and this_ratio<=0.415:
                    com_acc1[3,iter]=acc/real_unlabel_index.shape[0]
                    com_acc2[3,iter] = acc1 / real_unlabel_index.shape[0]
                    com_acc3[3,iter] = acc2 / real_unlabel_index.shape[0]
                if  this_ratio>=0.485 and this_ratio<=0.515:
                    com_acc1[4,iter]=acc/real_unlabel_index.shape[0]
                    com_acc2[4,iter] = acc1 / real_unlabel_index.shape[0]
                    com_acc3[4,iter] = acc2 / real_unlabel_index.shape[0]
                add_pseudo=[]
                select_real=[]
                add_real=[]
                class_sample=[]
                for k in range(num_classes):
                    class_sample.append([])
                for num in range(label_index.shape[0]):
                    class_sample[labels[label_index[num]]].append(label_index[num])
                max_value1=[]
                max_value2 = []
                _, _, _, label_class_specific_res, label_class_share_res, specific_con, share_fea = model(each_fea)
                label_class_specific_res=label_class_specific_res[unlabel_index].softmax(dim=-1)
                label_class_share_res=label_class_share_res[unlabel_index].softmax(dim=-1)

                cnt=0
                all=0
                tenporary=[]
                for num in range(unlabel_index.shape[0]):
                    which1 = torch.argmax(label_class_specific_res[num])
                    which2 = torch.argmax(label_class_share_res[num])
                    if  which1==which2:
                        add_pseudo.append(unlabel_index[num])
                        tenporary.append(num)
                        max_value1.append(max(label_class_specific_res[num]))
                        max_value2.append(max(label_class_share_res[num]))
                top_number= min(round(sample_num*top_ratio),len(add_pseudo))
                value1=heapq.nlargest(top_number, max_value1)
                value2=heapq.nlargest(top_number, max_value2)
                # print(value1,value2)
                psudo_add=[]
                pre_l=[]
                # top_ratio=top_ratio+0.01
                top_ratio=min(top_ratio,0.2)
                for num in range(len(add_pseudo)):
                    if max_value1[num] in  value1 and  max_value2[num] in  value2:
                        cnt = cnt + 1
                        psudo_add.append(add_pseudo[num])
                        pre_l.append(torch.argmax(label_class_specific_res[tenporary[num]]))
                        if torch.argmax(label_class_specific_res[tenporary[num]])== labels[add_pseudo[num]]:
                            all = all + 1
                # print(max_label_class_specific_res.shape,label_class_specific_res.shape)

                max_res=[]
                # max_res=np.zeros((unlabel_index.shape[0],))
                for num in range(unlabel_index.shape[0]):
                    max_res.append(max(label_class_specific_res[num])*max(label_class_share_res[num]))
                min_number=min(round(sample_num*args.select_each_ratio),unlabel_index.shape[0])
                min_value = heapq.nsmallest(min_number, max_res)
                select_sample=[]
                if  min_number>=0:
                    for num in range(unlabel_index.shape[0]):
                        if max_res[num] in  min_value:
                            select_sample.append(unlabel_index[num])
                    for num in range(len(select_sample)):
                        unlabel_index = np.delete(unlabel_index, np.where((unlabel_index == select_sample[num])))
                        real_unlabel_index = np.delete(real_unlabel_index,
                                                       np.where((real_unlabel_index == select_sample[num])))
                        label_index = np.append(label_index, values=select_sample[num])
                # print(len(select_sample)/sample_num)
                if len(psudo_add)>0  and this_ratio>=0.175:
                    top_ratio = top_ratio + 0.0015
                    print(top_ratio)
                    for num in range(len(psudo_add)):
                        train_label[psudo_add[num]]=pre_l[num]
                        unlabel_index = np.delete(unlabel_index, np.where((unlabel_index == psudo_add[num])))
                        label_index = np.append(label_index, values=psudo_add[num])
                this_ratio=this_ratio+args.select_each_ratio
                tmp_epoch=epoch_max-15
                epoch_max=max(150,tmp_epoch)
    avg1_acc=np.mean(doc1_acc,1)
    std1_acc = np.std(doc1_acc, 1)
    avg2_acc = np.mean(doc2_acc, 1)
    std2_acc = np.std(doc2_acc, 1)
    avg3_acc = np.mean(doc3_acc, 1)
    std3_acc = np.std(doc3_acc, 1)
    com_avg1=np.mean(com_acc1, 1)
    com_std1 = np.std(com_acc1, 1)
    com_avg2 = np.mean(com_acc2, 1)
    com_std2 = np.std(com_acc2, 1)
    com_avg3 = np.mean(com_acc3, 1)
    com_std3 = np.std(com_acc3, 1)
    file.close()
    file1 = open('D:\wxhwork/active_smvl_idea/my_icml/res/' + args.dataset + '.txt', 'w')
    file1.truncate(0)
    file1.write(str(doc1_acc))
    file1.write('\n\n')
    file1.write(str(avg1_acc))
    file1.write('\n\n')
    file1.write(str(std1_acc))
    file1.write('\n\n')
    file1.write(str(com_acc1))
    file1.write('\n\n')
    file1.write(str(com_avg1))
    file1.write('\n\n')
    file1.write(str(com_std1))
    file1.write('\n\n\n')

    file1.write(str(doc2_acc))
    file1.write('\n\n')
    file1.write(str(avg2_acc))
    file1.write('\n\n')
    file1.write(str(std2_acc))
    file1.write('\n\n')
    file1.write(str(com_acc2))
    file1.write('\n\n')
    file1.write(str(com_avg2))
    file1.write('\n\n')
    file1.write(str(com_std2))
    file1.write('\n\n\n')

    file1.write(str(doc3_acc))
    file1.write('\n\n')
    file1.write(str(avg3_acc))
    file1.write('\n\n')
    file1.write(str(std3_acc))
    file1.write('\n\n')
    file1.write(str(com_acc3))
    file1.write('\n\n')
    file1.write(str(com_avg3))
    file1.write('\n\n')
    file1.write(str(com_std3))
    file1.write('\n\n\n')
    file1.close()
    # print(max(acc_total))