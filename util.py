import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import h5py
from sklearn import metrics
import scipy
# data = hdf5storage.loadmat(str)


def get_data(name,device):
    # features = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/features.csv'.format((name)), delimiter=',')
    # targets = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/targets.csv'.format((name))).astype(int)
    # print('Dataset {0}:'.format(name), features.shape, targets.shape)
    data = h5py.File('D:\MultiView_Dataset\{0}'.format((name))+'.mat','r');
    num_view=len(data['X'][0])
    fea=[]
    dimension = []
    for i in range(num_view):
        feature = [data[element[i]][:] for element in data['X']]
        feature = np.array(feature)
        feature=np.squeeze(feature)
        feature=feature.T
        # print(feature.shape)
        feature=normalize(feature)
        if ss.isspmatrix(feature):
            feature = feature.todense()
        feature=torch.from_numpy(feature).float().to(device)
        fea.append(feature)
        dimension.append(feature.shape[1])
        del feature
    Y=np.array(data['Y'])
    Y=Y.T
    Y = Y - min(Y)
    Y = torch.from_numpy(Y).long()
    return fea,Y,num_view,dimension

# def get_data(name,device):
#     # features = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/features.csv'.format((name)), delimiter=',')
#     # targets = np.loadtxt('F:\wxh_work\py_semi\co_gcn_master\data_cogcn/{0}/targets.csv'.format((name))).astype(int)
#     # print('Dataset {0}:'.format(name), features.shape, targets.shape)
#     file_path='D:\MultiView_Dataset\{0}'.format((name))+'.mat'
#     data = scipy.io.loadmat(file_path)
#     # if os.path.exists(file_path):
#     #     # 文件路径存在，进行打开操作
#     #     data = h5py.File(file_path, "r")
#     # else:
#     #     print("File path not found!")
#     # # data = h5py.File('D:\MultiView_Dataset\{0}'.format((name))+'.mat','r');
#     num_view=len(data['X'])
#     print( num_view)
#     fea=[]
#     dimension = []
#     for i in range(num_view):
#         # feature = [data[element[i]][:] for element in data['X']]
#         feature = np.array(data['X'])
#         feature=np.squeeze(feature)
#         feature=feature.T
#         # print(feature.shape)
#         feature=normalize(feature)
#         if ss.isspmatrix(feature):
#             feature = feature.todense()
#         feature=torch.from_numpy(feature).float().to(device)
#         fea.append(feature)
#         dimension.append(feature.shape[1])
#         del feature
#     Y=np.array(data['Y'])
#     Y=Y.T
#     Y = Y - min(Y)
#     Y = torch.from_numpy(Y).long()
#     # print(fea[0][0].shape)
#     # a=[]
#     # for i in range(num_view):
#     #     a.append(fea[i][0].numpy())
#     # a=np.array(a[0])
#     return fea,Y,num_view,dimension

def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    return ACC, F1

def search_2(n):
    i=2
    while(i<=n):
        i=i*2
    return i

