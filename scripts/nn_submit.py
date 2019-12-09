import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.decomposition import PCA
import itertools

class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.tensor(X, requires_grad=True)
        if not torch.is_tensor(y):
            self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_pic_mats(n_components):
    pics_train = np.load('pics_train_tf.npy')
    pics_test = np.load('pics_test_tf.npy')

    pca = PCA(n_components=n_components)
    pca.fit(pics_train)
    pics_train = pca.transform(pics_train)
    pics_test = pca.transform(pics_test)
    return pics_train, pics_test

def fit(X,y,n_epochs,H_1):
    ds = PrepareData(X=X, y=y)
    dl = DataLoader(ds, batch_size=25, shuffle=True)
    
    device = torch.device('cpu')
    
    D_in, D_out = X.shape[1], y.shape[1]
    
    model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H_1),
#           torch.nn.Dropout(0.2),
#           torch.nn.BatchNorm1d(H_1),
          torch.nn.ReLU(),
          torch.nn.Linear(H_1, D_out),
        ).to(device)
    
    def loss_fn(y_pred, y):
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # return 1 - cos(y_pred, y).mean()
    #     return torch.norm((y_pred-y), p=2, dim=1).mean()
        s = y_pred.size()[0]
        c = 2*s**2-2*s
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        res = 1 - cos(y_pred[None,:,:], y[:,None,:])
        ret = c - (res.sum() - s*torch.diag(res).sum())
        if ret.item()<0:
            print('found neg')
            print(s,res)
        return ret
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5.5e-05, weight_decay=3.5)
    for t in range(n_epochs):
        for ix, (_x, _y) in enumerate(dl):
            _x = Variable(_x).float()
            _y = Variable(_y).float()

            y_pred = model(_x)

            loss = loss_fn(y_pred, _y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # y_pred = model(Variable(ds.X).float())
        # train_loss = loss_fn(y_pred, Variable(ds.y).float()).data.numpy()
        # #losses[t] = train_loss
        # print(t,train_loss)

        # ave, map_ = evaluate(y_pred.data.numpy(),ds.y.data.numpy())
        # ave = ave * 2000/X.shape[0]
        # map_ = map_ * X.shape[0]/2000
        # print("""Iter %d: Train Ave Rank: %g, Train MAP@20: %g""" % (t,ave,map_))

        if (t+1)%15==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 1.15
    
    return model

def predict(model,X):
    return model(Variable(torch.from_numpy(X)).float()).data.numpy()

def get_prediction(vecs,pics):
    dists = pairwise_distances(vecs,pics,metric='cosine')
    return dists.argsort(1)

def map_20(ranks):
    return np.mean([(20-rank)/20 if rank<20 else 0 for rank in ranks])

def evaluate(vectors,label_vectors):
    preds = get_prediction(vectors,label_vectors)
    ranks = [np.argwhere(vec==i)[0][0] for i,vec in enumerate(preds)]
    return np.mean(ranks),map_20(ranks)

def get_top_20(preds,descr_id):
    return preds[descr_id][:20]

def save_submission(preds):
    data = []
    for i in range(2000):
        data.append(['%d.txt' % (i,),' '.join('%d.jpg' % (pic_id,) for pic_id in get_top_20(preds,i))])
    pd.DataFrame(data,columns=['Descritpion_ID','Top_20_Image_IDs']).to_csv('submission.csv',index=False)

if __name__=="__main__":

    text_train = np.load('text_train_tf.npy')
    text_test = np.load('text_test_tf.npy')

    pics_train, pics_test = get_pic_mats(100)
    model = fit(text_train,pics_train,54,2048)
    vecs = predict(model,text_test)
    preds = get_prediction(vecs,pics_test)
    save_submission(preds)


