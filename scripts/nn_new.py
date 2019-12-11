import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import os
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import itertools
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.tensor(X, requires_grad=True).float()
        if not torch.is_tensor(y):
            self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def fit(X,y,X_val,y_val,H_1,lr,weight_decay,batch_size):
    losses, train_ave, train_map, val_ave, val_map, n_epochs = {}, {}, {}, {}, {}, 0
    
    ds = PrepareData(X=X, y=y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cpu')
    
    D_in, D_out = X.shape[1], y.shape[1]
    H_1, H_2 = 2048, 2048
    
    model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H_1),
#           torch.nn.Dropout(0.2),
#           torch.nn.BatchNorm1d(H_1),
          torch.nn.ReLU(),
          torch.nn.Linear(H_1, H_2),
          torch.nn.ReLU(),
          torch.nn.Linear(H_2, D_out),
        ).to(device)
    
    def loss_fn(y_pred, y):
        # loss 1 - simple L2 norm of different (euclidean)
        # return torch.norm((y_pred-y), p=2, dim=1).mean()
        
        ## loss 2 - cosine distance 
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # return 1 - cos(y_pred, y).mean()
    
        ## loss 3 - cosine distance of label and prediction minus
        ## every other pair (using out product)     
        s = y_pred.size()[0]
        c = 2*s**2-2*s
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        res = 1 - cos(y_pred[None,:,:], y[:,None,:])
        ret = (c - (res.sum() - s*torch.diag(res).sum()))/s**2
        return ret
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr/2, momentum=0.9)
    # iters = len(dl)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=iters, epochs=40)
    for t in range(100000):
        train_losses = []

        for ix, (_x, _y) in enumerate(dl):
            optimizer.zero_grad()

            y_pred = model(_x)
            loss = loss_fn(y_pred, _y)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            # scheduler.step(t + ix / iters)

        train_loss = np.mean(train_losses)
        losses[t] = train_loss

        y_pred_val = model(Variable(torch.from_numpy(X_val)).float())
        ave_val, map_val = evaluate(y_pred_val.data.numpy(),y_val)
        val_ave[t] = ave_val * 2000/X_val.shape[0]
        val_map[t] = map_val * X_val.shape[0]/2000
        n_epochs = t
        print("""Iter %d: Train Loss: %g, Val Ave Rank: %g, Val MAP@20: %g""" % (t,losses[t],val_ave[t],val_map[t]))
        
        best_val_so_far = val_map[max(val_map,key=val_map.get)]
        best_loss_so_far = losses[min(losses,key=losses.get)]
        # if val_map[t]<best_val_so_far*.95 or losses[t]>1.08*best_loss_so_far:
        #     print("EARLY STOP TRIGGERED BECAUSE VAL MAP HAS NOT IMPROVED FROM BEST")
        #     return model, losses, train_ave, train_map, val_ave, val_map, max(val_map,key=val_map.get)

        # if (t+1)%10==0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 1.10
    
    return model, losses, train_ave, train_map, val_ave, val_map, n_epochs

def predict(model,X):
    return model(Variable(torch.from_numpy(X)).float()).data.numpy()

def get_prediction(vecs,pics):
    dists = pairwise_distances(vecs,pics,metric='cosine')
    return dists.T.argsort(1)

def map_20(ranks):
    return np.mean([(20-rank)/20 if rank<20 else 0 for rank in ranks])

def evaluate(vectors,label_vectors):
    preds = get_prediction(vectors,label_vectors)
    ranks = [np.argwhere(vec==i)[0][0] for i,vec in enumerate(preds)]
    return np.mean(ranks),map_20(ranks)

def get_num(string):
    string = string.replace('.', ' ').replace('/', ' ')
    num = [int(s) for s in string.split() if s.isdigit()]
    return num[0]

def parse_to_numpy(pd):
    images_idx = []
    for string in pd[0]:
        images_idx.append(get_num(string))

    pd.insert(1, "Image_Index", images_idx, True)
    pd = pd.sort_values(by=['Image_Index'])
    pd = pd.reset_index(drop=True)
    del pd['Image_Index']
    del pd[0]
    np = pd.to_numpy()
    return np

if __name__=="__main__":

    ### Description for train data
    desc_files = len(os.listdir('../descriptions_train'))
    all_desc_train = []

    for i in range(desc_files):
        empty_str = ''
        for line in open(f'../descriptions_train/{i}.txt'):
            empty_str += line.replace('\n',' ')
        all_desc_train.append(empty_str)

    ### Tags for train data
    tag_files = len(os.listdir('../tags_train'))
    all_tags_train = []

    for i in range(tag_files):
        nouns = ''
        for line in open(f'../tags_train/{i}.txt'):
            nouns += line.replace(':',' ')
        all_tags_train.append(nouns.replace('\n', ' '))

    all_docs = all_desc_train + all_tags_train

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    vectorizer.fit(all_docs)

    train_desc_bow = np.array(vectorizer.transform(all_desc_train).todense())
    train_tags_bow = np.array(vectorizer.transform(all_tags_train).todense())

    train_1000 = pd.read_csv('../features_train/features_resnet1000_train.csv', header=None)
    train_1000 = parse_to_numpy(train_1000)
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    train_desc = embed(all_desc_train).numpy()
    train_tags = embed(all_tags_train).numpy()

    train_pic = np.hstack((train_1000, train_tags, train_tags_bow))
    train_desc = np.hstack((train_desc, train_desc_bow))


    n_hidden_units = [1024]
    learning_rates = [0.00001]
    weight_decay = [.2]#[2.2,3.5]
    batch_size = [32]

    results = {}

    for n_h, lr, wd, bs in itertools.product(n_hidden_units,learning_rates,weight_decay,batch_size):
        print("************ n_h=%d, lr=%g, wd=%g, bs=%d ************" % (n_h,lr,wd,bs))
        cv = KFold(n_splits=5)
        n_epochs_results = []
        val_map_results = []
        val_ave_results = []
        for train_index,test_index in cv.split(train_pic,train_desc):
            rets = fit(train_pic[train_index],train_desc[train_index],
                       train_pic[test_index],train_desc[test_index],n_h,lr,wd,bs)
            model, losses, train_ave, train_map, val_ave, val_map, n_epochs = rets
            n_epochs_results.append(n_epochs)
            val_map_results.append(val_map)
            val_ave_results.append(val_ave)
        print(n_epochs_results,np.mean(n_epochs_results))
        best_epoch = int(np.mean(n_epochs_results))
        nearest_epoch = [best_epoch if len(d)>=best_epoch else max(d) for d in val_map_results]
        print(nearest_epoch)
        vals = [d[ne] for d,ne in zip(val_map_results,nearest_epoch)]
        aves = [d[ne] for d,ne in zip(val_ave_results,nearest_epoch)]
        print(vals,np.mean(vals))
        results[(n_h, lr, wd, bs)] = (np.mean(vals),np.mean(aves),best_epoch)

        print(results)
