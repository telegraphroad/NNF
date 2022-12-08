import sys
nm,num_layers,lw = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
# for nm in [50,40,30,20,15,10]:
#     for num_layers in [50,45,40,35,30,25,20,15]:
#         for lw in [120,100,80,60,50,40,30,20]:
import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflows as nf

import argparse


import pandas as pd
from torch.utils.data import TensorDataset

from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import traceback
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import traceback
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.distributions as D
def distancesum (arr, n):

    # sorting the array.
    arr.sort()

    # for each point, finding
    # the distance.
    res = 0
    sum = 0
    for i in range(n):
        res += (arr[i] * i - sum)
        sum += arr[i]

    return res

def totaldistancesum( x , y , n ):
    return distancesum(x, n) + distancesum(y, n)

print(f'============================================={nm}_{lw}_{num_layers}')
max_iter = 10000
latent_size = 30

X = pd.read_csv('/home/samiri/PhD/Synth/VCNF/prep.csv')
X = X.drop(['Unnamed: 0'],1)
Y = X.Class
#X = X.drop(['Class'],1)
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Original dataset
scaler = MinMaxScaler()
scaler.fit(X)
XS = scaler.transform(X)
Xc = X.columns
del X,Y
dataset = TensorDataset(torch.tensor(XS, dtype=torch.float32))
num_samples = 2**9
train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
train_iter = iter(train_loader)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Set up model

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)
base = nf.distributions.base.MixtureofGaussians(n_modes = nm, dim = latent_size, trainable=True,loc_mult=2.)


ls = latent_size - latent_size//2

flows = []

b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([latent_size, lw * latent_size, lw * latent_size, lw * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, lw * latent_size, lw * latent_size, lw * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Construct flow model
model = nf.NormalizingFlow(base, flows)
bestmodel = nf.NormalizingFlow(base, flows)

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
with torch.no_grad():
    model = model.float().to(device)


# Train model

num_samples = 2 ** 11
show_iter = 500


loss_hist = np.array([])
bestloss = 9999999999999999999999999.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
optimizer2 = torch.optim.Adam(model.q0.parameters(), lr=1e-4, weight_decay=1e-6)
checkpoints = []
plt.figure()
fig, ax = plt.subplots(1,2,figsize=[25,10])
a = model.sample(10000)[0].detach().cpu().numpy()
colors = model.log_prob(torch.tensor(a).cuda()).exp().detach().cpu().numpy()  

ax[0].scatter(a[:,0],a[:,1],c=colors)

with torch.no_grad():
    model = model.float()
model.train()
compiled_nfm = torch.compile(model)
compiled_nfm.train()
mdsarr = []
bm = 0
loss = 9999999999999999999999999.
pbar = tqdm(range(max_iter), desc=f'Current loss:{loss}, Best loss:{bestloss}')

for it in pbar:

    optimizer.zero_grad()

    # Get training samples
    try:
        x = next(train_iter)
        with torch.no_grad():
            xt = x[0].to('cuda')#.float().to('cuda')

    except StopIteration:
        train_iter = iter(train_loader)
        x= next(train_iter)
        with torch.no_grad():
            xt = x[0].to('cuda')#.float().to('cuda')

    x = xt



    # Compute loss
    loss, z, log_q, z_layers, ld_layers = compiled_nfm.forward_kld(x,extended=True)
    del z_layers,ld_layers,log_q
    # xs = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    # ys = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    # mds = totaldistancesum(xs, ys, len(xs))
    # mdsarr.append(mds)
    # loss = loss# - torch.tensor(mds)
    #print(loss.item(),'========',mds)

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    #print(loss.to('cpu').data.numpy(),bestloss)
    #pbar.set_description(f'Current loss:{loss}, Best loss:{bestloss} in iteration {bm}')

    with torch.no_grad():
        compiled_nfm.q0.forward(5000)

    if (loss.to('cpu').data.numpy()< bestloss):
        bestmodel.state_dict = deepcopy(model.state_dict)
        bestloss = loss.to('cpu').data.numpy()
        bm = it
        pbar.set_description(f'Best loss:{bestloss} in iteration {bm}')



# except Exception as e:
#     print(e)
#traceback.print_exc()                        
del a,checkpoints
torch.save(bestmodel,f'zbestmodeltriple_{nm}_{lw}_{num_layers}.ph')
torch.save(model,f'zmodeltriple_{nm}_{lw}_{num_layers}.ph')
# import pandas as pd
# cat_cols = []
# bestmodel.cpu()
# model.cpu()
# bestmodel.eval()
# model.eval()
# bestmodel.cpu()
# model.cpu()

# del dataset,train_loader,base,flows,loss_hist

# real = pd.DataFrame(XS)
# fake = pd.DataFrame(bestmodel.sample(len(real))[0].detach().cpu().numpy())

# fake.columns = Xc
# real.columns = Xc
# plt.figure()
# #plt.plot(np.clip(loss_hist,-100.,100.))
# import seaborn as sns
# fig, ax = plt.subplots(5,len(fake.columns)//5 + 1,figsize=[25,25])

# for i,column in enumerate(np.sort(fake.columns)):
#     if column != 'Class':
#         #print(column)
#         sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,color='red')
#         sns.histplot(real[column],ax=ax.flatten()[i],kde=True,color='green')
#     else:
#         print(column)
#         sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,color='red',discrete=True)
#         sns.histplot(real[column],ax=ax.flatten()[i],kde=True,color='green',discrete=True)
# plt.savefig(f'zplotdoublebest_{nm}_{lw}_{num_layers}.png')    
# del fake,bestmodel


# fake = pd.DataFrame(model.sample(len(real))[0].detach().cpu().numpy())
# #fake.to_csv('./fake.csv')
# fake.columns = Xc
# plt.figure()
# #plt.plot(np.clip(loss_hist,-100.,100.))
# import seaborn as sns
# fig, ax = plt.subplots(5,len(fake.columns)//5 + 1,figsize=[25,25])

# for i,column in enumerate(np.sort(fake.columns)):
#     if column != 'Class':
#         #print(column)
#         sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,color='red')
#         sns.histplot(real[column],ax=ax.flatten()[i],kde=True,color='green')
#     else:
#         print(column)
#         sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,color='red',discrete=True)
#         sns.histplot(real[column],ax=ax.flatten()[i],kde=True,color='green',discrete=True)
# plt.savefig(f'zplotdoublelast_{nm}_{lw}_{num_layers}.png')    
# del fake
