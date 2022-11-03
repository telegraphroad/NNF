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

# Set up model

# Define flows

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

dataset = TensorDataset(torch.tensor(XS, dtype=torch.float32))
num_samples = 2**11
train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
train_iter = iter(train_loader)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Set up model

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)
base = nf.distributions.base.GaussianMixture(n_modes = 8, dim = latent_size, trainable=True)

# Define list of flows
#latent_size = 30
# b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
# flows = []
# for i in range(K):
#     s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
#     t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
#     if i % 2 == 0:
#         flows += [nf.flows.MaskedAffineFlow(b, t, s)]
#     else:
#         flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
#     flows += [nf.flows.ActNorm(latent_size)]


ls = latent_size - latent_size//2
num_layers = 16
flows = []
# for i in range(num_layers):
#     # Neural network with two hidden layers having 64 units each
#     # Last layer is initialized by zeros making training more stable
#     param_map = nf.nets.MLP([ls, 64, 64, latent_size], init_zeros=True)
#     # Add flow layer
#     flows.append(nf.flows.AffineCouplingBlock(param_map))
#     # Swap dimensions
#     flows.append(nf.flows.Permute(2, mode='swap'))

b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
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



# m2 = D.Normal(loc=torch.tensor([8.,8.]),scale=torch.tensor([0.5,0.5]))
# smpl = base.sample([1000]).cuda()

# z,l = bestmodel.forward_kld(smpl,extended=True)
# z = z.cpu().detach().numpy()

# ax[1].scatter(z[:,0],z[:,1])
# means = model.q0.loc.squeeze().cpu().detach().numpy()
# covs = model.q0.log_scale.exp().squeeze().cpu().detach().numpy()


# ax[1].scatter(means[:,0],means[:,1],color='red')
# ax[1].set_title('Samples from "tail area" mapped to latent space')
# for i, txt in enumerate(means[:,0]):
#     ax[1].annotate(txt.round(decimals=4), (means[i,0], means[i,1]))
# model.train()

# model.eval()

# # plt.figure(figsize=(15, 15)) 
# # plt.pcolormesh(xx, yy, prob.data.numpy())
# # plt.gca().set_aspect('equal', 'box')
# # plt.show()
# plt.figure()
# fig, ax = plt.subplots(1,2,figsize=[25,10])
# a = model.sample(10000)[0].detach().cpu().numpy()
# colors = model.log_prob(torch.tensor(a).cuda()).exp().detach().cpu().numpy()  

# ax[0].scatter(a[:,0],a[:,1],c=colors)



# m2 = D.Normal(loc=torch.tensor([8.,8.]),scale=torch.tensor([0.5,0.5]))
# smpl = m2.sample([1000]).cuda()
# m2 = D.Normal(loc=torch.tensor([5.,5.]),scale=torch.tensor([0.9,0.9]))
# smpl2 = m2.sample([1000]).cuda()

# z,l = bestmodel.forward_kld(smpl,extended=True)
# z = z.cpu().detach().numpy()

# ax[1].scatter(z[:,0],z[:,1])
# z,l = bestmodel.forward_kld(smpl2,extended=True)
# z = z.cpu().detach().numpy()

# ax[1].scatter(z[:,0],z[:,1])
# means = model.q0.loc.squeeze().cpu().detach().numpy()
# covs = model.q0.log_scale.exp().squeeze().cpu().detach().numpy()


# ax[1].scatter(means[:,0],means[:,1],color='red')
# ax[1].set_title('Samples from "tail area" mapped to latent space')
# for i, txt in enumerate(means[:,0]):
#     ax[1].annotate(txt.round(decimals=4), (means[i,0], means[i,1]))
with torch.no_grad():
    model = model.float()
model.train()
mdsarr = []



for it in tqdm(range(max_iter)):

    
    
    optimizer.zero_grad()
    
    # Get training samples
    try:
        x = next(train_iter)
        with torch.no_grad():
            xt = x[0].float().to('cuda')

    except StopIteration:
        train_iter = iter(train_loader)
        x= next(train_iter)
        with torch.no_grad():
            xt = x[0].float().to('cuda')
    
    x = xt
    # Compute loss
    loss = model.forward_kld(x)
    xs = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    ys = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    mds = totaldistancesum(xs, ys, len(xs))
    mdsarr.append(mds)
    loss = loss# - torch.tensor(mds)
    #print(loss.item(),'========',mds)
    
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    
    # Log loss
    #loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    
    if loss.detach().cpu().item() < bestloss:
        bestmodel.state_dict = model.state_dict
        bestloss = loss.detach().cpu().item()
    #############################################################
        
        
        
    # for z in range(10):    
        
    #     optimizer2.zero_grad()

    #     # Get training samples
    #     try:
    #         x = next(train_iter)
    #         with torch.no_grad():
    #             xt = x[0].float().to('cuda')

    #     except StopIteration:
    #         train_iter = iter(train_loader)
    #         x= next(train_iter)
    #         with torch.no_grad():
    #             xt = x[0].float().to('cuda')
        
    #     x = xt

    #     # Compute loss
    #     loss = model.forward_kld(x)
    #     xs = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    #     ys = model.q0.loc.squeeze().detach().cpu().numpy()[:,0]
    #     mds = totaldistancesum(xs, ys, len(xs))
    #     loss = loss - 100.*torch.tensor(mds)
    #     #print(loss.item(),'========',mds)


    #     # Do backprop and optimizer step
    #     if ~(torch.isnan(loss) | torch.isinf(loss)):
    #         loss.backward()
    #         optimizer2.step()


    #     # Log loss
    #     loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())


    #     if loss.detach().cpu().item() < bestloss:
    #         bestmodel.state_dict = model.state_dict
    #         bestloss = loss.detach().cpu().item()
        
        
        
        
        
        
        
    #############################################################    
    # Plot learned posterior
    if (it + 1) % show_iter == 0:
        checkpoints.append(model.state_dict)
        model.eval()

        # plt.figure(figsize=(15, 15)) 
        # plt.pcolormesh(xx, yy, prob.data.numpy())
        # plt.gca().set_aspect('equal', 'box')
        # plt.show()
        plt.figure()
        fig, ax = plt.subplots(1,2,figsize=[25,10])
        a = model.sample(10000)[0].detach().cpu().numpy()
        colors = model.log_prob(torch.tensor(a).cuda()).exp().detach().cpu().numpy()  

        ax[0].scatter(a[:,0],a[:,1],c=colors)
        
        

        # m2 = D.Normal(loc=torch.tensor([8.,8.]),scale=torch.tensor([0.5,0.5]))
        # smpl = m2.sample([1000]).cuda()
        # m2 = D.Normal(loc=torch.tensor([5.,5.]),scale=torch.tensor([0.9,0.9]))
        # smpl2 = m2.sample([1000]).cuda()

        # z,l = bestmodel.forward_kld(smpl,extended=True)
        # z = z.cpu().detach().numpy()
        
        # ax[1].scatter(z[:,0],z[:,1])
        # z,l = bestmodel.forward_kld(smpl2,extended=True)
        # z = z.cpu().detach().numpy()
        
        # ax[1].scatter(z[:,0],z[:,1])
        # means = model.q0.loc.squeeze().cpu().detach().numpy()
        # covs = model.q0.log_scale.exp().squeeze().cpu().detach().numpy()


        # ax[1].scatter(means[:,0],means[:,1],color='red')
        # ax[1].set_title('Samples from "tail area" mapped to latent space')
        # for i, txt in enumerate(means[:,0]):
        #     ax[1].annotate(txt.round(decimals=4), (means[i,0], means[i,1]))
        model.train()
        

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.figure(figsize=(10, 10))
plt.plot(mdsarr, label='sum distance')
plt.legend()
plt.show()