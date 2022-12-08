import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflow as nf

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


# Set up model

# Define flows

import argparse
 
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-mb", "--mbase")
parser.add_argument("-sc", "--scale")
parser.add_argument("-nc", "--ncomp")
parser.add_argument("-nu", "--nunit")
parser.add_argument("-b", "--base")
parser.add_argument("-trainable", "--trainablebase")

args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
mb = float(args.mbase)
sc = float(args.scale)
nc = int(args.ncomp)
nu = int(args.nunit)
tparam = bool(int(args.trainablebase))
based = str(args.base)
print(cb,mb,sc,nc,nu,tparam,based)

# for cb in [1.0001,2.,3.,10.,50.]:
#     for mb in [1.0001,2.,3.,10.,50.]:
#         for sc in [1.,2.,3.,4.,5.]:
#             for nc in [2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50,100,200,300,500,1000]:        
max_iter = 10000
num_samples = 2 ** 11
anneal_iter = 7000
annealing = True
show_iter = 25
# nc = 3
# mb = 1.00015
# cb = 1.00015
# scale = 1.

K = nu
torch.manual_seed(0)

latent_size = 30
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Set prior and q0
prior = nf.distributions.target.NealsFunnel(v1shift = mb, v2shift = 0.)
#q0 = nf.distributions.DiagGaussian(2)
q0 = nf.distributions.base.MultivariateGaussian()


weight = torch.ones(nc,device='cuda')
mbase = torch.tensor(mb,device='cuda')
vbase = torch.tensor(cb,device='cuda')
scale = torch.tensor(sc,device='cuda')

#q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,trainable = tparam)
print('~~~~~~~~',based)
if based == 'GaussianMixture':
    q0 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = 30, trainable=tparam)
    q1 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = 30, trainable=tparam)
elif based == 'GMM':
    q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,dim=30,trainable = tparam)
    q1 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,dim=30,trainable = tparam)
elif based == 'T':
    q0 = nf.distributions.base.T(n_dim=30, df = cb,trainable = tparam)
    q1 = nf.distributions.base.T(n_dim=30, df = cb,trainable = tparam)
elif based == 'GGD':
    q0 = nf.distributions.base.GGD(n_dim=30, beta = cb,trainable = tparam)
    q1 = nf.distributions.base.GGD(n_dim=30, beta = cb,trainable = tparam)

elif based == 'MultivariateGaussian':
    q0 = nf.distributions.base.MultivariateGaussian(n_dim=30,trainable=tparam)
    q1 = nf.distributions.base.MultivariateGaussian(n_dim=30,trainable=tparam)

with torch.no_grad():
    sample3,_ = q1.forward(20000)
    print(sample3.shape)
    sample3 = pd.DataFrame(sample3.detach().cpu().numpy())

# Construct flow model

#q0 = nf.distributions.base.DiagGaussian(shape=30)

nfm = nf.NormalizingFlow(q0=q0, flows=flows)
X = pd.read_csv('/home/samiri/PhD/Synth/VCNF/prep.csv')
X = X.drop(['Unnamed: 0'],1)
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Original dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
train_iter = iter(train_loader)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfm = nfm.double()

# Initialize ActNorm
# z, _ = nfm.sample(num_samples=2 ** 16)
# z_np = z.to('cpu').data.numpy()

# Plot prior distribution
# grid_size = 300
# xx, yy = torch.meshgrid(torch.linspace(-5, 5, grid_size), torch.linspace(-5, 5, grid_size))
# zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
# zz = zz.double().to(device)
# log_prob = prior.log_prob(zz).to('cpu').view(*xx.shape)
# prob_prior = torch.exp(log_prob)

# # Plot initial posterior distribution
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.cpu().detach().abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

# Train model


loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
sample0,_ = nfm.sample(20000)
sample0 = pd.DataFrame(sample0.cpu().detach().numpy())
gzarr = []
gzparr = []
phist = []
grads = []
wb = []
for it in tqdm(range(max_iter)):

    
        
#     if ~(torch.isnan(loss) | torch.isinf(loss)):
#         loss.backward()
#         optimizer.step()

#     loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
#     del(x, y, loss)
    
    
    
    

    oldm = nfm.state_dict
    try:
        try:
            x = next(train_iter)
            xt = x[0].to('cuda')

        except StopIteration:
            train_iter = iter(train_loader)
            x= next(train_iter)
            xt = x[0].to('cuda')
                
        optimizer.zero_grad()
        
        #loss = model.forward_kld(x.to(device), y.to(device))
        if annealing:
            #print('!!!!!!',x[0].shape)
            loss = nfm.forward_kld(xt)#.to('cuda')
            #print('111111111111111111111111')
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())

        else:
            loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)
            #print('222222222222222222222222')
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())
            

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward(retain_graph=True)
            with torch.no_grad():
                a = [p.grad for n, p in nfm.named_parameters()]
                #print(a[3].mean(),a[4].mean())
                grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
            
            #print('==================================================================================================================')
                #grads.append(a)     
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())

            optimizer.step()
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}
        #wb.append(pm)


        #
        
        #print(nfm.q0.mbase.detach().cpu().item())

        # Plot learned posterior
        if (it + 1) % show_iter == 0:
            
            


            # gzarr.append(zarr)
            # gzparr.append(zparr)

            if based == 'GaussianMixture':
                phist.append([nfm.q0.loc.detach().cpu().numpy(),nfm.q0.log_scale.detach().cpu().numpy(),nfm.q0.weight_scores.detach().cpu().numpy()])

            elif based == 'GMM':
                phist.append([nfm.q0.mbase.detach().cpu().item(),nfm.q0.vbase.detach().cpu().item(),nfm.q0.scale.detach().cpu().item(),nfm.q0.weight.detach().cpu().numpy()])
            elif based == 'GGD':
                phist.append([nfm.q0.loc.detach().cpu().numpy(),nfm.q0.scale.detach().cpu().numpy(),nfm.q0.p.detach().cpu().numpy()])
            elif based == 'T':
                phist.append([nfm.q0.df.detach().cpu().numpy()])

            elif based == 'MultivariateGaussian':
                phist.append([nfm.q0.loc.detach().cpu().numpy(),nfm.q0.scale.detach().cpu().numpy()])

        #     log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        #     prob = torch.exp(log_prob)
        #     prob[torch.isnan(prob)] = 0

        #     plt.figure(figsize=(15, 15))
        #     plt.pcolormesh(xx, yy, prob.data.numpy())
        #     plt.contour(xx, yy, prob_prior.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #     plt.gca().set_aspect('equal', 'box')
        #     plt.show()
    except Exception as e:
        print(e)
        traceback.print_exc()                        
        nfm.state_dict = oldm


# Plot learned posterior distribution
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0


#sample1 = pd.DataFrame(prior.sample(20000).cpu().detach().numpy())
sample2,_ = nfm.sample(20000)
sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
sample4,_ = nfm.q0.forward(20000)
sample4 = pd.DataFrame(sample4.detach().cpu().numpy())

torch.save(nfm, f'/home/samiri/PhD/Synth/VCNF/logs/model_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample0.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#sample1.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/target_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample2.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample3.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedbase_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample4.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedbase_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
pd.DataFrame(loss_hist).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/losshist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#pickle.dump(gzarr, open( f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
#pickle.dump(gzparr, open( f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
#pd.DataFrame(gzarr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#pd.DataFrame(gzparr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
pd.DataFrame(phist).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/phist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(grads, f'/home/samiri/PhD/Synth/VCNF/logs/grads_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#torch.save(wb, f'/home/samiri/PhD/Synth/VCNF/logs/wb_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')

