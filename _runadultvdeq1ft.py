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
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import wandb
wandb.init()
wandb.login()
# Set up model

# Define flows

import argparse

from normflow import utils
 
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-mb", "--mbase")
parser.add_argument("-sc", "--scale")
parser.add_argument("-nc", "--ncomp")
parser.add_argument("-nu", "--nunit")
parser.add_argument("-b", "--base")
#parser.add_argument("-ds", "--dataset")
parser.add_argument("-trainable", "--trainablebase")

args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
mb = float(args.mbase)
sc = float(args.scale)
nc = int(args.ncomp)
nu = int(args.nunit)
#ds = int(args.dataset)
tparam = bool(int(args.trainablebase))
based = str(args.base)
print(cb,mb,sc,nc,nu,tparam,based)

# for cb in [1.0001,2.,3.,10.,50.]:
#     for mb in [1.0001,2.,3.,10.,50.]:
#         for sc in [1.,2.,3.,4.,5.]:
#             for nc in [2,3,4,5,6,7,8,9,10,12,15,20,25,30,40,50,100,200,300,500,1000]:        
max_iter = 15000
num_samples = 2 ** 11
anneal_iter = 12000
annealing = True
show_iter = 25
# nc = 3
# mb = 1.00015
# cb = 1.00015
# scale = 1.

K = nu
torch.manual_seed(0)

latent_size = 15
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []


vquantizers = []
categorical = [1,3,4,5,6,7,8,9,13,14]
categorical_qlevels = [9,16,16,7,15,6,5,2,42,2]
#categorical = [13]
#categorical_qlevels = [42]
catlevels = [2]
lcm = utils.utils.lcm(categorical_qlevels)
vlayers = []
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

for i in range(8):
    vlayers += [nf.flows.ActNorm(1)]
    s = nf.nets.MLP([1, 4, 1], init_zeros=True)
    t = nf.nets.MLP([1, 4, 1], init_zeros=True)
    if i % 2 == 0:
        vlayers += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        vlayers += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    vlayers += [nf.flows.ActNorm(1)]

#vlayers.reverse()
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])

vquantizers = [nf.nets.VariationalDequantization(var_flows=torch.nn.ModuleList(vlayers),quants = categorical_qlevels[i]) for i in range(len(categorical))]
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
    q0 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = latent_size, trainable=tparam)
    q1 = nf.distributions.base.GaussianMixture(n_modes = nc, dim = latent_size, trainable=tparam)
elif based == 'GMM':
    q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,dim=latent_size,trainable = tparam)
    q1 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,dim=latent_size,trainable = tparam)
elif based == 'T':
    q0 = nf.distributions.base.T(n_dim=latent_size, df = cb,trainable = tparam)
    q1 = nf.distributions.base.T(n_dim=latent_size, df = cb,trainable = tparam)
elif based == 'GGD':
    q0 = nf.distributions.base.GGD(n_dim=latent_size, beta = cb,trainable = tparam)
    q1 = nf.distributions.base.GGD(n_dim=latent_size, beta = cb,trainable = tparam)

elif based == 'MultivariateGaussian':
    q0 = nf.distributions.base.MultivariateGaussian(n_dim=latent_size,trainable=tparam)
    q1 = nf.distributions.base.MultivariateGaussian(n_dim=latent_size,trainable=tparam) 

with torch.no_grad():
    sample3,_ = q1.forward(20000)
    print(sample3.shape)
    sample3 = pd.DataFrame(sample3.detach().cpu().numpy())

# Construct flow model

#q0 = nf.distributions.base.DiagGaussian(shape=30)

nfm = nf.NormalizingFlow(q0=q0, flows=flows,categoricals=categorical,catlevels=catlevels,catvdeqs=vquantizers)
nfmBest = nf.NormalizingFlow(q0=q0, flows=flows,categoricals=categorical,catlevels=catlevels,catvdeqs=vquantizers)

X = pd.read_csv('/home/samiri/PhD/Synth/VCNF/adult.csv')
X = X.drop(['Unnamed: 0'],1)
xcol = X.columns
for ii in range(len(categorical)):
    X[X.columns[categorical[ii]]] = X[X.columns[categorical[ii]]] * lcm / categorical_qlevels[ii]

# Original dataset
###################### SCALER
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#X = X.values
###################### SCALER



#X = X[X[:,-1]==sc]
#X = X[:,0:-1]

X = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
train_iter = iter(train_loader)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfmBest = nfmBest.to(device)
nfm = nfm.double()
nfmBest = nfmBest.double()

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
nfm.eval()
wandb.watch(nfm, log_freq=show_iter)
nfm.train()

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
phistg = []
logq = []
gradssteps = []
closs = 1e20
nfmBest.state_dict = nfm.state_dict
for it in tqdm(range(max_iter)):

    
        
#     if ~(torch.isnan(loss) | torch.isinf(loss)):
#         loss.backward()
#         optimizer.step()

#     loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
#     del(x, y, loss)
    
    
    
    

    oldm = nfm.state_dict
    oldp = q0.parameters

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
            loss,zarr, zparr,logqep = nfm.forward_kld(xt, extended=True)#.to('cuda')
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
                asteps = []
                for lctr in range(0,nu):
                    asteps.append([p.grad for n, p in nfm.named_parameters() if (('flows.'+str(lctr)+'.' in n) or ('flows.'+str(lctr+1)+'.' in n))])
                agrads = []
                #print('pgrad',q0.p.grad,'locgrad',q0.loc.grad,'scalegrad',q0.scale.grad)
                #if str(q0) == 'GGD()':
                for lg in asteps:
                    agrads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in lg if i is not None]) if i != 0]))
                #    tgrads.append(q0.p.grad,q0.loc.grad,q0.scale.grad)
                #print([[n,p] for n, p in nfm.named_parameters()])
                #print(a[3].mean(),a[4].mean())
                grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
                gradssteps.append(agrads)
            
            #print('==================================================================================================================')
                #grads.append(a)     
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())

            optimizer.step()
            # plt.figure()
            # plot_grad_flow(nfm.named_parameters())

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        logq.append(logqep.detach().cpu().numpy())

        pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}


        #
        
        #print(nfm.q0.mbase.detach().cpu().item())
        phistg.append([a.grad.detach().cpu().numpy() for a in q0.parameters() if a.grad is not None])

        #wb.append(pm)


        #
        
        #print(nfm.q0.mbase.detach().cpu().item())

        # Plot learned posterior
        if (it + 1) % show_iter == 0:
            wandb.log({"loss": loss})
            wb.append(pm)
            ss,_ = nfmBest.sample(10000)
            #print([np.unique(ss.detach().cpu()[:,i]).shape for i in range(1,15)])
            #print([np.unique(X[:,i]).shape for i in range(1,15)])
            
            


            gzarr.append(zarr)
            gzparr.append(zparr)
            phist.append([a.detach().cpu().numpy() for a in q0.parameters()])
            
            


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
        if loss.to('cpu').data.item()<closs:
            closs = loss.to('cpu').data.item()
            nfmBest.state_dict = nfm.state_dict
            q1.parameters = q0.parameters

    except Exception as e:
        print(e)
        traceback.print_exc()                        
        nfm.state_dict = oldm
        q0.parameters = oldp


# Plot learned posterior distribution
# log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
# prob = torch.exp(log_prob)
# prob[torch.isnan(prob)] = 0


# sample2,_ = nfm.sample(20000)
# sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
# sample4,_ = nfm.q0.forward(20000)
# sample4 = pd.DataFrame(sample4.detach().cpu().numpy())


sample2,_ = nfmBest.sample(32561)
sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
sample2.columns = xcol
for ii in range(len(categorical)):
    sample2[sample2.columns[categorical[ii]]] = np.floor((sample2[sample2.columns[categorical[ii]]] / lcm) * categorical_qlevels[ii])

sample4,_ = nfmBest.q0.forward(32561)
sample4 = pd.DataFrame(sample4.detach().cpu().numpy())

torch.save(nfmBest, f'/home/samiri/PhD/Synth/VCNF/logs/model_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample0.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample2.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedmodel_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample3.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/untrainedbase_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
sample4.to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/trainedbase_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
pd.DataFrame(loss_hist).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/losshist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#pickle.dump(gzarr, open( f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
#pickle.dump(gzparr, open( f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth', 'wb'))
#pd.DataFrame(gzarr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/z_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
pd.DataFrame(logq).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/logq_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
#pd.DataFrame(gzparr).to_csv(f'/home/samiri/PhD/Synth/VCNF/logs/zp_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(phist,f'/home/samiri/PhD/Synth/VCNF/logs/phist_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(phistg,f'/home/samiri/PhD/Synth/VCNF/logs/phistg_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(grads, f'/home/samiri/PhD/Synth/VCNF/logs/grads_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(gradssteps, f'/home/samiri/PhD/Synth/VCNF/logs/gradssteps_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')
torch.save(wb, f'/home/samiri/PhD/Synth/VCNF/logs/wb_nc_{nc}_cb_{cb}_mb_{mb}_scale_{sc}_trainable_{tparam}_nunit_{nu}_base_{based}.pth')

