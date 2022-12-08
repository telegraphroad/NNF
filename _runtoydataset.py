import torch
#torch.autograd.set_detect_anomaly(True)
import numpy as np
import normflows as nf

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
import torch.distributions as D
import wandb
from torch.utils.tensorboard import SummaryWriter

# wandb.login()
# wandb.init(project="nfm", entity="sabaa")
# Set up model

# Define flows

import argparse

from normflows import utils


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



parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-nc", "--ncomp")
parser.add_argument("-nu", "--nunit")
parser.add_argument("-b", "--base")
parser.add_argument("-t", "--target")
#parser.add_argument("-ds", "--dataset")
parser.add_argument("-p", "--param")
parser.add_argument("-trainable", "--trainablebase")

args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
nc = int(args.ncomp)
nu = int(args.nunit)
#ds = int(args.dataset)
tparam = bool(int(args.trainablebase))
based = str(args.base)
targetD = str(args.target)

max_iter = 10000
num_samples = 2 ** 11
anneal_iter = 3000
annealing = True
show_iter = 50
# nc = 3
# cb = 1.00015
# scale = 1.
prm = 0.
K = nu
torch.manual_seed(0)

latent_size = 3
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []


vquantizers = []
categorical = [2]
categorical_qlevels = [2]
#categorical = [13]
#categorical_qlevels = [42]
catlevels = [2]
lcm = utils.utils.lcm(categorical_qlevels)
vlayers = []
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

for i in range(4):
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
    s = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]

# Set prior and q0
#prior = nf.distributions.target.NealsFunnel(v1shift = mb, v2shift = 0.)
#q0 = nf.distributions.DiagGaussian(2)
#q0 = nf.distributions.base.MultivariateGaussian()


weight = torch.ones(nc,device='cuda')
vbase = torch.tensor(cb,device='cuda')

#q0 = nf.distributions.base.GMM(weights=weight, mbase=mbase, vbase=vbase, scale=scale,n_cell = nc,trainable = tparam)
print('~~~~~~~~',based)
if based == 'MixtureofGaussians':
    q0 = nf.distributions.base.MixtureofGaussians(n_modes = nc, dim = latent_size, trainable=tparam)
    q1 = nf.distributions.base.MixtureofGaussians(n_modes = nc, dim = latent_size, trainable=tparam)
elif based == 'T':
    q0 = nf.distributions.base.T(n_dim=latent_size, df = cb,trainable = tparam)
    q1 = nf.distributions.base.T(n_dim=latent_size, df = cb,trainable = tparam)
    print([p for p in q0.parameters()])
elif based == 'TMV':
    q0 = nf.distributions.base.TMV(n_dim=latent_size, df = cb,trainable = tparam)
    q1 = nf.distributions.base.TMV(n_dim=latent_size, df = cb,trainable = tparam)

elif based == 'GeneralizedGaussiansDistribution':
    q0 = nf.distributions.base.GeneralizedGaussiansDistribution(n_dim=latent_size, beta = cb,trainable = tparam)
    q1 = nf.distributions.base.GeneralizedGaussiansDistribution(n_dim=latent_size, beta = cb,trainable = tparam)

elif based == 'MultivariateGaussian':
    q0 = nf.distributions.base.MultivariateGaussian(n_dim=latent_size,trainable=tparam)
    q1 = nf.distributions.base.MultivariateGaussian(n_dim=latent_size,trainable=tparam) 
elif based == 'MixtureofGeneralizedGaussians':
    q0 = nf.distributions.base.MixtureofGeneralizedGaussians(n_dim=latent_size,trainable=tparam,n_components = nc,beta=cb)
    q1 = nf.distributions.base.MixtureofGeneralizedGaussians(n_dim=latent_size,trainable=tparam,n_components = nc,beta=cb)
elif based == 'MixtureofT':
    q0 = nf.distributions.base.MixtureofT(n_dim=latent_size,trainable=tparam,n_components = nc,df=cb)
    q1 = nf.distributions.base.MixtureofT(n_dim=latent_size,trainable=tparam,n_components = nc,df=cb)

#wandb.run.name = based
with torch.no_grad():
    sample3,_ = q1.forward(20000)
    print(sample3.shape)
    sample3 = pd.DataFrame(sample3.detach().cpu().numpy())

# Construct flow model

#q0 = nf.distributions.base.DiagGaussian(shape=30)

nfm = nf.NormalizingFlow(q0=q0, flows=flows,categoricals=categorical,catlevels=catlevels,catvdeqs=vquantizers)
nfmBest = nf.NormalizingFlow(q0=q0, flows=flows,categoricals=categorical,catlevels=catlevels,catvdeqs=vquantizers)

c1 = D.Normal(torch.tensor([5.,5.]),torch.tensor([0.9,0.9]))
c2 = D.Normal(torch.tensor([8.,8.]),torch.tensor([0.5,0.5]))

s1 = pd.DataFrame(c1.sample([20000]).detach().cpu().numpy())
s2 = pd.DataFrame(c2.sample([4000]).detach().cpu().numpy())
s1['class'] = 0
s2['class'] = 1
s = s1.append(s2)
#s.to_csv(f'/home/samiri/NNF/logs/target_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
xcol = s.columns
X = s.values


X = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4,shuffle=True)
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

mdsarr = []


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
optimizer2 = torch.optim.Adam(nfm.q0.parameters(), lr=1e-4, weight_decay=1e-6)
sample0,_ = nfm.sample(20000)#.detach().cpu().numpy()
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
nfm.eval()

#wandb.watch(nfm, log_freq=25)
writer = SummaryWriter('./run/' + targetD + '.' + str(prm) +'/'+ based+'.' + str(cb) + str(np.random.rand()))
nfm.train()
for it in tqdm(range(max_iter)):

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
            loss, z, log_q, z_layers, ld_layers = nfm.forward_kld(xt, extended=True)#.to('cuda')
            #print('===================',nfm.q0.loc.squeeze().shape,len(nfm.q0.loc.squeeze().shape))
            if len(nfm.q0.loc.squeeze().shape)>1:
                xs = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
                ys = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
            else:
                xs = nfm.q0.loc.squeeze().detach().cpu().numpy()
                ys = nfm.q0.loc.squeeze().detach().cpu().numpy()

            mds = totaldistancesum(xs, ys, len(xs))
            mdsarr.append(mds)
             
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
        logq.append(log_q.detach().cpu().numpy())

        pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}


        #
        
        #print(nfm.q0.mbase.detach().cpu().item())
        phistg.append([a.grad.detach().cpu().numpy() for a in q0.parameters() if a.grad is not None])
        ############################################################################################################################
        for r in range(3):
            try:
                x = next(train_iter)
                xt = x[0].to('cuda')

            except StopIteration:
                train_iter = iter(train_loader)
                x= next(train_iter)
                xt = x[0].to('cuda')
                    
            optimizer2.zero_grad()

            if annealing:
                #print('!!!!!!',x[0].shape)
                loss, z, log_q, z_layers, ld_layers = nfm.forward_kld(xt, extended=True)#.to('cuda')

                if len(nfm.q0.loc.squeeze().shape)>1:
                    xs = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
                    ys = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
                else:
                    xs = nfm.q0.loc.squeeze().detach().cpu().numpy()
                    ys = nfm.q0.loc.squeeze().detach().cpu().numpy()

                mds = totaldistancesum(xs, ys, len(xs))
                mdsarr.append(mds)
                loss = loss - 200.*torch.tensor(mds)
                
            else:
                loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)
                

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
                    grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
                    gradssteps.append(agrads)
                
                optimizer2.step()
                #print("loc========",nfm.q0.loc)

            #loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            #logq.append(log_q.detach().cpu().numpy())

            #pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}
            #phistg.append([a.grad.detach().cpu().numpy() for a in q0.parameters() if a.grad is not None])

        ############################################################################################################################

        # Plot learned posterior
        if (it + 1) % show_iter == 0:
            try:
                writer.add_scalar('Loss', loss.detach().cpu().numpy(), it)
                writer.add_histogram('z', z.detach().cpu().numpy(), it)
                for _v in range(z.shape[1]):
                    #writer.add_scalar(f'var_{_v}', z[:,_v].detach().cpu().numpy(), it)
                    writer.add_histogram(f'varr_{_v}', z[:,_v].detach().cpu().numpy(), it)
                #for _v in range(len(z_layers))
                writer.add_histogram(f'z_layer{_v}', z_layers[_v], it)
                for _v in range(len(ld_layers)):
                    writer.add_histogram(f'layer_{_v}', ld_layers[_v], it)
                writer.add_scalar('mds', mds, it)
                for _v in pm.keys():
                    writer.add_histogram(f'pm_{_v}', pm[_v], it)
                writer.add_histogram('log_q', log_q.detach().cpu().numpy(), it)

            except Exception as e:
                print(e)
                traceback.print_exc()                                        
            # wandb.log({"loss": loss.detach().cpu().numpy()})
            # #wandb.log({"z": z.detach().cpu().numpy()})
            # #wandb.log({"zz": wandb.Histogram(z.cpu().detach().numpy())})
            # for _v in range(z.shape[1]):
            #     wandb.log({f"var_{_v}": z[:,_v].detach().cpu().numpy()})
            #     wandb.log({f"varr_{_v}": wandb.Histogram(z[:,_v].detach().cpu().numpy())})
            # wandb.log({"log_q": log_q.detach().cpu().numpy()})
            # wandb.log({"z_layers": z_layers})
            # wandb.log({"ld_layers": [l.detach().cpu().numpy() for l in ld_layers]})
            # #wandb.log({"pm": pm})
            # wandb.log({"mds": mds})
            writer.flush()

            wb.append(pm)
            ss,_ = nfmBest.sample(10000)
            gzarr.append(z_layers)
            gzparr.append(ld_layers)
            phist.append([a.detach().cpu().numpy() for a in q0.parameters()])

        #     log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
        #     prob = torch.exp(log_prob)
        #     prob[torch.isnan(prob)] = 0

        #     plt.figure(figsize=(15, 15))
        #     plt.pcolormesh(xx, yy, prob.data.numpy())
        #     plt.contour(xx, yy, prob_prior.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
        #     plt.gca().set_aspect('equal', 'box')
        #     plt.show()
        if loss.to('cpu').data.item()<closs and torch.isnan(nfm.sample(3000)[0]).sum().item() == 0:
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

import seaborn as sns
fig, axes = plt.subplots(1,2, figsize=(70, 20))


sns.scatterplot(X.cpu().detach().numpy()[:,0],X.cpu().detach().numpy()[:,1],ax=axes[0])
sns.scatterplot(sample2.values[:,0],sample2.values[:,1],ax=axes[1])


writer.add_figure('Fig1', plt.gcf())
writer.close()


torch.save(nfmBest, f'/home/samiri/NNF/logs/model_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(X.cpu().detach().numpy(), f'/home/samiri/NNF/logs/targetDataset_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(sample0,f'/home/samiri/NNF/logs/untrainedmodel_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(sample2,f'/home/samiri/NNF/logs/trainedmodel_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(sample3,f'/home/samiri/NNF/logs/untrainedbase__targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(sample4,f'/home/samiri/NNF/logs/trainedbase_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(pd.DataFrame(loss_hist),f'/home/samiri/NNF/logs/losshist_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
#pickle.dump(gzarr, open( f'/home/samiri/NNF/logs/z_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth', 'wb'))
#pickle.dump(gzparr, open( f'/home/samiri/NNF/logs/zp_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth', 'wb'))
#pd.DataFrame(gzarr).to_csv(f'/home/samiri/NNF/logs/z_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(pd.DataFrame(logq),f'/home/samiri/NNF/logs/logq_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
#pd.DataFrame(gzparr).to_csv(f'/home/samiri/NNF/logs/zp_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(phist,f'/home/samiri/NNF/logs/phist_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(phistg,f'/home/samiri/NNF/logs/phistg_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(grads, f'/home/samiri/NNF/logs/grads_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(gradssteps, f'/home/samiri/NNF/logs/gradssteps_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
torch.save(wb, f'/home/samiri/NNF/logs/wb_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')

