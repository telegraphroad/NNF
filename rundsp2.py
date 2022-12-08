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
import argparse
import seaborn as sns
from normflows import utils
#from torch.utils.tensorboard import SummaryWriter

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

parser = argparse.ArgumentParser(description="Flows",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cb", "--cbase")
parser.add_argument("-nc", "--ncomp")
parser.add_argument("-nu", "--nunit")
parser.add_argument("-b", "--base")
parser.add_argument("-t", "--target")
parser.add_argument("-p", "--param")
#parser.add_argument("-ds", "--dataset")
parser.add_argument("-trainable", "--trainablebase")

args = parser.parse_args()
config = vars(args)
print(config)
cb = float(args.cbase)
prm = float(args.param)
nc = int(args.ncomp)
nu = int(args.nunit)
tparam = bool(int(args.trainablebase))
based = str(args.base)
targetD = str(args.target)

max_iter = 10000
num_samples = 2 ** 8
anneal_iter = 3000
annealing = True
show_iter = 50

K = nu
torch.manual_seed(0)

flows = []

vquantizers = []
# if targetD == 'Adult':
#     latent_size = 15
#     categorical = [1,3,4,5,6,7,8,9,13,14]
#     categorical_qlevels = [9,16,16,7,15,6,5,2,42,2]
#     catlevels = [2]
#     lcm = utils.utils.lcm(categorical_qlevels)
#     vlayers = []
#     b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])
if targetD == 'Adult':
    latent_size = 15
    categorical = []
    categorical_qlevels = []
    catlevels = [2]
    lcm = 0
    vlayers = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

elif targetD == 'Credit':
    latent_size = 30
    vquantizers = []
    categorical = [29]
    categorical_qlevels = [2]
    catlevels = [2]
    lcm = utils.utils.lcm(categorical_qlevels)
    vlayers = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])
elif targetD == 'Covertype':
    latent_size = 55
    vquantizers = []
    categorical = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
       44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
    categorical_qlevels = [2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    catlevels = [2]
    lcm = utils.utils.lcm(categorical_qlevels)
    vlayers = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])
elif targetD == 'Loan':
    latent_size = 11
    vquantizers = []
    categorical = [ 2,  4,  6,  7,  8,  9, 10]
    categorical_qlevels = [4, 3, 2, 2, 2, 2, 2]
    catlevels = [2]
    lcm = utils.utils.lcm(categorical_qlevels)
    vlayers = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

for i in range(4):
    vlayers += [nf.flows.ActNorm(1)]
    s = nf.nets.MLP([1, 4,4, 1], init_zeros=True)
    t = nf.nets.MLP([1, 4,4, 1], init_zeros=True)
    if i % 2 == 0:
        vlayers += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        vlayers += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    vlayers += [nf.flows.ActNorm(1)]

vquantizers = [nf.nets.VariationalDequantization(var_flows=torch.nn.ModuleList(vlayers),quants = categorical_qlevels[i]) for i in range(len(categorical))]

b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])

# for i in range(K):
#     # Neural network with two hidden layers having 64 units each
#     # Last layer is initialized by zeros making training more stable
#     param_map = nf.nets.MLP([latent_size, 64, 64, latent_size], init_zeros=True)
#     # Add flow layer
#     flows.append(nf.flows.AffineCouplingBlock(param_map))
#     # Swap dimensions
#     flows.append(nf.flows.Permute(2, mode='swap'))
    

# for i in range(K):
#     s = nf.nets.MLP([latent_size, 4 * latent_size,4 * latent_size, latent_size], init_zeros=True)
#     t = nf.nets.MLP([latent_size, 4 * latent_size,4 * latent_size, latent_size], init_zeros=True)
#     if i % 2 == 0:
#         flows += [nf.flows.MaskedAffineFlow(b, t, s)]
#     else:
#         flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
#     flows += [nf.flows.ActNorm(latent_size)]


flows = []
for i in range(K):
    s = nf.nets.MLP([latent_size, 30 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 30 * latent_size, latent_size], init_zeros=True)
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


with torch.no_grad():
    sample3,_ = q1.forward(20000)
    print(sample3.shape)
    sample3 = pd.DataFrame(sample3.detach().cpu().numpy())
    torch.save(sample3,f'./logs/untrainedbase__targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
    del sample3

# Construct flow model

#q0 = nf.distributions.base.DiagGaussian(shape=30)

nfm = nf.NormalizingFlow    (q0=q0, flows=flows,categoricals=categorical,catlevels=categorical_qlevels,catvdeqs=vquantizers)
nfmBest = nf.NormalizingFlow(q0=q0, flows=flows,categoricals=categorical,catlevels=categorical_qlevels,catvdeqs=vquantizers)

if targetD == 'Adult':
    X = pd.read_csv('./adult.csv')
    X = X.drop(['Unnamed: 0'],1)
    lx = len(X)
    xcol = X.columns
    # for ii in range(len(categorical)):
    #     X[X.columns[categorical[ii]]] = X[X.columns[categorical[ii]]] * lcm / categorical_qlevels[ii]
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
elif targetD == 'Credit':
    X = pd.read_csv('./prep.csv')
    X = X.drop(['Unnamed: 0'],1)
    xcol = X.columns
    lx = len(X)
    from sklearn.preprocessing import MinMaxScaler
    from collections import Counter
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

elif targetD == 'Covertype':
    X = pd.read_csv('./covertype.csv')
    X = X.drop(['Unnamed: 0'],1)
    xcol = X.columns
    lx = len(X)
    from sklearn.preprocessing import MinMaxScaler
    from collections import Counter
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

elif targetD == 'Loan':
    X = pd.read_csv('./loan.csv').drop(['ZIP Code'],1)
    X = X.drop(['Unnamed: 0'],1)
    xcol = X.columns
    lx = len(X)
    from sklearn.preprocessing import MinMaxScaler
    from collections import Counter
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

X = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4,shuffle=True)
train_iter = iter(train_loader)
torch.save(X.cpu().detach().numpy(), f'./logs/targetDataset_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
fig, axes = plt.subplots(1,2, figsize=(70, 20))
#sns.scatterplot(X.cpu().detach().numpy()[:,0],X.cpu().detach().numpy()[:,1],ax=axes[0])
del X
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
nfm = nfm.to(device)
nfmBest = nfmBest.to(device)
nfm = nfm.double()

nfmBest = nfmBest.double()
mdsarr = []

loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
optimizer2 = torch.optim.Adam(nfm.q0.parameters(), lr=1e-4, weight_decay=1e-6)
sample0,_ = nfm.sample(20000)#.detach().cpu().numpy()
sample0 = pd.DataFrame(sample0.cpu().detach().numpy())
torch.save(sample0,f'./logs/untrainedmodel_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del sample0

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

#writer = SummaryWriter('./run/' + targetD + '.' + str(prm) +'/'+ based+'.' + str(cb) + str(np.random.rand()))

nfm.train()
compiled_nfm = torch.compile(nfm)
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

        if annealing:
            loss, z, log_q, z_layers, ld_layers = compiled_nfm.forward_kld(xt, extended=True)#.to('cuda')
            if len(nfm.q0.loc.squeeze().shape)>1:
                xs = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
                ys = nfm.q0.loc.squeeze().detach().cpu().numpy()[:,0]
            else:
                xs = nfm.q0.loc.squeeze().detach().cpu().numpy()
                ys = nfm.q0.loc.squeeze().detach().cpu().numpy()

            mds = totaldistancesum(xs, ys, len(xs))
            mdsarr.append(mds)

        else:
            loss = compiled_nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward(retain_graph=True)
            with torch.no_grad():
                a = [p.grad for n, p in nfm.named_parameters()]
                asteps = []
                for lctr in range(0,nu):
                    asteps.append([p.grad for n, p in nfm.named_parameters() if (('flows.'+str(lctr)+'.' in n) or ('flows.'+str(lctr+1)+'.' in n))])
                agrads = []
                for lg in asteps:
                    agrads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in lg if i is not None]) if i != 0]))
                grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
                gradssteps.append(agrads)
            
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        logq.append(log_q.detach().cpu().numpy())
        pm = {n:p.detach().cpu().numpy() for n, p in nfm.named_parameters()}
        phistg.append([a.grad.detach().cpu().numpy() for a in q0.parameters() if a.grad is not None])

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
                loss, z, log_q, z_layers, ld_layers = compiled_nfm.forward_kld(xt, extended=True)#.to('cuda')

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
                loss = compiled_nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)
                
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward(retain_graph=True)
                # with torch.no_grad():
                #     a = [p.grad for n, p in nfm.named_parameters()]
                #     asteps = []
                #     for lctr in range(0,nu):
                #         asteps.append([p.grad for n, p in nfm.named_parameters() if (('flows.'+str(lctr)+'.' in n) or ('flows.'+str(lctr+1)+'.' in n))])
                #     agrads = []
                #     for lg in asteps:
                #         agrads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in lg if i is not None]) if i != 0]))
                #     grads.append(np.mean([i for i in np.hstack([i.detach().cpu().numpy().flatten() for i in a if i is not None]) if i != 0]))
                #     gradssteps.append(agrads)
                
                optimizer2.step()

        if (it + 1) % show_iter == 0:
            try:
                writer.add_scalar('Loss', loss.detach().cpu().numpy(), it)
                writer.add_histogram('z', z.detach().cpu().numpy(), it)
                #print('+++++++++++',z.shape,z[:,0].shape)
                for _v in range(z.shape[1]):
                    writer.add_histogram(f'varr_{_v}', z[:,_v].detach().cpu().numpy(), it)
                #print('===========',np.shape(z_layers),np.shape(z_layers[_v]))
                for _v in range(len(z_layers)):
                    writer.add_histogram(f'z_layer{_v}', z_layers[_v], it)
                for _v in range(len(ld_layers)):
                    writer.add_histogram(f'layer_{_v}', ld_layers[_v], it)
                writer.add_scalar('mds', mds, it)
                # for _v in pm.keys():
                #     writer.add_histogram(f'pm_{_v}', pm[_v], it)
                writer.add_histogram('log_q', log_q.detach().cpu().numpy(), it)
            except Exception as e:
                print(e)
                traceback.print_exc()
            writer.flush()

            wb.append(pm)
            ss,_ = nfmBest.sample(10000)
            gzarr.append(z_layers)
            gzparr.append([a.detach().cpu().numpy() for a in ld_layers])
            phist.append([a.detach().cpu().numpy() for a in q0.parameters()])

        if loss.to('cpu').data.item()<closs and torch.isnan(nfm.sample(10000)[0]).sum().item() == 0 and nfm.q0.forward(10000):
            closs = loss.to('cpu').data.item()
            nfmBest.state_dict = deepcopy(nfm.state_dict)
            q1.parameters = deepcopy(q0.parameters)
            #torch.save(nfmBest, f'./logs/model_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')

    except Exception as e:
        print(e)
        traceback.print_exc()                        
        nfm.state_dict = oldm
        q0.parameters = oldp




del nfm
torch.save(pd.DataFrame(loss_hist),f'./logs/losshist_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del loss_hist, loss
torch.save(pd.DataFrame(logq),f'./logs/logq_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del log_q,logq
torch.save(phist,f'./logs/phist_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del phist
torch.save(phistg,f'./logs/phistg_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del phistg
torch.save(grads, f'./logs/grads_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del grads
torch.save(gradssteps, f'./logs/gradssteps_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del gradssteps
torch.save(wb, f'./logs/wb_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del wb


# sample2,_ = nfmBest.sample(lx)
# sample2 = pd.DataFrame(sample2.cpu().detach().numpy())
# torch.save(sample2,f'./logs/trainedmodel_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
# sns.scatterplot(sample2.values[:,0],sample2.values[:,1],ax=axes[1])
# del sample2

sample4,_ = nfmBest.q0.forward(lx)
sample4 = pd.DataFrame(sample4.detach().cpu().numpy())


# writer.add_figure('Fig1', plt.gcf())
writer.close()

torch.save(sample4,f'./logs/trainedbase_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
del sample4
torch.save(nfmBest, f'./logs/model_targetD_{targetD}_nc_{nc}_cb_{cb}_trainable_{tparam}_nunit_{nu}_param_{prm}_base_{based}.pth')
