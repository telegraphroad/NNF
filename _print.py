import sys
nm,num_layers,lw,arc = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]

from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import pandas as pd
import seaborn as sns
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import gc
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
import seaborn as sns
import pandas as pd
import gc
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
import seaborn as sns
import pandas as pd
import gc

gc.collect()
print(f'=============================================== nm:{nm}, nl:{num_layers}, lw:{lw}, {arc}')
bestmodel = torch.load(f'zbestmodel{arc}_{nm}_{lw}_{num_layers}.ph')
bestmodel.float()
bestmodel.cpu()
bestmodel.eval()

X = pd.read_csv('/home/samiri/PhD/Synth/VCNF/prep.csv')
X = X.drop(['Unnamed: 0'],1)
Y = X.Class
Xc = X.columns
scaler = MinMaxScaler()
scaler.fit(X)
XS = scaler.transform(X)
del X,Y,scaler
gc.collect()
real = pd.DataFrame(XS)
del XS
real.columns = Xc
for c in real.columns:
    if real[c].dtype == 'float64':
        real[c] = real[c].astype('float32')

gc.collect()
fake = bestmodel.sample(len(real))[0].detach().cpu().numpy()
del bestmodel
gc.collect()
fake = pd.DataFrame(fake)
gc.collect()
gc.collect()
fake.columns = Xc
fake.replace([np.inf, -np.inf], np.nan, inplace=True)
fake = fake[np.isfinite(fake).all(1)]
for c in fake.columns:
    if fake[c].dtype == 'float64':
        fake[c] = fake[c].astype('float32')

gc.collect()
plt.figure()
import seaborn as sns
fig, ax = plt.subplots(5,len(fake.columns)//5 + 1,figsize=[25,25])

for i,column in enumerate(np.sort(fake.columns)):
    if column != 'Class':
        #print(column)
        sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,bins=300,color='red')
        sns.histplot(real[column],ax=ax.flatten()[i],kde=True,bins=300,color='green')
    else:
        print(column)
        sns.histplot(fake[column],ax=ax.flatten()[i],kde=True,bins=300,color='red',discrete=True)
        sns.histplot(real[column],ax=ax.flatten()[i],kde=True,bins=300,color='green',discrete=True)
del real,fake
plt.savefig(f'zbestmodel{arc}_{nm}_{lw}_{num_layers}.png')    

