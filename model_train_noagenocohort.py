#importing the libraries

# Test model performanceimport torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import warnings
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.stats as ss
import torch
from progressbar import ProgressBar as pb
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
# Working directory

os.chdir("C:/Arbeit/Paper/Alfonso_Steffe_ML/Otterbach")

# Load data
lfsat=pd.read_stata('lfsat_V12_2019-02-13c.dta').dropna()

# Clean up

cleanup_nums = {"lfsat": {"1.0": 1, "2.0": 2, "3.0":3, "4.0":4, "5.0":5, "6.0":6, "7.0":7, "8.0":8, "9.0":9, "[0] Completely dissatisfied    0": 0,"[10] Completely satisfied    10": 10 }}
lfsat=lfsat.replace(cleanup_nums)
# Exclude columns

l=["nkids","educ","lfsat","behinderung","age","unempl","nilf","eigenheim",
   "care","married","female","time","mode","ghealth","bhealth","rhhinc","hhincsat_n"]

for i in range(1,19):
    l+=["dagecat" + str(i)]
    
lfsat_pe=lfsat.loc[:,l]
lfsat_pe=lfsat_pe.drop(["bhealth"],axis=1)
lfsat_pe=lfsat_pe.drop(["dagecat1"],axis=1)
lfsat_pe=lfsat_pe.drop(["dagecat2"],axis=1)
lfsat_pe=lfsat_pe[lfsat_pe.age>=20]
lfsat_pe=lfsat_pe[lfsat_pe.age<=70]

lfsat_pe=lfsat_pe[lfsat_pe.behinderung!=300]


# Shuffle data set
lfsat_pe=lfsat_pe.sample(frac=1, random_state=100)

# Split in Training and Test data
random.seed(1)
train, test=train_test_split(lfsat_pe,test_size=0.2, random_state=10)

dl=l[19:]
dl.append("lfsat")
dl.append("age")

x_train=train.drop(dl,axis=1).to_numpy().astype(np.float64)
y_train=train.lfsat.to_numpy().astype(np.float64)

x_test=test.drop(dl,axis=1).to_numpy().astype(np.float64)
y_test=test.lfsat.to_numpy().astype(np.float64)

x_train_s=np.array(ss.zscore(x_train)).astype(np.float64)
x_test_s=np.array(ss.zscore(x_test)).astype(np.float64)

#test.groupby("age").lfsat.mean().plot()

#dataset
from torch.utils.data import Dataset, DataLoader
class prep(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]

  def __len__(self):
    return self.length



data_train = prep(x_train_s,y_train)
data_test=prep(x_test_s,y_test)


batch_size = 1
# n_iters = int(x_train.shape[0]/batch_size)
num_epochs = 20

#dataloader
dataloader_tr = DataLoader(dataset=data_train,shuffle=True,batch_size=batch_size)
dataloader_te = DataLoader(dataset=data_test,shuffle=True,batch_size=batch_size)



hid0=14
hid1=31
hid2=31
hid3=0
hid4=0
hid5=0
hid6=0
hid7=0
drout1=0.8



# Define model
net=nn.Sequential( #sequential operation
                  
            nn.Linear(hid0, hid1),
            nn.LeakyReLU(), 
            
            #nn.Dropout(drout1),
            nn.Linear(hid1, hid2), 
            #nn.Dropout(drout1),
            nn.LeakyReLU(),
            
           # nn.Linear(hid2, hid3), 
            #nn.LeakyReLU(), 
            
            #nn.Dropout(drout1),
            #nn.Linear(hid3, hid4), 
            #nn.LeakyReLU(),
            
           # nn.Linear(hid3, hid4), 
            #nn.LeakyReLU(),
            #nn.Linear(hid4,hid5),
            #nn.LeakyReLU(),
            #nn.Linear(hid5,hid6),
            #nn.LeakyReLU(),
            #nn.Linear(hid6,hid7),
            #nn.LeakyReLU(),
            nn.Linear(hid2,1),
            nn.ReLU())

# initalize random weights
    ## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)
        
        
net.apply(weights_init_normal)


# Initalize model
class lf_nn(nn.Module):
    def __init__(self, hid0, hid1, hid2, hid3, hid4, hid5, drout,net):
        super().__init__()
        torch.manual_seed(0)
        self.net = net

    def forward(self, X):
        return self.net(X)



model = lf_nn(hid0,hid1,hid2,hid3,hid4,hid5,drout1,net)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)



pbar= pb()

costval = [[],[]]
iter=0
for epoch in pbar(range(num_epochs)):
    for i, (x_tr, y_tr) in enumerate(dataloader_tr):

        
        x_tr = x_tr.to(device)
        y_tr = y_tr.to(device)       


        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        
        # Forward pass to get output/logits
        outputs = model(x_tr)
        

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, y_tr)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1
        if iter%10000 == 0:
            print('Epoch: {}. Iteration: {}. Loss: {}.'.format(epoch, iter, loss))
            costval[0].append(loss)
            costval[1].append(epoch)
    



#Example for saving a checkpoint assuming the network class
checkpoint = {'model': lf_nn(hid0,hid1,hid2,hid3,hid4,hid5,drout1,net),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'C:/Arbeit/Paper/Alfonso_Steffe_ML/Otterbach/checkpoint_noagenocohort.pth')




cost=[]
for i in range(len(costval[0])):
               cost.append(costval[0][i].cpu().detach().numpy())

cost=np.array(cost)
train_error=cost.mean()

y_hat=model.forward(data_test.x.to(device)).cpu().detach().numpy()


error=np.subtract(y_test,y_hat[:,0])**2
test_error=error.mean()


final=test
final["pre"]=y_hat

final.groupby("age").pre.mean().plot()
final.groupby("age").lfsat.mean().plot(label="True values")
plt.ylabel("life Satisfaction")
plt.legend(("predicted values", "actual values"))


years=["1996","2001","2006","2011","2016"]

final.groupby("time").pre.mean().plot()
final.groupby("time").lfsat.mean().plot(label="True values")
plt.ylabel("life Satisfaction")
plt.legend(("predicted values", "actual values"))
plt.xticks([5,10,15,20,25],years)

cohorts=[]
for i in range(3,19):
    cohorts+=["dagecat" + str(i)]
    
final["cohort"]=0

k=3
for i in cohorts:
    final.loc[final[i]==1, "cohort"]=k
    k+=1

final.groupby("cohort").pre.mean().plot()
final.groupby("cohort").lfsat.mean().plot(label="True values")
plt.ylabel("life Satisfaction")
plt.legend(("predicted values", "actual values"))



final["difference"]=final.lfsat-final.pre
final.groupby("age").difference.mean().plot()


co=pd.DataFrame(cost,columns=["loss"])
co["epoch"]=costval[1]
co.groupby("epoch").loss.mean().plot()

print('Epochs during training: {}. Average training MSE: {}. Test MSE: {}.'.format(epoch+1,train_error,test_error))











