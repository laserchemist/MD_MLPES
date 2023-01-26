import torch
from torch import Tensor
from torch.nn import Linear, Conv1d, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
import math
#
# freqtorch.py: Neural net vibrational frequency estimator  
# Jonathan Smith, PhD. Oslo 2019
#
# Classes and Functions
#
def molnametolist(text):
# text="CH13OH"
    d=dict.fromkeys(['H','C','O','N','S','P','F','Cl','Br'])
    ch=list(text)
    cfinal=[]
    for i in range(len(ch)):
        mult=1
        # First check for lower case letter after upper
        if i+1<len(ch) and ch[i].isalpha():
            if ch[i+1].islower():
                cfinal.append(ch[i]+ch[i+1])
                # DEBUG print("Two letter symbol: ",ch[i]+ch[i+1])
                i+=1
        if ch[i].isdigit():
            mult=int(ch[i])
            if i+1<len(ch):
                if ch[i+1].isdigit(): # Two digit
                    mult=int(ch[i])*10+int(ch[i+1])
        for j in range(mult):
            if not ch[i-1].isdigit():
                cfinal.append(ch[i-1])
    for c in cfinal: #Need to deal with two letter labels, change to X fornow
        if c in d:
            try:
                d[c] +=1
                ch.append(c)
            except:
                d[c]=1
                ch.append(c)
    mlist=[]
    for key, value in d.items():
        mlist.append(value)
    mlist=[0 if item is None else item for item in mlist] #Need to prevent additions to dictionary
    return mlist
# Read PES data
freq = pd.read_csv("freq.csv")
print("pandas dtype: ",freq.dtypes)
nfreq=freq.to_numpy()
str="CH3OH"
print(freq)
print(nfreq)
X, y, energy = freq['B3LYP631Gd'].to_numpy(), freq['experiment'].to_numpy(), freq['energy'].to_numpy()
moltext=freq['molecule']
print("First molecule name: ",molnametolist(moltext[0]))
print(X.dtype)
print(X)
Xs=X*0.96
X=X/1000
Xs=Xs/1000
y=y/1000
energy=np.log10(-energy)
dim=1
data_size=X.shape[0]
# define the model 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(11, 1320)
        self.fc2 = Linear(1320, 1320)
        self.fc3 = Linear(1320,2640)
        self.fc4 = Linear(2640,640)
        self.cc  = Conv1d(in_channels=16, out_channels=32,kernel_size=1, stride=1, padding=1)
        self.fc5 = Linear(640, 1)
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
#        x = F.relu(self.fc2(x))
#        x=self.fc2(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
#        x=F.dropout(F.relu(self.fc3(x)),p=0.33)
        x= F.relu(self.fc4(x))
#        x=self.fc4(x)
        x=self.fc5(x)
        return x
model = Net()
# define the loss function
#criterion = MSELoss()
criterion=torch.nn.SmoothL1Loss()
def smoothl1loss(x, y):
    if abs(x-y)<1: return 1/2*(x-y)**2
    else: return abs(x-y)-1/2

#criterion = torch.nn.CrossEntropyLoss()
# define the optimizer
loss_start=criterion(Variable(Tensor([X])),Variable(Tensor([y]))).item()
loss_scale=criterion(Variable(Tensor([Xs])),Variable(Tensor([y]))).item()
print('Initial loss: ',loss_start,loss_scale)
#optimizer = SGD(model.parameters(), lr=1e-2)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=True)

#optimizer=Adam(model.parameters(), lr=1e-3)
# define the number of epochs and the data set size
nb_epochs = 6000
X=np.transpose(X)
data_size=y.shape[0]
#y=y.tolist()
y=np.transpose(y)
loss_start=criterion(Variable(Tensor([X])),Variable(Tensor([y]))).item()
pl=[] #List to store results
for epoch in range(nb_epochs):
    #X1, y1 = data_generator(data_size)
    yp=[]
    yexp=[]
    epoch_loss = 0;
    for ix in range(data_size):
        params=molnametolist(moltext[ix][:7])

        params.insert(0,X[ix])
        params.insert(1,energy[ix])
#        print(ix,params)
        y_pred = model(Variable(Tensor([params])))
        yp.append([y[ix],X[ix]])
        yexp.append([y[ix],y_pred.item()])
        loss = criterion(y_pred, Variable(Tensor([y[ix]]), requires_grad=False))
        epoch_loss = loss.data
#        clipping_value = 1#arbitrary number of your choosing
#        torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
    pl.append(epoch_loss.item())
model.eval()
torch.save(model, 'modelfreq.pt')
w1=model.fc1.weight
print(torch.sum(w1))
nw1=w1.detach().numpy()
print("first layer weights: ",nw1)
nwsum=np.sum(nw1**2,axis=0)
print(params)
print(nwsum)
w = list(model.parameters())
test_data = y[0]
#prediction = model(Variable(Tensor([X[0]])))
#print("Prediction: {}".format(prediction.data[0]))
#print("Expected: {}".format(test_data))
# Create a dataframe to create a table
exnum = [random.randrange(0, data_size, 1) for _ in range(25)]

lst=[]
for i in range(15):
    r=exnum[i]
    lst.append([moltext[r],X[r]*1000,yexp[r][1]*1000,yexp[r][0]*1000])

df = pd.DataFrame(lst)
df.columns = ['Molecule', 'B3LYP/6-31G(d)','Prediction (fundamental)', 'Experiment (fundamental)']
df['B3LYP/6-31G(d)'] = df['B3LYP/6-31G(d)'].map('{:,.2f}'.format)
df['Prediction (fundamental)'] = df['Prediction (fundamental)'].map('{:,.2f}'.format)
df['Experiment (fundamental)'] = df['Experiment (fundamental)'].map('{:,.2f}'.format)
d=dict.fromkeys(['H','C','O','N','S','P','F','Cl','Br'])
plabel=[*d]
plabel.insert(0,'Freq.')
plabel.insert(1,'Energy')
print(plabel)
nwsum=nwsum/np.sum(nwsum)
nwlist=nwsum.tolist()
print('nwlist: ',nwlist)

df2=pd.DataFrame([nwlist] ,columns=plabel)
df2 = df2.applymap("{0:.2f}".format)
print(df2)
# Plot the loss as a function of epoch
#fig, ax = plt.subplots()
f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2,figsize=(9,8))
ax1.plot(pl)

ax1.set(xlabel='epoch', ylabel='MSE',
       title='Frequency Neural Net')
ax1.grid()
s = pd.Series(yp)
s*=1000
data = np.array(yp)
x, y = data.T*1000
xs=Xs.T*1000
ax2.scatter(x, y, s=10, c='b', marker="s",label='Experimental to computed'+' MSE: {:.2E}'.format(loss_start))
datexp=np.array(yexp)
xexp,ypre=datexp.T*1000
ax2.scatter(xexp,ypre, s=10, c='r', marker="o",label='Experimental to prediction MSE: '+'{:.2E}'.format(pl[-1]))
ax2.scatter(xs,y, s=10, c='c',marker="P",label="Computation scaled 0.96 MSE: "+'{:.2E}'.format(loss_scale))
ax2.plot([400, 4000], [400, 4000], 'k-', color = 'g')
ax2.legend(loc='upper left',prop={'size': 6})
#ax2.annotate('Scaled 0.96 MSE: '+'{:.2E}'.format(loss_scale), xy=(1, 1), xytext=(3, 4),
#            arrowprops=dict(facecolor='black', shrink=0.05))
ax2.set(xlabel='Original frequency', ylabel='Predicted Frequency')
ax2.set_aspect('equal')
ax2.set_xticks(np.arange(0, 4500, 1000))
ax2.set_yticks(np.arange(0, 4500, 1000))
ax2.grid()
# Table
ax3.axis('off')
ax3.axis('tight')
ax3.table(cellText=df.values, colLabels=df.columns, loc='center')
ax4.axis('off')
t4=ax4.table(cellText=df2.values, colLabels=df2.columns, loc='center')
t4.auto_set_font_size(False)
t4.set_fontsize(6)
#t4.title("Parameter weights",color='b',fontsize=10)
#t4._text.set_color('blue')
f.savefig("freqneural.pdf", format='pdf')
plt.show()
