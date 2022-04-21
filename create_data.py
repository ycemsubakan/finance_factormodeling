import torch
import os
import numpy.random as numpy_rand
import math
import pandas as pd
import matplotlib.pyplot as plt

## create data

for M in range(1, 2): 
    path='./Simu'; 
    
    name1='SimuData_p50'; # Case Pc=50 
    name2='SimuData_p100'; # Case Pc=100
    # os.makedirs(path, exist_ok=True);
    os.makedirs(os.path.join(path,name1), exist_ok=True);
    os.makedirs(os.path.join(path,name2), exist_ok=True);

    # %% Case Pc=100 %%%
    # cem: it seems N needs to be larger than T, otherwise the code breaks
    N=25    # 200
    m=20
    T=20    # 180
    stdv=0.05
    theta_w=0.02
    stde=0.05
    
    rho=(torch.rand((m, 1))/10) + 0.9  
    c=torch.zeros(N*T, m);
    for i in range(m):
        x=torch.zeros(N,T);
        x[:, 0]=torch.randn((N,)) 
        for t in range(1, T):
            x[:, t]=rho[i]*x[:,t-1] + torch.randn((N, ))*torch.sqrt(1-rho[i]**2)
        
        _, r=torch.sort(x);
        szx=x.shape;
        x1=torch.zeros(szx);
        ridx= torch.arange(0, szx[0])    # 1:szx(1);
        for k in range(szx[1]):
            x1[r[:,k],k]=ridx*2/(N+1)-1;
        
        c[:,i]=x1.reshape(-1)
    
    # per = torch.arange(0, N).unsqueeze(1).repeat(1, T)  # repmat(1:N,1,T);
    time = torch.repeat_interleave(torch.arange(0, T), N)               # repelem(1:T ,N);
    vt = torch.randn((3, T))*stdv

    # vt=normrnd(0,1,[3,T])*stdv;
    # beta=c(:,[1,2,3]);
    beta = c[:, [0, 1, 2]]
    betav = torch.zeros(N*T);
    for t in range(0, T):  # t=1:T

        ind = (time==t);
        betav[ind]= torch.matmul(beta[ind,:], vt[:,t])
    
    y=torch.zeros((T,));
    y[0]=torch.randn(1)
    q=0.95;
    for t in range(1, T): # t=2:T
        y[t]=q*y[t-1]+torch.randn(1)*math.sqrt(1-q**2);
        
    cy=c
    for t in range(T):    # 1:T
        ind=(time==t)
        cy[ind,:]=c[ind,:]*y[t]

    ep=torch.from_numpy(numpy_rand.standard_t(5,[N*T,1])*stde)
     
    # %%% Model 1
    theta=[[1, 1], [0]*(m-2),[0,0,1], [0]*(m-3)] #*theta_w;
    combined = []
    for lst in theta:
        combined = combined + lst
    theta = torch.tensor(combined).float()
    cat = torch.cat([c, cy], dim=1)     
    
    rt = torch.matmul(cat, theta.unsqueeze(1))
    r1 = rt + betav.unsqueeze(1) + ep

    pathc = os.path.join(path, name2, 'c{}.csv'.format(M)) 

    df_cat = pd.DataFrame(cat)
    df_cat.to_csv(pathc)

    pathr = os.path.join(path, name2, 'r1_{}.csv'.format(M))
    df_r1 = pd.DataFrame(r1)
    df_r1.to_csv(pathr)

    # plot characteristics
    plt.figure(figsize=[30, 5], dpi=100)
    for i in range(1,5):
        plt.subplot(5,1,i)
        plt.plot(cat[:,i])
        plt.grid()
        plt.title('Characteristic {}'.format(i))

    # plot returns
    plt.subplot(5,1,5)

    #plt.subplots_adjust(bottom=0.4, top=0.5)
    plt.tight_layout()
    plt.plot(r1[:,0],'r')
    plt.grid()
    plt.title('Return')
    plt.savefig('data.png')

    #  %%% Model 2
    # theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    z = torch.cat([c, cy], dim=1)    #  horzcat(c,cy);
    z[:,0]=2 * c[:,0]**2;
    z[:,1]=1.5 * c[:,0]*c[:,1]*1.5;
    z[:,m+2]=torch.sign(cy[:,2])*0.6;
    
    r1 = torch.matmul(z, theta.unsqueeze(1)) + betav + ep  #z*theta'+betav+ep;
    rt = torch.matmul(z, theta.unsqueeze(1)) 
    # %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));

    pathr = os.path.join(path, name2, 'r2_{}.csv'.format(M))
    df_r1 = pd.DataFrame(r1)
    df_r1.to_csv(pathr)
    

    # # %%% Case Pc=50 %%%
    # cem : This is the same thing but with 50 characteristics 
    # m = 50
    # 
    # # %%% MOdel 1
    # 
    # theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    # r1=horzcat(c(:,1:m),cy(:,1:m))*theta'+betav+ep;
    # rt=horzcat(c(:,1:m),cy(:,1:m))*theta';
    # %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    # 
    # pathc=sprintf('%s',path,name1);
    # pathc=sprintf('%s',pathc,'/c');
    # pathc=sprintf('%s%d',pathc,M);
    # pathc=sprintf('%s',pathc,'.csv');
    # csvwrite(pathc,horzcat(c(:,1:m),cy(:,1:m)));
    # 
    # pathr=sprintf('%s',path,name1);
    # pathr=sprintf('%s',pathr,'/r1');
    # pathr=sprintf('%s_%d',pathr,M);
    # pathr=sprintf('%s',pathr,'.csv');
    # csvwrite(pathr,r1);
    # 
   
    # %%% Model 2
    # 
    # theta=[1,1,repelem(0,m-2),0,0,1,repelem(0,m-3)]*theta_w;
    # z=horzcat(c(:,1:m),cy(:,1:m));
    # z(:,1)=c(:,1).^2*2;
    # z(:,2)=c(:,1).*c(:,2)*1.5;
    # z(:,m+3)=sign(cy(:,3))*0.6;
    # 
    # r1=z*theta'+betav+ep;
    # rt=z*theta';
    # %disp(1-sum((r1-rt).^2)/sum((r1-mean(r1)).^2));
    # 
    # pathr=sprintf('%s',path,name1);
    # pathr=sprintf('%s',pathr,'/r2');
    # pathr=sprintf('%s_%d',pathr,M);
    # pathr=sprintf('%s',pathr,'.csv');
    # csvwrite(pathr,r1);
