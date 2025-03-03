#Author: KiurB
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim
import torch.distributions as dists
from memory import StaticData

class Actor(nn.Module):

    def __init__(self,n_obs,n_act,n_hidden):
        super(Actor,self).__init__()
        self.lin1 = nn.Linear(n_obs,n_hidden*2)
        self.lin2 = nn.Linear(n_hidden*2,n_hidden)
        self.lin_mu = nn.Linear(n_hidden,n_act)
        self.lin_lsig = nn.Linear(n_hidden,n_act)

    def forward(self,x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        sigma = -F.relu(self.lin_lsig(x))
        torch.clamp_(sigma,min=-10.)
        return F.sigmoid(self.lin_mu(x)),torch.exp(sigma)

class Q(nn.Module):

    def __init__(self,n_obs,n_act,n_hidden):
        super(Q,self).__init__()
        self.lin1 = nn.Linear(n_obs+n_act,n_hidden*2)
        self.lin2 = nn.Linear(n_hidden*2,n_hidden)
        self.lin3 = nn.Linear(n_hidden,1)

    def forward(self,x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        return self.lin3(x)

class V(nn.Module):

    def __init__(self,n_obs,n_hidden):
        super(V,self).__init__()
        self.lin1 = nn.Linear(n_obs,n_hidden*2)
        self.lin2 = nn.Linear(n_hidden*2,n_hidden)
        self.lin3 = nn.Linear(n_hidden,1)

    def forward(self,x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        return self.lin3(x)
    

class TrainingIQL():

    def __init__(self,
                 GAMMA=0.95,
                 TAU=0.01,
                 RHO=0.9,
                 BETA=3.,
                 LR_C=3e-4,
                 LR_A=3e-4,
                 n_act=7,
                 n_obs=81,
                 n_hidden=512):
        
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.RHO = RHO
        self.BETA = BETA
        self.LR_C = LR_C
        self.LR_A = LR_A
        self.n_act = n_act
        self.n_obs = n_obs
        self.n_hidden = n_hidden

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )

        self.actor = Actor(n_obs,n_act,n_hidden).to(self.device)
        self.q1 = Q(n_obs,n_act,n_hidden).to(self.device)
        self.q2 = Q(n_obs,n_act,n_hidden).to(self.device)
        self.q1_target = Q(n_obs,n_act,n_hidden).to(self.device)
        self.q2_target = Q(n_obs,n_act,n_hidden).to(self.device)
        self.v = V(n_obs,n_hidden).to(self.device)

        self.optim_act = optim.AdamW(self.actor.parameters(),
                                     lr=LR_A,
                                     amsgrad=True)
        self.optim_q1 = optim.AdamW(self.q1.parameters(),
                                    lr=LR_C,
                                    amsgrad=True)
        self.optim_q2 = optim.AdamW(self.q2.parameters(),
                                    lr=LR_C,
                                    amsgrad=True)
        self.optim_v = optim.AdamW(self.v.parameters(),
                                   lr=LR_C,
                                   amsgrad=True)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def optimize_model(self,batch,i,skip):

        #update the V function    
        sa_couple = torch.cat((batch["state"],
                                   batch["action"]),dim=1)
        with torch.no_grad():
            q1_t = self.q1_target(sa_couple)
            q2_t = self.q2_target(sa_couple)
            q_t = torch.min(q1_t,q2_t)
        value = self.v(batch["state"])
        
        u = q_t-value
        mask = u<0        
        loss_v1 = abs(self.RHO-1)*(u**2)
        loss_v2 = abs(self.RHO-0)*(u**2)
        loss_v = torch.where(mask,loss_v1,loss_v2)
        loss_v = torch.mean(loss_v)
        self.optim_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(),1)
        self.optim_v.step()

        #update the Q function

        #prepare soft update
        old_q1_dict = self.q1_target.state_dict()
        old_q2_dict = self.q2_target.state_dict()

        #q update
        with torch.no_grad():
            v_t = self.v(batch["n_state"])
            target = batch["reward"]+self.GAMMA*v_t    
        if i%2 != 0:
            loss_q1 = F.mse_loss(self.q1(sa_couple),target)
            self.optim_q1.zero_grad()
            loss_q1.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(),1)
            self.optim_q1.step()
        else:
            loss_q2 = F.mse_loss(self.q2(sa_couple),target)
            self.optim_q2.zero_grad()
            loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(),1)
            self.optim_q2.step()

        #soft update
        new_q1_dict = self.q1.state_dict()
        new_q2_dict = self.q2.state_dict()
        for key in new_q1_dict:
            new_q1_dict[key]=(1-self.TAU)*old_q1_dict[key].clone()+\
                self.TAU*new_q1_dict[key].clone()
            new_q2_dict[key]=(1-self.TAU)*old_q2_dict[key].clone()+\
                self.TAU*new_q2_dict[key].clone()
        self.q1_target.load_state_dict(new_q1_dict)
        self.q2_target.load_state_dict(new_q2_dict)

        #update policy

        if i%skip == 0:
            with torch.no_grad():
                q1_t = self.q1_target(sa_couple)
                q2_t = self.q2_target(sa_couple)
                q_t = torch.min(q1_t,q2_t)
                value = self.v(batch["state"])
            mu,sigma = self.actor(batch["state"])
            mn = dists.MultivariateNormal(mu,scale_tril=torch.diag_embed(sigma,
                                                                         offset=0,
                                                                         dim1=-2,
                                                                         dim2=-1))
            logp = mn.log_prob(batch["action"]) 
            loss_a = -torch.exp(self.BETA*(q_t-value))*logp
            loss_a = torch.mean(loss_a)
            self.optim_act.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1)
            self.optim_act.step()

    def training_loop(self, dataset, n, batch_size, skip):

        loader = D.DataLoader(dataset,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)

        for i in range(n):
            for batch in loader:
                self.optimize_model(batch,i,skip)
            print(i)



