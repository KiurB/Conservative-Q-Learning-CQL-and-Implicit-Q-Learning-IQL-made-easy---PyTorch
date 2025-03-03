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

    def forward(self,x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        return F.sigmoid(self.lin_mu(x))

class Q(nn.Module):

    def __init__(self,n_obs,n_act,n_hidden):
        super(Q,self).__init__()
        self.lin1 = nn.Linear(n_obs+n_act,n_hidden*2)
        self.lin2 = nn.Linear(n_hidden*2,n_hidden)
        self.lin3 = nn.Linear(n_hidden,1)

    def forward(self,x):
        x = F.silu(self.lin1(x))
        x = F.silu(self.lin2(x))
        x = self.lin3(x)
        return F.tanh(x)*100


class TrainingCQL():

    def __init__(self,
                 GAMMA=0.99,
                 TAU=0.01,
                 ALPHA=3.,
                 LR_C=1e-4,
                 LR_A=1e-4,
                 n_act=7,
                 n_obs=81,
                 n_hidden=1024):

        self.GAMMA = GAMMA
        self.TAU = TAU
        self.ALPHA = ALPHA
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

        self.optim_act = optim.AdamW(self.actor.parameters(),
                                     lr=LR_A,
                                     amsgrad=True)
        self.optim_q1 = optim.AdamW(self.q1.parameters(),
                                    lr=LR_C,
                                    amsgrad=True)
        self.optim_q2 = optim.AdamW(self.q2.parameters(),
                                    lr=LR_C,
                                    amsgrad=True)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def optimize_model(self,batch,i,skip,sample_size):

        #update the Q function... this will take a bit.
        sa_couple = torch.cat((batch["state"],
                               batch["action"]),dim=1)
        noise = torch.randn_like(batch["action"],device=self.device)*0.01
        sa_noise = torch.cat((batch["state"],
                              torch.clamp(batch["action"]+noise,min=0.,max=1.)),
                              dim=1)
        
        with torch.no_grad():
            #for BE loss
            sa_n_couple = torch.cat((batch["n_state"],
                                     self.actor(batch["n_state"])),dim=1)
            q1_t = self.q1_target(sa_n_couple)
            q2_t = self.q2_target(sa_n_couple)
            q_t = torch.min(q1_t,q2_t)
            print(q_t)
            target = batch["reward"]+self.GAMMA*q_t
            #for CQL loss
            sa_on_couple = torch.cat((batch["state"],
                                      self.actor(batch["state"])),dim=1)
        z = 5000
        flag = (i%2 != 0) #i can probably use a mask here ngl
        if flag:
            q_up = self.q1(sa_couple)
            q_unif = self.q1(sa_on_couple)
            q_unif *= torch.exp(q_unif)/z
            q_samp = self.q1(sa_noise)
            old_dict = self.q1_target.state_dict()
        else:
            q_up = self.q2(sa_couple)
            q_unif = self.q2(sa_on_couple)
            q_unif *= torch.exp(q_unif)/z
            q_samp = self.q2(sa_noise)
            old_dict = self.q2_target.state_dict()
     
        #first loss (Bellman error)
        loss_b = F.mse_loss(target,q_up)/2

        #CQL loss
        mean_unif = torch.mean(q_unif,dim=0)
        loss_cql = mean_unif-q_samp
        loss_cql = self.ALPHA*torch.mean(loss_cql)

        #evaluation
        loss_q = loss_b+loss_cql
        if flag:
            self.optim_q1.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(),1)
            self.optim_q1.step()

            #soft update
            new_dict = self.q1.state_dict()
            for key in new_dict:
                new_dict[key]=(1-self.TAU)*old_dict[key].clone()+\
                                  self.TAU*new_dict[key].clone()
            self.q1_target.load_state_dict(new_dict)

        else:
            self.optim_q2.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(),1)
            self.optim_q2.step()

            #soft update
            new_dict = self.q2.state_dict()
            for key in new_dict:
                new_dict[key]=(1-self.TAU)*old_dict[key].clone()+\
                                  self.TAU*new_dict[key].clone()
            self.q2_target.load_state_dict(new_dict)

        #policy update
        if i%skip == 0:

            act = self.actor(batch["state"])
            acst_c = torch.cat((batch["state"],act),dim=1)
            q_c = torch.min(self.q1(acst_c),self.q2(acst_c))

            loss = torch.mean(-q_c)
            self.optim_act.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1)
            self.optim_act.step()

    
    def training_loop(self, dataset, n, batch_size, skip, sample_size):

        loader = D.DataLoader(dataset,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=0)
        for i in range(n):
            for batch in loader:
                self.optimize_model(batch,i,skip,sample_size)
            print(i)

        

        

        



