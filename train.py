import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from diffusion_utilities import *
from model import *


# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise, ab_t):
  return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

   
def forward_diffusion(dataloader, nn_model, n_epoch, lrate, timesteps, save_dir, ab_t, device):
  # set into train mode
  nn_model.train()
  # initialize optimizer
  optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

  for ep in range(n_epoch):
      print(f'epoch {ep}')
    
      # linearly decay learning rate
      optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

      for x, c in dataloader:   # x: images  c: context
          optim.zero_grad()
          x = x.to(device)
          c = c.to(x)
        
          # randomly mask out c
          '''
          For some random noises, we completely mask out the contexts so that the
          model is able to learn general features without depending on the context.
          '''
          context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
          c = c * context_mask.unsqueeze(-1)
        
          # sample random Gaussian Noise
          noise = torch.randn_like(x)
          t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
          x_pert = perturb_input(x, t, noise, ab_t)
        
          # use network to recover noise
          # we add context when we call the neural network
          pred_noise = nn_model(x_pert, t / timesteps, c=c)
        
          # loss is mean squared error between the predicted and true noise
          loss = F.mse_loss(pred_noise, noise)
          loss.backward()
        
          optim.step()

      # save model periodically
      if ep%4==0 or ep == int(n_epoch-1):
          if not os.path.exists(save_dir):
              os.mkdir(save_dir)
          torch.save(nn_model.state_dict(), save_dir + f"context_model_{ep}.pth")
          # print('saved model at ' + save_dir + f"context_model_{ep}.pth")