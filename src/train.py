
# Imports

import statistics
import queue

#     DNN
import torch
from torch.optim.lr_scheduler import LambdaLR



# Learnrate Tools
###
class Adaptive_Overload_Scheduler:
    def __init__(self, optimizer, initial_lr, step_size=10, n_last_losses=10):
        """
        Schedular which works like the adaptive overloading from TCP. 
        The idea is, that the schedular tries to find the ideal learnrate. 
        
        :param optimizer: Optimizer to which the scheduler will be applied
        :param initial_lr: Initial learning rate
        :param decay_factor: Factor by which to multiply the learning rate at each step
        :param step_size: Number of epochs between each learning rate adjustment
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        # self.decay_factor = decay_factor
        self.step_size = step_size
        self.steps = 0
        self.n_last_losses = n_last_losses
        self.last_losses = queue.Queue()
        
        # Set initial learning rate
        self._set_lr(initial_lr)
    
    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def step(self, new_loss):
        """Update the learning rate based on the epoch and decay factor."""
        if self.steps % self.step_size == 0 and self.steps > 0 and not self.last_losses.empty():
            var_lr = statistics.pvariance(list(self.last_losses.queue))
            performance = 0
            for q_loss in list(self.last_losses.queue):
                performance -= q_loss
            performance += list(self.last_losses.queue)[0]


            # check if varianz is too small, then the learning rate is ropably too small
            if var_lr < 0.1:    # FIXME adjust the upper and lower limits
                pass

            # FIXMEhow to use performance and variance?

            new_lr = self.get_lr() * self.decay_factor
            self._set_lr(new_lr)

        if type(new_loss) not in [int, float]:
            try:
                if type(new_loss) == dict:
                    new_loss = sum([value.cpu().detach().numpy() for value in new_loss.values()])
                else:
                    new_loss = float(new_loss)
            except Exception as e:
                raise ValueError(f"Adaptive_Overload_Scheduler could not use loss with type: {type(new_loss)}.")
        
        self.last_losses.put(new_loss)
        self.steps += 1

        # reduce queue if 'full'
        if len(list(self.last_losses.queue)) > self.n_last_losses:
            self.last_losses.get()

    
    



