
# Imports

import statistics
import queue

#     DNN
import torch
from torch.optim.lr_scheduler import LambdaLR



# Learnrate Tools
###
class Adaptive_Overload_Optimizer:
    def __init__(self, model, learning_rate, momentum, weight_decay, 
                 step_size=100, warm_up_steps=100,
                 n_last_losses=100, n_last_longterm_losses=1000, 
                 should_log=False, log_path="./",
                 lower_loss_border = 0.05,
                 higher_loss_border = 0.5,
                 lower_std_border = 0.05,
                 higher_std_border = 0.5):

        self.init_momentum = momentum
        self.init_weight_decay = weight_decay
        self.init_learning_rate = learning_rate
        self.adam_optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.sgd_optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = self.adam_optimizer
        self.optimizer_name = "Adam"
        self.param_groups = self.optimizer.param_groups

        self.momentum = self.init_momentum
        self.weight_decay = self.init_weight_decay
        self.learning_rate = self.init_learning_rate

        self.warm_up_steps = warm_up_steps
        self.step_size = step_size
        self.steps = 0
        self.n_last_losses = n_last_losses
        self.n_last_longterm_losses = n_last_longterm_losses
        self.last_losses = queue.Queue()
        self.last_longterm_loss = queue.Queue()
        self.last_longterm_std = queue.Queue()

        self.std_performance = None
        self.loss_performance = None

        # update automatically?
        self.lower_loss_border = lower_loss_border
        self.higher_loss_border = higher_loss_border
        self.lower_std_border = lower_std_border
        self.higher_std_border = higher_std_border

        self.should_log = should_log
        self.log_path = log_path

        if self.should_log:
            os.makedirs(self.log_path, exist_ok=True)
            with open(os.path.join(self.log_path, "Adaptive_Overload_Optimizer.txt"), "w") as file:
                log_str = "Adaptive_Overload_Optimizer Logging"
                log_str += f"\n    - init momentum: {self.init_momentum}"
                log_str += f"\n    - init decay: {self.init_weight_decay}"
                log_str += f"\n    - init learning-rate: {self.init_learning_rate}"
                log_str += f"\n    - step-size: {self.step_size}"
                log_str += f"\n    - warm-up-steps: {self.warm_up_steps}"
                log_str += f"\n    - n last losses: {self.n_last_losses}"
                log_str += f"\n    - n last longterm losses: {self.n_last_longterm_losses}"
                file.write(log_str)

    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.param_groups = self.optimizer.param_groups

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def step(self, new_loss):
        if self.steps % self.step_size == 0 and self.steps > self.warm_up_steps and not self.last_losses.empty():
            
            # std fluctuation performance in percentage
            if not self.last_longterm_std.empty():
                std_lr = statistics.pstdev(list(self.last_losses.queue))
                mean_std = statistics.mean(list(self.last_longterm_std.queue))
                std_performance = max(std_lr/mean_std, 1) - std_lr/mean_std
                self.std_performance = std_performance
            else:
                self.std_performance = 1

            # loss direction performance in percentage
            if not self.last_losses.empty():
                mean_learning_rate = statistics.mean(list(self.last_losses.queue))
                mean_longterm_learning_rate = statistics.mean(list(self.last_longterm_loss.queue))
                loss_performance = max(mean_learning_rate/mean_longterm_learning_rate, 1) - (mean_learning_rate/mean_longterm_learning_rate)
                self.loss_performance = loss_performance
            else:
                self.loss_performance = 1

            # adjust optimizer if needed
            if self.loss_performance < self.lower_loss_border and std_performance < self.lower_std_border:
                self.optimizer = self.sgd_optimizer
                self.optimizer_name = "SGD"
            elif self.loss_performance > self.higher_loss_border and std_performance < self.lower_std_border:
                self.optimizer = self.adam_optimizer
                self.optimizer_name = "Adam"

            # adjust learning rate
            # FIXME
            # if std is high and loss is small -> smaller learnrate
            # if std is small and loss is high -> bigger lernrate
            # if std is small and loss is small -> bigger learnrate (local minima)

            new_lr = self.get_lr() 
            self._set_lr(new_lr)

            self.update_log()

        if type(new_loss) not in [int, float]:
            try:
                if type(new_loss) == dict:
                    new_loss = sum([value.cpu().detach().numpy() for value in new_loss.values()])
                else:
                    new_loss = float(new_loss)
            except Exception as e:
                raise ValueError(f"Adaptive_Overload_Scheduler could not use loss with type: {type(new_loss)}.")
        
        # update loss queuess and optimizer step
        self.last_losses.put(new_loss)
        self.last_longterm_loss.put(new_loss)
        if not self.last_losses.empty():
                std_lr = statistics.pstdev(list(self.last_losses.queue))
                self.last_longterm_std.put(std_lr)
        self.steps += 1

        # reduce queue if 'full'
        if len(list(self.last_losses.queue)) > self.n_last_losses:
            self.last_losses.get()

        if len(list(self.last_longterm_std.queue)) > self.n_last_longterm_losses:
            self.last_longterm_std.get()

        if len(list(self.last_longterm_loss.queue)) > self.n_last_longterm_losses:
            self.last_longterm_loss.get()

        # make optimizer step
        self.optimizer.step()

    def update_log(self):
        log_str = f"\n\n{'-'*50}\nStep: {self.steps}"
        log_str += f"\n    - momentum: {self.momentum}"
        log_str += f"\n    - decay: {self.weight_decay}"
        log_str += f"\n    - learning-rate: {self.learning_rate}"
        log_str += f"\n    - optimizer: {self.optimizer_name}"
        log_str += f"\n    - std base: {None if self.last_longterm_std.empty() else statistics.mean(list(self.last_longterm_std.queue))}"
        log_str += f"\n    - loss base: {None if self.last_longterm_loss.empty() else statistics.mean(list(self.last_longterm_loss.queue))}"
        log_str += f"\n    - std performance: {self.std_performance}"
        log_str += f"\n    - loss performance: {self.loss_performance}"

        self.log(log_str)

    def log(self, content):
        if self.should_log:
            with open(os.path.join(self.log_path, "Adaptive_Overload_Optimizer.txt"), "a") as file:
                file.write(f"\n{content}")
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    
    



