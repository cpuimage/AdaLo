# mypy: allow-untyped-defs
from typing import Tuple, Union

import torch


class AdaLo(torch.optim.Optimizer):
    r"""
       AdaLo: Adaptive Learning Rate Optimizer with Loss for Classification
       paper: https://www.sciencedirect.com/science/article/abs/pii/S0020025524015214
       code:  https://github.com/cpuimage/AdaLo

       usage:
           for inputs, labels in dataloader:
               def closure(inp=inputs, lbl=labels):
                   optimizer.zero_grad()
                   loss = criterion(model(inp), lbl)
                   loss.backward()
                   return loss
               optimizer.step(closure)

       Args:
           params: Iterable of parameters to optimize or dicts defining
            parameter groups.
           lr: (not used for step size; only a lower-bound clamp value for numerical stability)
           betas: (beta1, beta2) coefficients for gradient momentum and loss-EMA smoothing respectively
           weight_decay: L2 weight decay
           kappa: loss scaling factor
       """

    def __init__(self,
                 params,
                 lr: Union[float, torch.Tensor] = 1e-8,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-2,
                 kappa: float = 10.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, kappa=kappa)
        super(AdaLo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            min_lr = group['lr']
            weight_decay = group['weight_decay']
            kappa = group['kappa']
            for p in group['params']:
                if p.grad.is_sparse:
                    raise RuntimeError("current optimizer does not support sparse gradients")
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['loss_ema'] = 0.0
                state['step'] += 1
                m = state['m']
                loss_ema = state['loss_ema']
                if weight_decay != 0:
                    p.grad = p.grad.add(p.data, alpha=weight_decay)
                m.lerp_(p.grad, 1.0 - beta1)
                if loss is not None:
                    loss_value = loss.item()
                    if loss_value > 0:
                        scaled_loss = torch.log1p(torch.tensor(loss_value))
                    else:
                        scaled_loss = loss
                    loss_val = (torch.tanh(-scaled_loss * 0.5).item() + 1.0) * 0.5
                    loss_ema = beta2 * loss_ema + (1.0 - beta2) * loss_val
                    state['loss_ema'] = loss_ema
                lr_t = max(min_lr, loss_ema / kappa)
                p.data.add_(m, alpha=-lr_t)
        return loss