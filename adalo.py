# mypy: allow-untyped-defs
from typing import Tuple, Union

import torch
from torch import GradScaler


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
           mode: control learning rate adaptation mode ('adversarial' or 'compliant')
                 'adversarial': decrease learning rate when loss increases (conservative strategy)
                 'compliant': increase learning rate when loss increases (aggressive strategy)
       """

    def __init__(self,
                 params,
                 lr: Union[float, torch.Tensor] = 1e-8,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-2,
                 kappa: float = 3.0,
                 mode: str = 'adversarial'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(betas[0]))
        if betas[1] < 0.0 or betas[1] >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], weight_decay=weight_decay, kappa=kappa,
                        mode=mode)
        super(AdaLo, self).__init__(params, defaults)

    def step(self, closure=None, scaler: GradScaler = None, loss=None):
        already_updated_by_scaler = False
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self)
                scaler.step(self, loss=loss)
                scaler.update()
                already_updated_by_scaler = True
        if not already_updated_by_scaler:
            for group in self.param_groups:
                beta1 = group['beta1']
                beta2 = group['beta2']
                min_lr = group['lr']
                weight_decay = group['weight_decay']
                kappa = group['kappa']
                mode = group['mode']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError("current optimizer does not support sparse gradients")
                    state = self.state[p]
                    if len(state) == 0:
                        state['m'] = torch.zeros_like(p.data)
                        state['loss_ema'] = torch.tensor(0.0, device=p.device, dtype=p.dtype)
                    m = state['m']
                    loss_ema = state['loss_ema']
                    m.lerp_(p.grad, 1.0 - beta1)
                    if loss is not None:
                        loss_value = loss.detach()
                        if loss_value > 0:
                            scaled_loss = torch.log1p(loss_value)
                        else:
                            scaled_loss = loss_value
                        transformed_loss = (torch.tanh(-scaled_loss * 0.5) + 1.0) * 0.5
                        loss_ema.lerp_(transformed_loss, 1.0 - beta2)
                    if mode == 'adversarial':
                        lr_t = loss_ema.div(kappa).clamp_min_(min_lr)
                    else:
                        lr_t = (1.0 - loss_ema).div(kappa).clamp_min_(min_lr)
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr_t * weight_decay)
                    p.data.sub_(m * lr_t)
        return loss
