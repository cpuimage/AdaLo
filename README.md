*Unofficial* *Pytorch* *implementation* of [AdaLo: Adaptive learning rate optimizer with loss for classification](https://www.sciencedirect.com/science/article/abs/pii/S0020025524015214)

**AdaLo: Adaptive Learning Rate Optimizer with Loss for Classification**  
A light-weight, loss-driven gradient optimiser that automatically schedules the learning rate without a scheduler.

## What's AdaLo?

AdaLo (Adaptive Learning Rate with Loss) is a first-order optimiser proposed in [the Information Sciences paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025524015214).  
The key idea is embarrassingly simple:

> **Big loss → big step; small loss → small step.**

Instead of using gradient statistics to tune the learning-rate, AdaLo directly exploits the **current value of the loss**:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\alpha_t=\beta_2\alpha_{t-1}+(1-\beta_2)\frac{\mathcal{L}(\theta_{t-1})}{\kappa}">
</p>

Thus the effective learning rate decreases **automatically** as training converges—no extra scheduler, no hyper-grid search.

## Highlights

- **Scheduler-free** – one less hyper-parameter to tune

- **Low cost** – only one scalar EMA per parameter group

- **First-order only** – no second-moment storage (≈ 3× fewer multiplications than Adam)

- **Wide applicability** – works robustly on CV, NLP and speech benchmarks

## Usage

AdaLo is a drop-in replacement for Adam/SGD:



```python
from adalo import AdaLo

optimizer = AdaLo(model.parameters(),
                  lr=1e-8,        #  (not used for step size; only a lower-bound clamp value for numerical stability)
                  betas=(0.9, 0.999),
                  weight_decay=1e-2,
                  kappa=10.0)      # loss scaling factor

for inputs, labels in dataloader:
    def closure(inp=inputs, lbl=labels):
        optimizer.zero_grad()
        loss = criterion(model(inp), lbl)
        loss.backward()
        return loss
    optimizer.step(closure)
```

That's it—no `LambdaLR`, no `ReduceLROnPlateau`, no cosine tricks.

## Tuning κ (kappa)

κ rescales the loss into a reasonable learning-rate range.  
Rule of thumb from the paper:

| Task                            | Recommendation                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| ImageNet / CIFAR-10 / CIFAR-100 | `kappa = 0.2 ~ 0.3 × initial_loss`                                                                                   |
| Smaller dataset / fine-tuning   | start with `kappa = 3.0` (default) and **increase** if loss curve is noisy, **decrease** if convergence is too slow. |


Training time per epoch is **shorter** than Adam/AdaBelief/diffGrad due to lower computational complexity.



## Citation

If you use this implementation in your research, please cite:

bibtex

```bibtex
@article{AdaLo2024,
  title   = {AdaLo: Adaptive Learning Rate Optimizer with Loss for Classification},
  journal = {Information Sciences},
  year    = {2024},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0020025524015214}
}
```
