import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, ".")

from settings import epsilon, alpha, pgd_alpha, pgd_attack_iters

from src.data.preprocess import lower_limit, upper_limit, tensor_std
from src.utils.ct import ctx_noparamgrad_and_eval


__all__ = ["fgsm", "pgd"]


epsilon = (epsilon / 255.0) / tensor_std
alpha = (alpha / 255.0) / tensor_std
pgd_alpha = (pgd_alpha / 255.0) / tensor_std


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def fgsm(model, X, y):

    with ctx_noparamgrad_and_eval(model):
        delta = torch.zeros_like(X).cuda()
        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(
                -epsilon[j][0][0].item(), epsilon[j][0][0].item()
            )
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True

        output, _ = model(X + delta[: X.size(0)])
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[: X.size(0)] = clamp(
            delta[: X.size(0)], lower_limit - X, upper_limit - X
        )
        delta = delta.detach()

    return X + delta[: X.size(0)]


def pgd(model, X, y):

    with torch.enable_grad():
        with ctx_noparamgrad_and_eval(model):
            delta = torch.zeros_like(X).cuda()
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(
                    -epsilon[i][0][0].item(), epsilon[i][0][0].item()
                )
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            for _ in range(pgd_attack_iters):
                output, _ = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
                loss.backward()

                grad = delta.grad.detach()
                d = delta[index[0], :, :, :]
                g = grad[index[0], :, :, :]
                d = clamp(d + pgd_alpha * torch.sign(g), -epsilon, epsilon)
                d = clamp(
                    d,
                    lower_limit - X[index[0], :, :, :],
                    upper_limit - X[index[0], :, :, :],
                )
                delta.data[index[0], :, :, :] = d
                delta.grad.zero_()

    delta = delta.detach()
    return X + delta
