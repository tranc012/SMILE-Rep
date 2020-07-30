import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
#         self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
#         self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
#         self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
#         self.l1 = nn.Linear(512*20*20, 64)

#         self.b1 = nn.BatchNorm2d(128)
#         self.b2 = nn.BatchNorm2d(256)
#         self.b3 = nn.BatchNorm2d(512)

#     def forward(self, x):
#         h = F.relu(self.c0(x))
#         features = F.relu(self.b1(self.c1(h)))
#         h = F.relu(self.b2(self.c2(features)))
#         h = F.relu(self.b3(self.c3(h)))
#         encoded = self.l1(h.view(x.shape[0], -1))
#         return encoded, features


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(2048, 64, kernel_size=1)  # kernel_size=3
        self.c1 = nn.Conv2d(64, 32, kernel_size=1)  # kernel_size=3
        self.l0 = nn.Linear(160, 512)  # 416
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(2176, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(128, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = nn.DataParallel(GlobalDiscriminator())
        self.local_d = nn.DataParallel(LocalDiscriminator())
        self.prior_d = nn.DataParallel(PriorDiscriminator())
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, M.size(2), M.size(3))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR
