import torch
from torch.nn.modules.loss import _Loss

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10, device='cuda'):
        super(SpreadLoss, self).__init__()
        self.dev = device
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        x = x.to(self.dev)
        at = torch.cuda.FloatTensor(b).fill_(0).to(self.dev)
        print('MEEP')
        print(at)
        target = torch.argmax(target, dim=1)
        print(target)
        for i, lb in enumerate(target):
            idx = torch.LongTensor(i).to(self.dev)
            lbx = lb.to(self.dev)
            print(lb)
            at[idx] = x[idx][lbx]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss
