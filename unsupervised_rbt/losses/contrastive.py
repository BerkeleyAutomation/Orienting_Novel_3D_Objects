import torch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, prob, label):
        
        loss_contrastive = torch.mean((1-label) * torch.pow(prob, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - prob, min=0.0), 2))
        return loss_contrastive
