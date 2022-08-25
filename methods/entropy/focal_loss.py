from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable

class focal_loss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples


    """
    def __init__(self, alpha=None, gamma=2,class_num=10):
        super(focal_loss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
            self.alpha = alpha
        self.gamma = gamma

    def forward(self,inputs,targets):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, targets, self.alpha)
        return loss
