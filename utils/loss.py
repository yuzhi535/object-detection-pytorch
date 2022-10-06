import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, preds, targets):
        """
        计算边界框回归损失。N表示RoI个数
        :param preds: 大小为[N, 4]，分别保存每个边界框的预测x,y,w,h
        :param targets: 大小为[N, 4]，分别保存每个边界框的实际x,y,w,h
        :return:
        """
        res = self.smooth_l1(preds - targets)
        return torch.sum(res)

    def smooth_l1(self, x):
        if torch.abs(x) < 1:
            return 0.5 * torch.pow(x, 2)
        else:
            return torch.abs(x) - 0.5
        

class MultiTaskLoss(nn.Module):

    def __init__(self, lam=1):
        super(MultiTaskLoss, self).__init__()
        self.lam = lam
        # L_cls使用交叉熵损失
        self.cls = nn.CrossEntropyLoss()
        # L_loc自定义
        self.loc = SmoothL1Loss()

    def forward(self, scores, preds, targets):
        """
        计算多任务损失。N表示RoI数目
        :param scores: [N, C]，C表示类别数
        :param preds: [N, 4]，4表示边界框坐标x,y,w,h
        :param targets: [N]，0表示背景类
        :return:
        """
        N = targets.shape[0]
        return self.cls(scores, targets) + self.loc(scores[range(N), self.indicator(targets)],
                                                    preds[range(N), self.indicator(preds)])

    def indicator(self, cate):
        return cate != 0