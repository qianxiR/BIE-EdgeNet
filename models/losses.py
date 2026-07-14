import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import rearrange
from .ChangeMambaBCD import lovasz_loss as L

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)

    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

#Focal Loss
def get_alpha(supervised_loader):
    # get number of classes
    num_labels = 0
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        num_labels = max(max(list_unique),num_labels)
    num_classes = num_labels + 1
    # count class occurrences
    alpha = [0 for i in range(num_classes)]
    for batch in supervised_loader:
        label_batch = batch['L']
        label_batch.data[label_batch.data==255] = 0 # pixels of ignore class added to background
        l_unique = torch.unique(label_batch.data)
        list_unique = [element.item() for element in l_unique.flatten()]
        l_unique_count = torch.stack([(label_batch.data==x_u).sum() for x_u in l_unique]) # tensor([65920, 36480])
        list_count = [count.item() for count in l_unique_count.flatten()]
        for index in list_unique:
            alpha[index] += list_count[list_unique.index(index)]
    return alpha

# for FocalLoss
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
            alpha = 1/alpha # inverse of class frequency
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()

        # to resolve error in idx in scatter_
        idx[idx==225]=0
        
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


#miou loss
from torch.autograd import Variable
def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = torch.squeeze(tensor, dim=1).size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.type(torch.int64).view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.weights = Variable(weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

#Minimax iou
class mmIoULoss(nn.Module):
    def __init__(self, n_classes=2):
        super(mmIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        iou = inter/ (union + 1e-8)

        #minimum iou of two classes
        min_iou = torch.min(iou)

        #loss
        loss = -min_iou-torch.mean(iou)
        return loss

def dice_loss(logits, true, eps = 1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    print(num_classes)

    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class myloss(nn.Module):

    def __init__(self, apply_nonlin=None, alpha=None, gamma=1,  smooth=1e-5):
        super(myloss, self).__init__()
        self.focalloss = FocalLoss(apply_nonlin=apply_nonlin, alpha=alpha, gamma=gamma, smooth=smooth)
        self.edgeloss = weight_mse_loss()
        # self.dloss = dice_loss()

    def forward(self, inputs, target, input_edge, target_edge):
        # loss_f = self.focalloss(inputs, target)
        loss_c = cross_entropy(inputs, target)
        # loss_d = self.dloss(inputs, target)
        loss_edge = self.edgeloss(target_edge, input_edge)
        loss = loss_c + loss_edge
        return loss


class weight_mse_loss(nn.Module):
    def __init__(self):
        super(weight_mse_loss, self).__init__()
        # self.batch = batch
        # self.bce_loss = nn.BCELoss()
        # self.target = target
        # self.input = input

    def weight_mse_coeff2(self, target, input):
        err = ((target > 0).float() - input)
        sq_err = err ** 2
        # mean = torch.mean(sq_err)
        # return mean
        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2.0
        is_neg_err = (sign_err - 1) / -2.0

        edge_mass = torch.sum(target == 2).float()
        mid_mass = torch.sum(target == 1).float()
        empty_mass = torch.sum(target == 0).float()
        total_mass = edge_mass + empty_mass + mid_mass

        weight_pos_err = 0.8  # empty_mass  / total_mass
        weight_neg_err = 0.1  # edge_mass / total_mass
        weight_mid_err = 0.2  # mid_mass / total_mass

        pos_part = is_pos_err * sq_err * weight_pos_err
        neg_part = is_neg_err * sq_err * weight_neg_err
        mid_part = is_pos_err * sq_err * weight_mid_err

        weighted_sq_errs = neg_part + pos_part + mid_part

        mean = torch.mean(weighted_sq_errs)
        # if torch.isnan(mean):
        #    mean=torch.Tensor(0)
        return mean

    def weight_mse_coeff(self, target, input):
        err = (target - input)
        sq_err = err ** 2

        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2
        is_neg_err = (sign_err - 1) / -2

        edge_mass = torch.sum(target)
        empty_mass = torch.sum(1 - target)
        total_mass = edge_mass + empty_mass

        weight_pos_err = empty_mass / total_mass
        weight_neg_err = edge_mass / total_mass
        # print(weight_pos_err)

        pos_part = weight_pos_err * is_pos_err * sq_err
        neg_part = weight_neg_err * is_neg_err * sq_err * 2.0

        weighted_sq_errs = neg_part + pos_part

        return torch.mean(weighted_sq_errs)

    def weight_mse_coeff3(self, target, input):
        input = torch.sigmoid(input)
        err = (target - input)
        sq_err = err ** 2
        # mean = torch.mean(sq_err)
        # return mean
        sign_err = torch.sign(err)
        is_pos_err = (sign_err + 1) / 2.0
        is_neg_err = (sign_err - 1) / -2.0

        # edge_mass = torch.sum(target==2).float()
        # mid_mass = torch.sum(target==1).float()
        # empty_mass = torch.sum(target==0).float()
        # total_mass = edge_mass + empty_mass + mid_mass

        weight_pos_err = 0.8  # empty_mass  / total_mass
        weight_neg_err = 0.3  # edge_mass / total_mass
        # weight_mid_err = 0.2#mid_mass / total_mass

        pos_part = is_pos_err * sq_err * weight_pos_err
        neg_part = is_neg_err * sq_err * weight_neg_err
        # mid_part = is_neg_err * (target==1).float() * sq_err * weight_mid_err

        weighted_sq_errs = neg_part + pos_part  # + mid_part

        mean = torch.mean(weighted_sq_errs)
        # if torch.isnan(mean):
        #    mean=torch.Tensor(0)
        return mean

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def weight_bce_loss(self, target, input):
        beta = 1 - torch.mean(target)
        # alpha = 1 - torch.mean(input)
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * target

        return F.binary_cross_entropy(input, target, weights, True)

    def weight_bce_loss2(self, target, input):
        beta = 1 - torch.mean((input > 0.5).float())
        # alpha = 1 - torch.mean(input)
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = 1 - beta + (2 * beta - 1) * (input > 0.5).float()

        return F.binary_cross_entropy(input, target, weights, True)

    def __call__(self, y_true, y_pred):
        a = self.weight_mse_coeff3(y_true, y_pred)
        # b =  self.weight_bce_loss(y_true, y_pred)
        # b = self.soft_dice_loss(y_true, y_pred)
        return a  # + b



# fcddn loss
class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        return self.soft_dice_loss(y_true, y_pred.to(dtype=torch.float32))

class dice_bce_loss(nn.Module):
    """Binary"""
    def __init__(self):
        super(dice_bce_loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels, do_sigmoid=True):

        if len(scores.shape) > 3:
            scores = scores.squeeze(1)
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        if do_sigmoid:
            scores = torch.sigmoid(scores.clone())
        diceloss = self.binnary_dice(scores, labels)
        bceloss = self.bce_loss(scores, labels)
        return diceloss + bceloss


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label==255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 * mask *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss

class BCEDiceLoss(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape)
        bce = F.binary_cross_entropy(inputs, targets)
        inter = (inputs * targets).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
        return bce + 1 - dice


class RCDT_MultiScale_Loss(nn.Module):
    """
    完全适配RCDT模型的多尺度损失函数（修复nearest插值不支持align_corners的错误）
    适配RCDT输出：3个中间尺度（32×32、16×16、8×8） + 1个最终尺度（256×256）
    论文逻辑：总损失 = Σ(每个尺度的 (交叉熵损失 + α·Dice损失)) → 4尺度等权重累加
    适配场景：二分类变化检测（0=无变化，1=有变化），输入标签尺寸固定为256×256
    """

    def __init__(self,
                 alpha=0.4,  # 平衡交叉熵与Dice损失的系数（沿用你的实证最优值）
                 num_scales=4,  # 适配RCDT：3中间尺度 + 1最终尺度 = 4尺度
                 pos_weight=None,  # 二分类交叉熵的正样本权重（应对类别不平衡）
                 downsample_mode="nearest"):  # 离散标签下采样模式（nearest无align_corners）
        super().__init__()
        self.alpha = alpha
        self.num_scales = num_scales
        self.downsample_mode = downsample_mode

        # 1. 交叉熵损失（支持正样本权重，缓解类别不平衡，完全沿用你的配置）
        self.ce_loss = nn.CrossEntropyLoss(weight=pos_weight)

        # 2. RCDT各尺度对应的标签下采样比例（输入256×256时）：
        # - 32×32尺度：256 / 32 = 8倍下采样（RCDT第1个中间尺度）
        # - 16×16尺度：256 / 16 = 16倍下采样（RCDT第2个中间尺度）
        # - 8×8尺度：256 / 8 = 32倍下采样（RCDT第3个中间尺度）
        # - 256×256尺度：256 / 256 = 1倍下采样（RCDT最终预测尺度，不下采样）
        self.scale_ratios = [8, 16, 32, 1]  # 顺序严格对应RCDT输出：[32,16,8,256]

    def _single_scale_dice_loss(self, pred, target):
        """完全沿用你的Dice损失逻辑：只关注变化区域（类别1）"""
        # 取“变化区域（类别1）”的概率（softmax后取第1通道）
        pred_change_prob = F.softmax(pred, dim=1)[:, 1, :, :]  # (B, S, S)

        # 计算交集和并集（加1e-6避免除零错误）
        intersection = (pred_change_prob * target).sum()  # 变化区域的真实正例与预测正例交集
        union = pred_change_prob.sum() + target.sum() + 1e-6  # 变化区域的预测正例与真实正例并集

        # Dice系数 = 2*交集/(并集)，损失 = 1 - Dice系数（损失越小，性能越好）
        dice_coeff = 2 * intersection / union
        return 1 - dice_coeff

    def forward(self, multi_scale_preds, final_pred, target):
        """
        前向计算：4尺度损失求和（32×32 → 16×16 → 8×8 → 256×256）
        Args:
            multi_scale_preds (list[torch.Tensor]): RCDT中间三尺度预测列表
                每个元素形状：(B, 2, S, S)，顺序为 [32×32, 16×16, 8×8]
            final_pred (torch.Tensor): RCDT最终预测（256×256），形状：(B, 2, 256, 256)
            target (torch.Tensor): 真实标签，形状：(B, 1, 256, 256)，值为0或1（float类型）
        Returns:
            total_loss (torch.Tensor): 4尺度损失总和（标量）
        """
        # -------------------------- 1. 输入合法性校验 --------------------------
        # 校验中间多尺度预测数量是否为3
        assert len(multi_scale_preds) == 3, \
            f"RCDT中间尺度预测数量需为3，实际为{len(multi_scale_preds)}"
        # 校验最终预测尺寸是否为256×256
        assert final_pred.shape[2:] == (256, 256), \
            f"最终预测尺寸需为(256,256)，实际为{final_pred.shape[2:]}"
        # 校验标签尺寸是否为256×256
        assert target.shape[2:] == (256, 256), \
            f"标签尺寸需为(256,256)，实际为{target.shape[2:]}"

        # -------------------------- 2. 合并4个尺度的预测（适配你的损失逻辑） --------------------------
        # 顺序：中间3尺度 [32,16,8] + 最终尺度 [256] → 4尺度列表
        all_scale_preds = multi_scale_preds + [final_pred]  # 长度=4

        # -------------------------- 3. 标签预处理 --------------------------
        # 挤压通道维（(B,1,256,256) → (B,256,256)）+ 转long类型（适配CrossEntropyLoss）
        target = target.squeeze(1).long()  # (B, 256, 256)
        B, H, W = target.shape

        # -------------------------- 4. 遍历4尺度计算损失（修复interpolate参数错误） --------------------------
        total_loss = 0.0
        for i in range(self.num_scales):
            # 4.1 获取当前尺度的预测和下采样比例
            scale_pred = all_scale_preds[i]  # (B, 2, S, S)，S=32/16/8/256
            scale_ratio = self.scale_ratios[i]  # 当前尺度的标签下采样比例

            # 4.2 标签下采样/保持到当前尺度（与预测尺寸匹配）
            # 临时添加通道维：(B,256,256) → (B,1,256,256)（适配interpolate输入要求）
            target_unsqueeze = target.unsqueeze(1).float()

            # 修复核心：根据插值模式决定是否传入align_corners
            if self.downsample_mode in ["linear", "bilinear", "bicubic", "trilinear"]:
                # 仅支持的模式才传align_corners
                target_scale = F.interpolate(
                    input=target_unsqueeze,
                    size=(H // scale_ratio, W // scale_ratio),  # 适配RCDT各尺度尺寸
                    mode=self.downsample_mode,
                    align_corners=False  # 避免角点偏移
                ).squeeze(1).long()
            else:
                # nearest模式（离散标签）：不传入align_corners（PyTorch不支持）
                target_scale = F.interpolate(
                    input=target_unsqueeze,
                    size=(H // scale_ratio, W // scale_ratio),
                    mode=self.downsample_mode  # 仅保留模式参数
                ).squeeze(1).long()

            # 4.3 计算当前尺度的交叉熵损失和Dice损失（完全沿用你的公式）
            ce_scale = self.ce_loss(scale_pred, target_scale)  # 单尺度交叉熵损失
            dice_scale = self._single_scale_dice_loss(scale_pred, target_scale)  # 单尺度Dice损失

            # 4.4 累加当前尺度的损失（等权重累加，严格遵循你的论文逻辑）
            total_loss += ce_scale + self.alpha * dice_scale

        return total_loss

class SegEvaluator:
    def __init__(self, class_num=4):
        if class_num == 1:
            class_num = 2
        self.num_class = class_num
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def kappa(self,OA):
        pe_rows = np.sum(self.confusion_matrix, axis=0)
        pe_cols = np.sum(self.confusion_matrix, axis=1)
        sum_total = np.sum(self.confusion_matrix)
        pe = np.dot(pe_rows, pe_cols) / (sum_total ** 2)
        #po = self.pixel_oa()
        po = OA
        return (po - pe) / (1 - pe)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.mat=self.confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def loss_weight(self):
        TN = self.confusion_matrix[0][0]
        FP = self.confusion_matrix[0][1]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        w_00 = TP / (TP + FP + FN)
        w_11 = TN / (TN + FN + FP)
        return w_00, w_11

    def matrix(self,class_index):
        metric = {}
        recall = 0.0
        precision = 0.0
        for i in range(self.num_class):
            recall += self.confusion_matrix[i, i] / (np.sum(self.confusion_matrix[:, i]) + 1e-8)
            precision += self.confusion_matrix[i, i] / (np.sum(self.confusion_matrix[i, :]) + 1e-8)
        precision_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        recall_cls = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        iou_per_class = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        metric['0_IoU'] = iou_per_class[0]
        metric['1_IoU'] = iou_per_class[1]
        metric['IoU'] = np.nanmean(iou_per_class)
        metric['Precision'] = precision_cls[class_index]  #precision / self.num_class
        metric['Recall'] = recall_cls[class_index]          #recall / self.num_class
        metric['OA'] = OA
        metric['F1'] = (2 * precision_cls[class_index] * recall_cls[class_index]) / (precision_cls[class_index] + recall_cls[class_index])
        Kappa = self.kappa(OA)
        metric['Kappa'] = Kappa
        return metric

# class AERNet_Loss(nn.Module):
#     def __init__(self):
#         super(AERNet_Loss,self).__init__()
#
#     def forward(self, input, target):
#
#
#         evaluator = SegEvaluator(1)
#         evaluator.reset()
#         pred = torch.where(torch.sigmoid(input) > 0.5, 1, 0)
#         evaluator.add_batch(gt_image=target.cpu().numpy(), pre_image=pred.cpu().numpy())
#         w_00,w_11 = evaluator.loss_weight()
#         weight1 = torch.zeros_like(target)
#         weight1 = torch.fill_(weight1, w_00)
#         weight1[target > 0] = w_11
#         loss = F.binary_cross_entropy_with_logits(input, target,weight=weight1,reduction="mean")
#
#         return loss



class AERNet_Loss(nn.Module):
    def __init__(self, smooth=1e-6, weight_max=10.0):
        super(AERNet_Loss, self).__init__()
        self.smooth = smooth
        self.weight_max = weight_max
        self.evaluator = SegEvaluator(1)

    def forward(self, input, target):
        # -------------------------- 1. 输入校验与预处理 --------------------------
        target = target.float()
        assert input.shape == target.shape, f"输入形状不匹配：input={input.shape}, target={target.shape}"
        input_clamped = torch.clamp(input, min=-10.0, max=10.0)

        # -------------------------- 2. 预测结果计算 --------------------------
        pred = (torch.sigmoid(input_clamped) > 0.5).float().detach()

        # -------------------------- 3. 鲁棒权重计算（核心修复：处理全背景/全前景） --------------------------
        self.evaluator.reset()
        self.evaluator.add_batch(
            gt_image=target.detach().cpu().numpy().astype(np.int64),
            pre_image=pred.cpu().numpy().astype(np.int64)
        )
        w_00, w_11 = self.evaluator.loss_weight()

        # 关键修复1：获取当前批次的标签占比（判断是否全背景/全前景）
        target_ratio = target.mean().item()  # 0.0=全背景，1.0=全前景，0~1=混合
        # print(f"批次标签占比：{target_ratio:.4f} | 原始w00：{w_00} | 原始w11：{w_11}")  # 调试用，可后续删除

        # 关键修复2：强制替换 NaN/Inf 为默认值（1.0），再处理特殊批次
        w_00 = 1.0 if np.isnan(w_00) or np.isinf(w_00) else w_00
        w_11 = 1.0 if np.isnan(w_11) or np.isinf(w_11) else w_11

        # 关键修复3：针对全背景/全前景批次，直接设置合理权重（避免依赖错误统计）
        if target_ratio == 0.0:  # 全背景批次：背景权重=1.0，前景权重无意义（设1.0）
            w_00 = 1.0
            w_11 = 1.0
        elif target_ratio == 1.0:  # 全前景批次：前景权重=1.0，背景权重无意义（设1.0）
            w_00 = 1.0
            w_11 = 1.0

        # 转换为张量并钳位（确保权重在合理范围）
        w_00 = torch.tensor(w_00, dtype=input.dtype, device=input.device)
        w_11 = torch.tensor(w_11, dtype=input.dtype, device=input.device)
        w_00 = torch.clamp(w_00, min=self.smooth, max=self.weight_max)
        w_11 = torch.clamp(w_11, min=self.smooth, max=self.weight_max)

        # -------------------------- 4. 构建权重张量 --------------------------
        weight1 = torch.full_like(target, fill_value=w_00, dtype=input.dtype)
        weight1[target > 0.5] = w_11

        # -------------------------- 5. 稳定损失计算 --------------------------
        loss = F.binary_cross_entropy_with_logits(
            input_clamped,
            target,
            weight=weight1,
            reduction="mean"
        )

        # 最终校验（避免遗漏）
        assert not torch.isnan(loss), f"损失为NaN！w00={w_00.item()}, w11={w_11.item()}, 标签占比={target_ratio}"

        return loss

class HSANet_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss().cuda()

    def forward(self, input, target):
        preds = input.cuda()
        Y = target.cuda()
        loss = self.criterion(preds[:, 0:1, :, :], Y) + self.criterion(preds[:, 1:2, :, :], Y)

        return loss


class LENetLoss(nn.Module):
    def __init__(self, main_loss_weight=1.0, aux_loss_weight=0.3, ignore_index=255):
        super().__init__()
        self.main_loss_weight = main_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index

        # 定义损失函数（与配置一致的CrossEntropyLoss）
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=None,
            ignore_index=ignore_index,
            reduction='mean'
        )

    def forward(self, main_logits, aux_logits, data_samples):
        """
        计算总损失
        Args:
            main_logits: 主解码头输出 (batch_size, num_classes, H, W)
            aux_logits: 辅助解码头输出列表 [logits1, logits2]
            data_samples: 包含标签的SegDataSample列表
        Returns:
            loss_dict: 包含总损失、主损失、辅助损失的字典
        """
        # 提取标签 (batch_size, H, W)
        labels = torch.stack([ds.gt_sem_seg.data for ds in data_samples], dim=0).long()
        batch_size, H, W = labels.shape

        # 调整logits尺寸与标签一致
        main_logits = F.interpolate(main_logits, size=(H, W), mode='bilinear', align_corners=False)
        aux_logits1 = F.interpolate(aux_logits[0], size=(H, W), mode='bilinear', align_corners=False)
        aux_logits2 = F.interpolate(aux_logits[1], size=(H, W), mode='bilinear', align_corners=False)

        # 计算损失
        main_loss = self.cross_entropy_loss(main_logits, labels)
        aux_loss1 = self.cross_entropy_loss(aux_logits1, labels)
        aux_loss2 = self.cross_entropy_loss(aux_logits2, labels)

        # 总损失 = 主损失×权重 + 辅助损失×权重
        total_loss = self.main_loss_weight * main_loss + self.aux_loss_weight * aux_loss1 + self.aux_loss_weight * aux_loss2


        # return {
        #     'total_loss': total_loss,
        #     'main_loss': main_loss,
        #     'aux_loss1': aux_loss1,
        #     'aux_loss2': aux_loss2
        # }
        return total_loss


class B2CNetLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super().__init__()
        # 论文默认类别权重：未变化1.0，变化2.0（适配样本不平衡）
        self.weight = weight if weight is not None else torch.tensor([1.0, 2.0])
        self.smooth = smooth  # Dice损失平滑项
        self.lambda1 = 1.0  # 主输出out权重（论文固定）
        self.lambda2 = 0.5  # 辅助输出out2权重（论文固定）

    def forward(self, out, out2, gt):
        """
        Args:
            out: 主输出 [B, 2, H, W]
            out2: 辅助输出 [B, 2, H, W]
            gt: 真实标签 [B, 1, H, W] 或 [B, H, W]（兼容两种格式）
        Returns:
            total_loss: 双分支加权总损失
        """
        # 关键修复：去除标签的冗余通道维（dim=1），转为3D张量 [B, H, W]
        if gt.dim() == 4 and gt.shape[1] == 1:
            gt = gt.squeeze(1)  # 压缩dim=1，[B,1,H,W] → [B,H,W]

        # 确保标签是long类型（cross_entropy要求类别标签为整数）
        gt = gt.long()

        # 计算单分支损失和总损失
        loss_out = self._single_branch_loss(out, gt)
        loss_out2 = self._single_branch_loss(out2, gt)
        total_loss = self.lambda1 * loss_out + self.lambda2 * loss_out2
        return total_loss

    def _single_branch_loss(self, pred, gt):
        """计算单个输出（out/out2）的混合损失：L_wce + L_dice"""
        B, C, H, W = pred.shape

        # ---------------------- 加权交叉熵损失（L_wce）----------------------
        l_wce = F.cross_entropy(
            input=pred,  # [B, 2, H, W]（4D）
            target=gt,  # [B, H, W]（3D，修复后格式）
            weight=self.weight.to(pred.device),  # 类别权重适配设备
            reduction='mean'
        )

        # ---------------------- Dice损失（L_dice）----------------------
        # 取变化类（class=1）的概率图
        pred_softmax = F.softmax(pred, dim=1)[:, 1, :, :]  # [B, H, W]
        gt_flat = gt.view(B, -1)  # [B, H*W]
        pred_flat = pred_softmax.view(B, -1)  # [B, H*W]

        intersection = (pred_flat * gt_flat).sum(dim=1)  # 批次内交集
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)  # 批次内并集
        l_dice = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        l_dice = l_dice.mean()  # 批次平均

        return l_wce + l_dice

class DINOFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0):
        super(DINOFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.as_tensor(alpha)

    def forward(self, input, target):
        N, C, H, W = input.size()
        assert C == 2
        # input = input.view(N, C, -1)
        # input = input.transpose(1, 2)
        # input = input.contiguous().view(-1, C)
        input = rearrange(input, 'b c h w -> (b h w) c')
        # input = input.contiguous().view(-1)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.mean()

from kornia.losses import dice_loss

class DINODICELoss(nn.Module):
    def __init__(self):
        super(DINODICELoss, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1)
        loss = dice_loss(input, target)

        return loss

class DINO_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4):
        super().__init__()
        self.focal = DINOFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DINODICELoss()

    def forward(self, final_pred, preds, target):
        label = target.long()
        focal = self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for i in range(len(preds)):
            focal += self.focal(preds[i], label)
            dice += 0.5 * self.dice(preds[i], label)

        loss = focal * 0.5 + dice

        return loss

class MambaBCDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        labels = target.cuda().long().squeeze(1)

        ce_loss_1 = F.cross_entropy(input, labels, ignore_index=255)
        lovasz_loss = L.lovasz_softmax(F.softmax(input, dim=1), labels, ignore=255)
        main_loss = ce_loss_1 + 0.75 * lovasz_loss
        final_loss = main_loss

        return final_loss

class EGRCNN_FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(EGRCNN_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class EGRCNN_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = EGRCNN_FocalLoss(gamma=2.0, alpha=0.25)
        self.criterion1 = nn.MSELoss()

    def forward(self, list_out, list_edge, target, target_edge):
        d6_out, d5_out, d4_out, d3_out, d2_out = list_out
        d3_edge, d2_edge = list_edge
        labels = (target > 0).squeeze(1).type(torch.LongTensor).to("cuda")
        # mse label
        labels_edge_2 = (target_edge > 0).type(torch.LongTensor).to("cuda")
        labels_edge_1 = torch.ones((labels_edge_2.shape[0], 1, labels_edge_2.shape[2], labels_edge_2.shape[3])).type(
            torch.LongTensor).to("cuda")
        labels_edge_1 = torch.sub(labels_edge_1, labels_edge_2)
        labels_edge = torch.cat((labels_edge_1, labels_edge_2), dim=1)

        # Calculate Loss
        loss_seg_2 = self.criterion(d2_out, labels)
        loss_seg_3 = self.criterion(d3_out, labels)
        loss_seg_4 = self.criterion(d4_out, labels)
        loss_seg_5 = self.criterion(d5_out, labels)
        loss_seg_6 = self.criterion(d6_out, labels)
        loss_edge_2 = self.criterion1(F.softmax(d2_edge, dim=1), labels_edge.float())  # mse_loss
        loss_edge_3 = self.criterion1(F.softmax(d3_edge, dim=1), labels_edge.float())
        loss_edge = 10 * (loss_edge_2 + loss_edge_3)
        loss_seg = loss_seg_2 + loss_seg_3 + loss_seg_4 + loss_seg_5 + loss_seg_6
        loss = loss_edge + loss_seg

        return loss