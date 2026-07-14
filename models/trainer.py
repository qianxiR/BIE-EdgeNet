import numpy as np
import matplotlib.pyplot as plt
import os
import gc

import utils
from models.networks import *

import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy, myloss
import models.losses as losses
from models.losses import (get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss, BCEDiceLoss,
                           RCDT_MultiScale_Loss, AERNet_Loss, HSANet_Loss, LENetLoss, B2CNetLoss,
                           DINO_Loss, MambaBCDLoss, EGRCNN_Loss)
from models.EGPNet.loss import EGP_FocalLoss,EGP_dice_loss
from models.BGSLoss import CrossEntropyLoss2d, weighted_bce, ChangeSimilarity, AutomaticWeightedLoss

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm



# import swanlab

# from torch.cuda.amp import GradScaler, autocast


class CDTrainer():

    def __init__(self, args, dataloaders):

        # 新增：初始化梯度缩放器
        # self.scaler = GradScaler()

        self.args = args
        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        self.net_G.to(self.device)
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                          weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)

        # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0.0
        self.best_val_acc = 0.0
        self.best_epoch_acc_id = 0
        self.epoch_f1 = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.multi_scale_preds = None
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)

        if args.loss == 'ce':
            self._pxl_loss = cross_entropy

        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce

        elif args.loss == 'fl':
            print('\n Calculating alpha in Focal-Loss (FL) ...')
            alpha = get_alpha(dataloaders['train'])  # calculare class occurences
            print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=2, smooth=1e-5)

        elif args.loss == "miou":
            print('\n Calculating Class occurances in training set...')
            alpha = np.asarray(get_alpha(dataloaders['train']))  # calculare class occurences
            alpha = alpha / np.sum(alpha)
            # weights = torch.tensor([1.0, 1.0]).cuda()
            weights = 1 - torch.from_numpy(alpha).cuda()
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()

        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()

        elif args.loss == "eas":
            # print('\n Calculating Class occurances in training set...')
            # alpha = get_alpha(dataloaders['train'])
            # print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss = myloss(apply_nonlin=softmax_helper, alpha=None, gamma=2, smooth=1e-5)

        elif args.loss == "BCEDiceLoss":
            self._pxl_loss = BCEDiceLoss()

        elif args.loss == "RCDT_MultiScale_Loss":
            self._pxl_loss = RCDT_MultiScale_Loss()

        elif args.loss == "AERNet_Loss":
            self._pxl_loss = AERNet_Loss()

        elif self.args.net_G == 'HSANet':
            self._pxl_loss = HSANet_Loss()

        elif args.net_G == 'LENet':
            self._pxl_loss = LENetLoss(main_loss_weight=1.0, aux_loss_weight=0.3)

        elif args.net_G == 'B2CNet':
            self._pxl_loss = B2CNetLoss()

        elif args.net_G == 'ChangeDINO' or args.net_G == 'BIE_ChangeDINO':
            self._pxl_loss = DINO_Loss()

        elif args.net_G == 'ChangeMamba':
            self._pxl_loss = MambaBCDLoss()

        elif args.net_G == 'EGRCNN':
            self._pxl_loss = EGRCNN_Loss()

        elif args.net_G == 'EGPNet':
            self._pxl_loss = [EGP_FocalLoss(apply_nonlin=nn.Softmax(dim=1)),EGP_dice_loss]

        elif args.net_G == 'EATDer':
            self._pxl_loss = [nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda()),
                              nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).cuda())]

        elif args.net_G == 'BGSNet':
            self._pxl_loss = AutomaticWeightedLoss(3)

        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


        # 配置API Key
        # swanlab.login(api_key="jjNAAfoJUvmzWOIClfOqL", save=True)
        # 初始化项目（项目名使用args.project_name）
        # self.current_score_dict = None
        # self.swan_exp = swanlab.init(
        #     project=args.project_name,
        #     config=vars(args)  # 自动记录所有args参数（如lr、batch_size、loss类型等）
        # )

    def _load_checkpoint(self, ckpt_name='best_ckpt.pt'):
        print("\n")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading [ {} ] checkpoint...\n'.format(ckpt_name))
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device, weights_only=False)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_acc_id = checkpoint['best_epoch_acc_id']
            self.best_val_f1 = checkpoint['best_val_f1']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, \nHistorical_best_acc = %.4f (at epoch %d)\nHistorical_best_f1=%.4f (at epoch %d)\n'
                              %(self.epoch_to_start, self.best_val_acc, self.best_epoch_acc_id, self.best_val_f1, self.best_epoch_id))
            self.logger.write('\n')
        elif self.args.pretrain is not None:
            print("Initializing backbone weights from: " + self.args.pretrain)
            self.net_G.load_state_dict(torch.load(self.args.pretrain), strict=False)
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        # torch.save({
        #     'epoch_id': self.epoch_id,
        #     'best_val_acc': self.best_val_acc,
        #     'best_epoch_id': self.best_epoch_id,
        #     'model_G_state_dict': self.net_G.state_dict(),
        #     'optimizer_G_state_dict': self.optimizer_G.state_dict(),
        #     'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        # }, os.path.join(self.checkpoint_dir, ckpt_name))
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_acc_id': self.best_epoch_acc_id,
            'best_val_f1': self.best_val_f1,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()

        G_pred = self.G_final_pred.detach()

        # G_pred = torch.argmax(G_pred, dim=1)
        if self.n_class == 2:
            if self.args.loss == 'BCEDiceLoss' and self.args.net_G == 'ChangeViT':
                G_pred_prob = torch.sigmoid(G_pred)  # (16,1,256,256)
                G_pred = (G_pred_prob > 0.7).float()

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.args.net_G == 'AERNet':
                G_pred = torch.where(torch.sigmoid(G_pred) > 0.5, 1, 0)

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            elif self.args.net_G == 'HSANet':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.args.net_G == 'RCDT':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))
            elif self.args.net_G == 'LENet':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))
            elif self.args.net_G == 'B2CNet':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.args.net_G == 'ChangeDINO' or self.args.net_G == 'BIE_ChangeDINO':
                # torch.argmax返回指定维度最大值的序号,如果c=1，那么所有的序号都是0
                G_pred = torch.argmax(G_pred.detach(), dim=1)
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            elif self.args.net_G == 'EGRCNN':
                # torch.argmax返回指定维度最大值的序号,如果c=1，那么所有的序号都是0
                G_pred = torch.argmax(G_pred, dim=1)
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            # elif self.args.net_G == 'ChangeMamba':
            elif self.args.net_G == 'EGPNet':
                G_pred = torch.argmax(G_pred, dim=1).long()
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.detach().cpu().numpy(),
                                                              gt=target.detach().cpu().numpy())

            elif self.args.net_G == 'EATDer':
                G_pred = torch.sigmoid(G_pred)
                G_pred = torch.where(G_pred > 0.5, 1, 0).int()

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            else:
                # torch.argmax返回指定维度最大值的序号,如果c=1，那么所有的序号都是0
                G_pred = torch.argmax(G_pred, dim=1)
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

        self.current_score_dict = self.running_metric.get_scores()
        return current_score
    # def _update_metric(self):
    #     """简化metric计算，减少CPU-GPU交互"""
    #     target = self.batch['L'].to(self.device).detach()
    #     G_pred = self.G_final_pred.detach()
    #
    #     if self.n_class == 2:
    #         if self.args.loss == 'BCEDiceLoss' and self.args.net_G == 'ChangeViT':
    #             G_pred_prob = torch.sigmoid(G_pred)
    #             G_pred = (G_pred_prob > 0.7).float()
    #         elif self.args.net_G in ['AERNet', 'HSANet', 'RCDT', 'LENet', 'B2CNet']:
    #             G_pred = F.sigmoid(G_pred[:, 1, :, :])
    #             G_pred = (G_pred >= 0.5).float()
    #         elif self.args.net_G in ['ChangeDINO', 'EGRCNN', 'EGPNet']:
    #             G_pred = torch.argmax(G_pred, dim=1).long()
    #         elif self.args.net_G == 'EATDer':
    #             G_pred = torch.sigmoid(G_pred)
    #             G_pred = (G_pred > 0.5).int()
    #         else:
    #             G_pred = torch.argmax(G_pred, dim=1)
    #
    #     # -------------------------- 优化：减少CPU拷贝频率，批量更新 --------------------------
    #     current_score = self.running_metric.update_cm(
    #         pr=G_pred.cpu().numpy().astype(np.int64),
    #         gt=target.cpu().numpy().astype(np.int64)
    #     )
    #     self.current_score_dict = self.running_metric.get_scores()
    #     return current_score



    def _collect_running_batch_states(self):

        # F1
        running_f1 = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        # if np.mod(self.batch_id, 20) == 1:  # 每10个batch打印一次
        #     self.print_gpu_memory()
        if np.mod(self.batch_id, 100) == 1:

            # 构建批次日志字典
            batch_log = {
                f"{'train' if self.is_training else 'val'}/step_F1_1": running_f1,
                f"{'train' if self.is_training else 'val'}/step_acc": self.current_score_dict['acc'],
                f"{'train' if self.is_training else 'val'}/step_iou_1": self.current_score_dict['iou_1'],
                f"{'train' if self.is_training else 'val'}/step_precision_1": self.current_score_dict['precision_1'],
                f"{'train' if self.is_training else 'val'}/step_recall_1": self.current_score_dict['recall_1']
            }
            # 训练阶段额外记录loss
            if self.is_training:
                batch_log["train/step_loss"] = self.G_loss.item()
            # 写入SWANLab（step使用global_step确保连续）
            # swanlab.log(batch_log, step=self.global_step)

                message = 'Is_training: %s | Epoch:[%d,%d] | Images(split by batch_size):[%d,%d] | imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' % \
                            (self.is_training, self.epoch_id + 1, self.max_num_epochs, self.batch_id, m,
                            imps * self.batch_size, est,
                            self.G_loss.item(), running_f1)
                self.logger.write(message)

        # if np.mod(self.batch_id, 500) == 1:
        #     vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        #     vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        #
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        #
        #     vis_gt = utils.make_numpy_grid(self.batch['L'])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         self.vis_dir, 'istrain_' + str(self.is_training) + '_' +
        #                       str(self.epoch_id) + '_' + str(self.batch_id) + '.jpg')
        #     plt.imsave(file_name, vis)

            # # -------------------------- 4. 记录可视化图片到SWANLab --------------------------
            # swanlab.log({
            #     f"{'train' if self.is_training else 'val'}/visualization": swanlab.Image(
            #         vis,
            #         caption=f"Epoch {self.epoch_id + 1}, Batch {self.batch_id}: A→B→Pred→GT"
            #     )
            # }, step=self.global_step)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['acc']
        self.epoch_f1 = scores['F1_1']
        self.epoch_iou_1 = scores['iou_1']
        self.epoch_precision_1 = scores['precision_1']
        self.epoch_recall_1 = scores['recall_1']

        # # -------------------------- 5. 记录epoch级核心指标到SWANLab --------------------------
        # epoch_log = {
        #     f"{'train' if self.is_training else 'val'}/epoch_acc": self.epoch_acc,
        #     f"{'train' if self.is_training else 'val'}/epoch_F1_1": self.epoch_f1,
        #     f"{'train' if self.is_training else 'val'}/epoch_iou_1": self.epoch_iou_1,
        #     f"{'train' if self.is_training else 'val'}/epoch_precision_1": self.epoch_precision_1,
        #     f"{'train' if self.is_training else 'val'}/epoch_recall_1": self.epoch_recall_1,
        #     "lr": self.optimizer_G.param_groups[0]['lr']  # 记录当前学习率
        # }
        # # 训练阶段额外记录epoch平均loss（需先在train循环中计算epoch_loss）
        # if self.is_training:
        #     epoch_log["train/epoch_loss"] = self.epoch_loss  # self.epoch_loss需在train循环中累加计算
        # # 写入SWANLab（step使用epoch_id，更直观）
        # swanlab.log(epoch_log, step=self.epoch_id + 1)


        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_F1_1= %.5f, epoch_acc= %.5f\n' %
                          (self.is_training, self.epoch_id + 1, self.max_num_epochs, self.epoch_f1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message + '\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        epoch_name = 'last_ckpt_{}.pt'.format(self.epoch_id + 1)
        self._save_checkpoint(ckpt_name=epoch_name)
        self.logger.write('The .pt for the epoch %d has been synchronized.\n' % (self.epoch_id + 1))

        # update the best model (based on eval acc)
        if self.epoch_f1 > self.best_val_f1:
            self.best_val_f1 = self.epoch_f1
            self.best_epoch_id = self.epoch_id + 1
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('Best model updated!\n'
                              'Historical_best_f1=%.4f (at epoch %d)\n--------------------------------\n'
                              % (self.best_val_f1, self.best_epoch_id))

            # # -------------------------- 6. 记录最佳模型指标到SWANLab --------------------------
            # swanlab.log({
            #     "val/best_F1_1": self.best_val_f1,
            #     "val/best_epoch": self.best_epoch_id
            # }, step=self.epoch_id + 1)

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_acc_id = self.epoch_id + 1
            self._save_checkpoint(ckpt_name='best_acc_ckpt.pt')
            self.logger.write('Highest Accuracy model updated!\n'
                              'Historical_best_acc=%.4f (at epoch %d)\n--------------------------------\n'
                              % (self.best_val_acc, self.best_epoch_acc_id))
            # # 记录最佳acc指标
            # swanlab.log({
            #     "val/best_acc": self.best_val_acc,
            #     "val/best_acc_epoch": self.best_epoch_acc_id
            # }, step=self.epoch_id + 1)

        # save current model

        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated.\n'
                          'Epoch_f1=%.4f, Epoch_acc=%.4f(at epoch %d)\n'
                          'best_f1=%.4f (at epoch %d)\n'
                          'best_acc=%.4f (at epoch %d)\n--------------------------------\n'
                          % (self.epoch_f1, self.epoch_acc, self.epoch_id + 1, self.best_val_f1, self.best_epoch_id,
                             self.best_val_acc, self.best_epoch_acc_id))
        self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()
        # 重置epoch级loss累加器（训练阶段用）
        self.epoch_loss = 0.0

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device, non_blocking=True)
        img_in2 = batch['B'].to(self.device, non_blocking=True)

        # -------------------------------------------------------------------------
        if self.args.net_G == 'RCDT':
            self.multi_scale_preds, self.G_pred = self.net_G(img_in1, img_in2)

        elif self.args.net_G == 'HSANet':
            self.G_pred_tuple = self.net_G(img_in1, img_in2)
            self.G_pred = torch.cat(self.G_pred_tuple, dim=1)

        elif self.args.net_G == 'LENet':
            self.Combined_inputs = torch.cat([img_in1, img_in2], dim=1)
            self.main_logits, self.aux_logits = self.net_G(self.Combined_inputs)

        elif self.args.net_G == 'B2CNet':
            self.out, self.out2 = self.net_G(img_in1, img_in2)

        elif self.args.net_G == 'ChangeDINO' or self.args.net_G == 'BIE_ChangeDINO':
            self.G_pred, self.preds = self.net_G(img_in1, img_in2)

        elif self.args.net_G == 'EGRCNN':
            self.image_input = torch.stack([img_in1, img_in2], dim=0)
            self.out_list, self.edge_list = self.net_G(self.image_input)
            d6_out, d5_out, d4_out, d3_out, d2_out = self.out_list
            self.G_pred = d2_out

        elif self.args.net_G == 'EGPNet':
            self.G_pred_List, self.Edge_pred = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred_List[0]

        elif self.args.net_G == 'EATDer':
            self.Edge_pred, self.G_pred = self.net_G(img_in1, img_in2)

        elif self.args.net_G == 'BGSNet':
            self.G_pred_List = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred_List[0]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)
        # -------------------------------------------------------------------------
        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size()).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred = self.G_final_pred + F.interpolate(pred, size=self.G_pred[-1].size(2),
                                                                          mode="nearest")
                else:
                    self.G_final_pred = self.G_final_pred + pred
            self.G_final_pred = self.G_final_pred / len(self.G_pred)
        elif self.args.loss == 'BCEDiceLoss' and self.args.net_G == 'ChangeViT':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'RCDT':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'VcT':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'HSANet':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'LENet':
            self.G_final_pred = self.main_logits
        elif self.args.net_G == 'B2CNet':
            self.G_final_pred = self.out
        elif self.args.net_G == 'ChangeDINO' or self.args.net_G == 'BIE_ChangeDINO':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'ChangeMamba':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'CTDFormer':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'EGRCNN':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'EGPNet':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'EATDer':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'BIT_ResNet':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'BIT_ResNet1':
            self.G_final_pred = self.G_pred
        elif self.args.net_G == 'BGSNet':
            self.G_final_pred = self.G_pred
        else:
            self.G_final_pred = self.G_pred[-1]

    def _backward_G(self):
        gt = self.batch['L'].to(self.device, non_blocking=True).float()

        if self.multi_scale_train == "True":
            i = 0
            temp_loss = 0.0
            for pred in self.G_pred:
                if pred.size(2) != gt.size(2):
                    temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2),
                                                                                                 mode="nearest"))
                else:
                    temp_loss = temp_loss + self.weights[i] * self._pxl_loss(pred, gt)
                i += 1
            self.G_loss = temp_loss
        else:
            if self.args.loss == 'eas':
                gt_edge = self.batch['L_edge'].to(self.device, non_blocking=True).float()
                self.G_loss = self._pxl_loss(self.G_pred[-1], gt, self.G_pred[-2], gt_edge)
            elif self.args.loss == 'BCEDiceLoss' and self.args.net_G == 'ChangeViT':
                self.G_loss = self._pxl_loss(self.G_pred, gt)
            elif self.args.loss == 'RCDT_MultiScale_Loss' and self.args.net_G == 'RCDT':
                self.G_loss = self._pxl_loss(self.multi_scale_preds, self.G_pred, gt)
            # elif self.args.loss == 'ce' and self.args.net_G == 'RCDT':
            #     self.G_loss = self._pxl_loss(self.G_pred, gt)
            elif self.args.loss == 'AERNet_Loss' and self.args.net_G == 'AERNet':
                self.G_loss = self._pxl_loss(self.G_pred[-1], gt)

            elif self.args.net_G == 'VcT':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            elif self.args.net_G == 'HSANet':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            elif self.args.net_G == 'LENet':
                self.G_loss = self._pxl_loss(self.main_logits, self.aux_logits, gt)

            elif self.args.net_G == 'B2CNet':
                self.G_loss = self._pxl_loss(self.out, self.out2, gt)

            elif self.args.net_G == 'ChangeDINO' or self.args.net_G == 'BIE_ChangeDINO':
                self.G_loss = self._pxl_loss(self.G_pred, self.preds, gt)

            elif self.args.net_G == 'ChangeMamba':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            elif self.args.net_G == 'CTDFormer':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            elif self.args.net_G == 'EGRCNN':
                gt_edge = self.batch['L_edge'].to(self.device).float()
                self.G_loss = self._pxl_loss(self.out_list, self.edge_list, gt, gt_edge)

            elif self.args.net_G == 'EGPNet':
                pre1, pre2, pre3, pre4, pre5 = self.G_pred_List
                predge = self.Edge_pred
                gt_edge = self.batch['L_edge'].to(self.device).float()

                l1 = self._pxl_loss[0](pre1, gt)
                l2 = self._pxl_loss[0](pre2, gt)
                l3 = self._pxl_loss[0](pre3, gt)
                l4 = self._pxl_loss[0](pre4, gt)
                l5 = self._pxl_loss[0](pre5, gt)
                ltotal = l1 + (l2 + l3 + l4 + l5) / 4
                # 边缘损失
                ledge = self._pxl_loss[1](predge, gt_edge)
                self.G_loss = ltotal + 0.1 * ledge

            elif self.args.net_G == 'EATDer':
                gt_edge = self.batch['L_edge'].to(self.device).float()
                predge = self.Edge_pred
                phi = 0.3
                self.G_loss = ((1-phi) *
                               self._pxl_loss[0](predge,gt_edge.cuda()) +
                               phi *
                               self._pxl_loss[1](self.G_pred,gt.cuda()))

            elif self.args.net_G == 'BGSNet':
                gt_edge = self.batch['L_edge'].to(self.device).float()
                out_change, outputs_A, outputs_B, out_bd = self.G_pred_List

                # loss_seg1 = self._pxl_loss(outputs_A, labels_A) * 0.5
                # loss_seg2 = self._pxl_loss(outputs_B, labels_B) * 0.5

                cretion1 = weighted_bce()
                loss_bn = cretion1(out_change, gt)
                loss_edge = cretion1(out_bd, gt_edge)
                criterion_sc = ChangeSimilarity().cuda()
                loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], gt)
                # uwl = AutomaticWeightedLoss(3)
                self.G_loss = self._pxl_loss(loss_bn, loss_edge, loss_sc)

            elif self.args.net_G == 'BIT_ResNet':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            elif self.args.net_G == 'BIT_ResNet1':
                self.G_loss = self._pxl_loss(self.G_pred, gt)

            else:
                self.G_loss = self._pxl_loss(self.G_pred[-1], gt)

        # 累加批次loss到epoch_loss（用于计算epoch平均loss）
        if self.is_training:
            self.epoch_loss += self.G_loss.item() * self.batch_size  # 乘以batch_size，后续除以总样本数

            # print(self.G_pred[-1].shape)
            # print(gt.shape)

        self.G_loss.backward()

    def print_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # 已分配显存（GB）
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3  # 峰值显存（GB）
            print(f"当前显存占用：{allocated:.2f}GB | 峰值显存：{max_allocated:.2f}GB")
            # 清理无用地张量（可选）
            torch.cuda.empty_cache()
            gc.collect()

    def _clean_batch_tensors(self):
        """清理当前batch的无用张量，释放显存"""
        # 1. 清空所有模型输出/中间张量（补充遗漏的张量）
        tensors_to_clear = [
            'multi_scale_preds', 'G_pred', 'pred_vis', 'batch', 'G_loss',
            'G_final_pred', 'G_pred_tuple', 'Combined_inputs', 'main_logits', 'aux_logits',
            'out', 'out2', 'preds', 'image_input', 'out_list', 'edge_list',
            'G_pred_List', 'Edge_pred', 'predge', 'out_change', 'outputs_A',
            'outputs_B', 'out_bd', 'gt', 'gt_edge', 'loss', 'ltotal', 'ledge'  # 补充遗漏
        ]
        for tensor_name in tensors_to_clear:
            if hasattr(self, tensor_name):
                delattr(self, tensor_name)

        # 2. 主动释放GPU缓存 + Python垃圾回收（增强）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 清理跨进程显存
        gc.collect()
        gc.collect()  # 两次GC确保清理干净

        # 3. 重置峰值显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)


    def train_models(self, mode='CNN_Tr'):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            
            # Ablation mode setting
            if self.args.net_G == 'BIE_EdgeNet':
                if mode=='CNN_Tr':
                    self.net_G.FE_IMD.set_test_mode('CNN',False)
                    self.net_G.FE_IMD.set_test_mode('Tr',True)
                    self.net_G.FE_IMD.set_test_mode('BIE',False)
                    self.net_G.FE_IMD.set_test_mode('Edge_Fusion',False)
                    self.net_G.CD_ED.set_Edge_mode(False)
                elif mode=='CNN_Tr_BIE':
                    self.net_G.FE_IMD.set_test_mode('CNN',False)
                    self.net_G.FE_IMD.set_test_mode('Tr',True)
                    self.net_G.FE_IMD.set_test_mode('BIE',True)
                    self.net_G.FE_IMD.set_test_mode('Edge_Fusion',False)
                    self.net_G.CD_ED.set_Edge_mode(False)
                elif mode=='CNN_Tr_Edge':
                    self.net_G.FE_IMD.set_test_mode('CNN',False)
                    self.net_G.FE_IMD.set_test_mode('Tr',True)
                    self.net_G.FE_IMD.set_test_mode('BIE',False)
                    self.net_G.FE_IMD.set_test_mode('Edge_Fusion',True)
                    self.net_G.CD_ED.set_Edge_mode(True)
                # elif mode=='CNN_BIE':
                #     self.net_G.FE_IMD.set_test_mode('CNN',True)
                #     self.net_G.FE_IMD.set_test_mode('Tr',False)
                #     self.net_G.FE_IMD.set_test_mode('BIE',True)
                #     self.net_G.FE_IMD.set_test_mode('Edge_Fusion',False)
                #     self.net_G.CD_ED.set_Edge_mode(False)
                # elif mode=='Tr_BIE':
                #     self.net_G.FE_IMD.set_test_mode('CNN',False)
                #     self.net_G.FE_IMD.set_test_mode('Tr',True)
                #     self.net_G.FE_IMD.set_test_mode('BIE',True)
                #     self.net_G.FE_IMD.set_test_mode('Edge_Fusion',False)
                #     self.net_G.CD_ED.set_Edge_mode(False)
                # elif mode=='Tr_Edge':
                #     self.net_G.FE_IMD.set_test_mode('CNN',False)
                #     self.net_G.FE_IMD.set_test_mode('Tr',True)
                #     self.net_G.FE_IMD.set_test_mode('BIE',False)
                #     self.net_G.FE_IMD.set_test_mode('Edge_Fusion',True)
                #     self.net_G.CD_ED.set_Edge_mode(True)
                # elif mode=='Tr_BIE_Edge':
                #     self.net_G.FE_IMD.set_test_mode('CNN',False)
                #     self.net_G.FE_IMD.set_test_mode('Tr',True)
                #     self.net_G.FE_IMD.set_test_mode('BIE',True)
                #     self.net_G.FE_IMD.set_test_mode('Edge_Fusion',True)
                #     self.net_G.CD_ED.set_Edge_mode(True)
                elif mode=='ALL':
                    self.net_G.FE_IMD.set_test_mode('CNN',True)
                    self.net_G.FE_IMD.set_test_mode('Tr',True)
                    self.net_G.FE_IMD.set_test_mode('BIE',True)
                    self.net_G.FE_IMD.set_test_mode('Edge_Fusion',True)
                    self.net_G.CD_ED.set_Edge_mode(True)
            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])

            # 计算训练集总样本数（用于后续求epoch平均loss）
            total_train_samples = len(self.dataloaders['train'].dataset)

            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                # update G
                self.optimizer_G.zero_grad()
                
                
                # with torch.cuda.amp.autocast(enabled=True):
                self._forward_pass(batch)
                self._backward_G()

                # self.scaler.scale(self.G_loss).backward()
                self.optimizer_G.step()
                # self.scaler.step(self.optimizer_G)
                # self.scaler.update()
                self._collect_running_batch_states()
                self._timer_update()



            # 计算训练epoch平均loss（总loss / 总样本数）
            self.epoch_loss = self.epoch_loss / total_train_samples
            # 收集并记录训练epoch指标
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            self._clean_batch_tensors()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            num_val = len(self.dataloaders['val'])
            # Iterate over data.
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['val'], 0), total=num_val):
                with torch.no_grad():
                    # with autocast(dtype=torch.bfloat16):
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

            self._clean_batch_tensors()
        # # -------------------------- 7. 训练结束后关闭SWANLab实验 --------------------------
        # self.swan_exp.finish()