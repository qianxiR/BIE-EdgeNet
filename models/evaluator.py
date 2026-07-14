import os
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable as V
from datasets.data_utils import CDDataAugmentation
import torchvision.transforms.functional as TF
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2
from tqdm import tqdm
from osgeo import gdal,ogr,osr

from heatmap import generate_heatmap
from save_result import save_binary_label
from color_compare import save_tp_fp_tn_fn_vis
from att_vis import *

import torch
from torch.utils.data import TensorDataset, DataLoader  # 核心缺失的导入
import torch.nn.functional as F
import gc

gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # 512MB缓存
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
gdal.SetConfigOption('VSI_CACHE_SIZE', '1000000')


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader,test_name=None):

        self.dataloader = dataloader
        self.test_name = test_name

        self.n_class = args.n_class
        # define G
        self.loss = args.loss
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.net_name = args.net_G
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_acc_id = 0
        self.best_val_f1 = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            if checkpoint_name=='best_acc_ckpt.pt':
                self.logger.write('loading best ACC checkpoint...\n')
            elif checkpoint_name=='best_ckpt.pt':
                self.logger.write('loading best F1 checkpoint...\n')
            elif checkpoint_name=='last_ckpt.pt':
                self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device, weights_only=False)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_acc_id = checkpoint['best_epoch_acc_id']
            self.best_val_f1 = checkpoint['best_val_f1']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\nEval Historical_best_f1 = %.4f (at epoch %d)' %
                  (self.best_val_acc, self.best_epoch_acc_id, self.best_val_f1, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        # G_pred = torch.argmax(G_pred, dim=1)
        #
        # current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        if self.n_class == 2:
            if self.loss == 'BCEDiceLoss' and self.net_name == 'ChangeViT':
                G_pred_prob = torch.sigmoid(G_pred)  # (16,1,256,256)
                G_pred = (G_pred_prob > 0.7).float()

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.loss == 'AERNet_Loss' and self.net_name == 'AERNet':
                G_pred = torch.where(torch.sigmoid(G_pred) > 0.5, 1, 0)

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            elif self.loss == 'RCDT_MultiScale_Loss' and self.net_name == 'RCDT':
                G_pred_prob = F.softmax(G_pred, dim=1)[:, 1, :, :]
                G_pred = (G_pred_prob > 0.5).float()
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.net_name == 'HSANet':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.net_name == 'B2CNet':
                G_pred = F.sigmoid(G_pred[:, 1, :, :])
                G_pred[G_pred >= 0.5] = 1
                G_pred[G_pred < 0.5] = 0

                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy().astype(np.int64),
                                                              gt=target.cpu().numpy().astype(np.int64))

            elif self.net_name == 'ChangeDINO' or self.net_name == 'BIE_ChangeDINO':
                # torch.argmax返回指定维度最大值的序号,如果c=1，那么所有的序号都是0
                G_pred = torch.argmax(G_pred.detach(), dim=1)
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            elif self.net_name == 'EGRCNN':
                # torch.argmax返回指定维度最大值的序号,如果c=1，那么所有的序号都是0
                G_pred = torch.argmax(G_pred, dim=1)
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(),
                                                              gt=target.cpu().numpy())

            elif self.net_name == 'EGPNet':
                G_pred = torch.argmax(G_pred, dim=1).long()
                # F1
                current_score = self.running_metric.update_cm(pr=G_pred.detach().cpu().numpy(),
                                                              gt=target.detach().cpu().numpy())

            elif self.net_name == 'EATDer':
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



        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 100) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        # if self.loss == 'BCEDiceLoss' and self.net_name == 'ChangeViT':
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        # else:
        #     vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['F1_1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        if self.loss == 'BCEDiceLoss' and self.net_name == 'ChangeViT':
            self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.loss == 'AERNet_Loss' and self.net_name == 'AERNet':
            self.G_pred = self.net_G(img_in1, img_in2)[-1]

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'RCDT':
            _, self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'VcT':
            self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'HSANet':
            self.G_pred_tuple = self.net_G(img_in1, img_in2)
            self.G_pred = torch.cat(self.G_pred_tuple, dim=1)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'B2CNet':
            self.G_pred, _ = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'ChangeDINO' or self.net_name == 'BIE_ChangeDINO':
            self.G_pred, _ = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'ChangeMamba':
            self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'CTDFormer':
            self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'EGRCNN':
            self.image_input = torch.stack([img_in1, img_in2], axis=0)
            self.out_list, self.edge_list = self.net_G(self.image_input)
            d6_out, d5_out, d4_out, d3_out, d2_out = self.out_list
            self.G_pred = d2_out

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'EGPNet':
            self.G_pred_List, self.Edge_pred = self.net_G(img_in1, img_in2)
            self.G_pred = self.G_pred_List[0]

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'EATDer':
            self.Edge_pred, self.G_pred = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                save_binary_label(pred_result, img_name, save_root=self.test_name)

                gt_binary = batch['L'][idx]
                save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'BIT_ResNet':
            self.G_pred, feat1_batch, feat2_batch = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                # 提取单张图像的特征（去除batch维度）
                feat1 = feat1_batch[idx]  # [32, H, W]
                feat2 = feat2_batch[idx]  # [32, H, W]

                feat1_1 = visualize_global_avg_attention(feat1, 64, 64)
                feat2_2 = visualize_global_avg_attention(feat2, 64, 64)

                # 3. 生成并保存时相1/时相2的特征热力图
                heatmap1_path = f"{self.test_name}/feat1_att/{img_name}_feat1.jpg"
                heatmap2_path = f"{self.test_name}/feat2_att/{img_name}_feat2.jpg"
                generate_heatmap(feat1_1, method="channel_weight", save_path=heatmap1_path)
                generate_heatmap(feat2_2, method="channel_weight", save_path=heatmap2_path)

                # （可选）生成特征差分的热力图（对比双时相特征差异）
                feat_diff = torch.abs(feat1 - feat2)
                heatmap_diff_path = f"{self.test_name}/diff_att/{img_name}_diff.jpg"
                generate_heatmap(feat_diff, method="channel_weight", save_path=heatmap_diff_path)

                # pred_result = self.G_pred[idx]
                # save_binary_label(pred_result, img_name, save_root=self.test_name)

                # gt_binary = batch['L'][idx]
                # save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        elif self.net_name == 'BIT_ResNet1':
            self.G_pred, feat1_batch, feat2_batch, feat1_ori, feat2_ori = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                # 提取单张图像的特征（去除batch维度）
                feat1 = feat1_batch[idx]  # [32, H, W]
                feat2 = feat2_batch[idx]  # [32, H, W]
                feat1_without = feat1_ori[idx]
                feat2_without = feat2_ori[idx]

                # 3. 生成并保存时相1/时相2的特征热力图
                heatmap1_path = f"{self.test_name}/feat1/{img_name}_feat1.jpg"
                heatmap2_path = f"{self.test_name}/feat2/{img_name}_feat2.jpg"
                heatmap1_without_path = f"{self.test_name}/feat1_without/{img_name}_feat1_without.jpg"
                heatmap2_without_path = f"{self.test_name}/feat2_without/{img_name}_feat2_without.jpg"
                generate_heatmap(feat1, method="channel_weight", save_path=heatmap1_path)
                generate_heatmap(feat2, method="channel_weight", save_path=heatmap2_path)
                generate_heatmap(feat1_without, method="channel_weight", save_path=heatmap1_without_path)
                generate_heatmap(feat2_without, method="channel_weight", save_path=heatmap2_without_path)

                # feat_diff = torch.abs(feat1 - feat2)
                # heatmap_diff_path = f"{self.test_name}/diff/{img_name}_diff.jpg"
                # generate_heatmap(feat_diff, method="channel_weight", save_path=heatmap_diff_path)

                # pred_result = self.G_pred[idx]
                # save_binary_label(pred_result, img_name, save_root=self.test_name)
                #
                # gt_binary = batch['L'][idx]
                # save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)

        else:
            # self.G_pred = self.net_G(img_in1, img_in2)[-1]
            self.edge, self.G_pred, self.edge_c = self.net_G(img_in1, img_in2)

            for idx in range(len(batch['name'])):
                img_name = batch['name'][idx]
                pred_result = self.G_pred[idx]
                edge_result = self.edge[idx]
                edge_c = self.edge_c[idx]

                edge_c_path = f"{self.test_name}/wihtout_edge_c/{img_name}.jpg"
                generate_heatmap(edge_c, method="channel_weight", save_path=edge_c_path)


                # save_binary_label(pred_result, img_name, save_root=self.test_name)
                # save_binary_label(edge_result, img_name + 'edge', save_root=self.test_name)

                # gt_binary = batch['L'][idx]
                # save_tp_fp_tn_fn_vis(pred_result, gt_binary, img_name, save_root=self.test_name)


    def eval_models(self,checkpoint_name='last_ckpt.pt', mode='CNN_Tr'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        # self.net_G.set_test_mode(True)
        # Ablation mode setting
        if self.net_name == 'BIE_EdgeNet':
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
            elif mode=='Edge':
                self.net_G.FE_IMD.set_test_mode('CNN',True)
                self.net_G.FE_IMD.set_test_mode('Tr',True)
                self.net_G.FE_IMD.set_test_mode('BIE',True)
                self.net_G.FE_IMD.set_test_mode('Edge_Fusion',False)
                self.net_G.CD_ED.set_Edge_mode(False)

        num_test = len(self.dataloader)
        # Iterate over data.
        for self.batch_id, batch in tqdm(enumerate(self.dataloader, 0), total=num_test):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()

    def block_gdal_input(self, img, img_size, crop=512, pad=0): # gdal分块读取
        [img_width, img_height] = img_size
        x_height = x_width = crop
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad

        numBand = 3
        # numBand = img.RasterCount
        num_Xblock = img_width // crop_width
        x_start, x_end = [], []
        x_start.append(0)
        for i in range(num_Xblock):
            xs = crop_width * (i + 1) - pad
            xe = crop_width * i + x_width - pad
            if (i == num_Xblock - 1):
                xs = img_width - crop_width - pad
                xe = min(xe, img_width)
            x_start.append(xs)
            x_end.append(xe)
        x_end.append(img_width)

        num_Yblock = img_height // crop_height
        y_start, y_end = [], []
        y_start.append(0)
        for i in range(num_Yblock):
            ys = crop_height * (i + 1) - pad
            ye = crop_height * i + x_height - pad
            if (i == num_Yblock - 1):
                ys = img_height - crop_height - pad
                ye = min(ye, img_height)
            y_start.append(ys)
            y_end.append(ye)
        y_end.append(img_height)

        if img_width % crop_width > 0:
            num_Xblock = num_Xblock + 1
        if img_height % crop_height > 0:
            num_Yblock = num_Yblock + 1
        for i in range(num_Yblock):
            for j in range(num_Xblock):
                [x0, x1, y0, y1] = [x_start[j], x_end[j], y_start[i], y_end[i]]

                feature = np.zeros(np.append([y1 - y0, x1 - x0], numBand), np.float32)
                for ii in range(numBand):
                    floatData = np.array(img.GetRasterBand(ii + 1).ReadAsArray(x0, y0, x1 - x0, y1 - y0),dtype=np.float32)
                    # floatData = np.array(img.GetRasterBand(4-ii).ReadAsArray(x0,y0,x1-x0,y1-y0))

                    feature[..., ii] = (floatData/255-0.5)/0.5
                    # feature[..., ii] = floatData

                if (i == 0):
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     pad, x_height - pad - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     0, x_height - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                if (j == 0):
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, pad, x_width - pad - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, 0, x_width - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)

                yield feature_pad, [x0, x1, y0, y1]

    def pred_gdal_blocks_write(self, img_pathA, img_pathB,out_path=''):
        self._load_checkpoint()

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        #logger.info('predicting %s' % img_pathA)

        batch_size = 1
        pad = 16
        x_width = 256
        x_height = 256
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad
        datasetname = gdal.Open(img_pathA, gdal.GA_ReadOnly)
        # datasetname = reproject_dataset(img_path,5500,5500)
        if datasetname is None:
            print('Could not open %s' % img_pathA)
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]
        nBand = datasetname.RasterCount

        datasetname2 = gdal.Open(img_pathB, gdal.GA_ReadOnly)
        if datasetname2 is None:
            print('Could not open %s' % img_pathB)
        img_width2 = datasetname2.RasterXSize
        img_height2 = datasetname2.RasterYSize

        if img_width != img_width2 or img_height != img_height2:
            print("范围不一致")
            return

        driver = gdal.GetDriverByName('GTiff')
        if out_path == '':
            out_path = img_pathA.rsplit('.', 1)[0] + '_res.tif'
        outRaster = driver.Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
        outband = outRaster.GetRasterBand(1)
        outRaster.SetGeoTransform(datasetname.GetGeoTransform())
        outRaster.SetProjection(datasetname.GetProjection())

        num_Xblock = img_width // crop_width
        if img_width % crop_width > 0:
            num_Xblock += 1
        num_Yblock = img_height // crop_height
        if img_height % crop_height > 0:
            num_Yblock += 1
        i = 0
        blocks = num_Xblock * num_Yblock
        # mask = np.zeros([batch_size, img_height, img_width],dtype=np.float32)
        input_gen = self.block_gdal_input(datasetname, imageSize, x_width, pad)
        input_gen2 = self.block_gdal_input(datasetname2, imageSize, x_width, pad)
        for i in tqdm(range(blocks)):
            imgA, xy = next(input_gen)
            imgB, xyB = next(input_gen2)
            if (xy[0] > 0):
                xs = xy[0] + pad
            else:
                xs = xy[0]

            if (xy[2] > 0):
                ys = xy[2] + pad
            else:
                ys = xy[2]
            #if np.max(imgA[pad: pad + crop_height, pad: pad + crop_width]) < 5:
                #predictions = np.zeros([batch_size, x_height, x_width])
            #else:
            imgs = []
            imgs.append(imgA)
            imgs = np.array(imgs)
            # imgs = imgs[:,np.newaxis]
            # np.squeeze(imgs)
            # np.expand_dims(imgs,axis=1)
            imgs = imgs.transpose(0, 3, 1, 2)
            imgs = V(torch.Tensor(np.array(imgs, np.float32)).to(self.device))

            # imgs = TF.normalize(imgs, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # imgs = imgs.resize(1,3,256,256)

            imgs2 = []
            imgs2.append(imgB)
            imgs2 = np.array(imgs2)
            # imgs = imgs[:,np.newaxis]
            # np.squeeze(imgs)
            # np.expand_dims(imgs,axis=1)
            imgs2 = imgs2.transpose(0, 3, 1, 2)
            imgs2 = V(torch.Tensor(np.array(imgs2, np.float32)).to(self.device))
            # imgs2 = TF.to_tensor(np.array(imgs2[0], np.float32)).to(self.device)
            # imgs2 = TF.normalize(imgs2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # imgs2 = imgs2.resize(1, 3, 256, 256)

            predictions = self.net_G(imgs, imgs2)[-1]
            # predictions = predictions.numpy()
            predictions = torch.argmax(predictions, dim=1, keepdim=True)
            # print(predictions)
            predictions = np.array(predictions)[0][0]

            prediction = predictions[pad: pad + crop_height,
                         pad: pad + crop_width]

            outband.WriteArray((prediction * 255).astype(np.int), xs, ys)
            # mask[0,ys: ys+crop_height,\
            #    xs : xs+crop_width] = prediction.astype(np.float32)

            # if(i%num_Xblock==0):
            #    y=i//num_Xblock
            #    logger.info('predicting data: [{}{}] {}%'.\
            #                 format('=' * (y+1),
            #                        ' ' * (num_Yblock - y-1),
            #                        100 * (y+1)/num_Yblock))
            outband.FlushCache()
            # sys.stdout.flush()
            # i=i+1
        datasetname = None
        datasetname2 = None
        outRaster = None
        return  # np.squeeze(mask*255).astype(np.int)

    def _process_model_output(self, pred_tensor):
        """统一处理不同模型的输出，转换为二值变化检测结果（0/1）"""
        if self.n_class == 2:
            if self.loss == 'BCEDiceLoss' and self.net_name == 'ChangeViT':
                pred_prob = torch.sigmoid(pred_tensor)
                pred = (pred_prob > 0.7).float()
            elif self.loss == 'AERNet_Loss' and self.net_name == 'AERNet':
                pred = torch.where(torch.sigmoid(pred_tensor) > 0.5, 1.0, 0.0).float()
            elif self.loss == 'RCDT_MultiScale_Loss' and self.net_name == 'RCDT':
                pred_prob = torch.softmax(pred_tensor, dim=1)[:, 1, :, :].unsqueeze(1)
                pred = (pred_prob > 0.5).float()
            elif self.net_name == 'HSANet' or self.net_name == 'B2CNet':
                pred_prob = torch.sigmoid(pred_tensor[:, 1, :, :]).unsqueeze(1)
                pred = (pred_prob >= 0.5).float()
            elif self.net_name == 'ChangeDINO' or self.net_name == 'BIE_ChangeDINO' or self.net_name == 'EGRCNN' or self.net_name == 'EGPNet':
                pred = torch.argmax(pred_tensor, dim=1, keepdim=True).float()
            elif self.net_name == 'EATDer':
                pred_prob = torch.sigmoid(pred_tensor)
                pred = torch.where(pred_prob > 0.5, 1.0, 0.0).float()
            else:
                pred = torch.argmax(pred_tensor, dim=1, keepdim=True).float()
        else:
            pred = torch.argmax(pred_tensor, dim=1, keepdim=True).float()
        return pred

    def _load_image_with_gdal(self, img_path):
        """读取大影像，返回数据、地理变换、投影、尺寸信息"""
        ds = gdal.Open(img_path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"无法打开影像文件: {img_path}")

        # 读取影像数据
        img_width = ds.RasterXSize
        img_height = ds.RasterYSize
        img_bands = ds.RasterCount

        # 读取所有波段并拼接 [H, W, C]
        img_data = []
        for b in range(img_bands):
            band_data = ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float32)
            img_data.append(band_data)
        img_data = np.stack(img_data, axis=-1)

        # 归一化（和训练时保持一致: (x/255 - 0.5)/0.5）
        img_data = (img_data / 255.0 - 0.5) / 0.5

        # 获取地理信息
        geotrans = ds.GetGeoTransform()
        proj = ds.GetProjection()

        ds = None  # 释放资源
        return img_data, geotrans, proj, img_width, img_height

    def _generate_crop_coords(self, img_width, img_height, crop_size=256, pad_size=16):
        """生成裁剪块的坐标，处理边缘填充（确保覆盖整个影像）"""
        # 计算填充后的尺寸
        pad_width = img_width + 2 * pad_size
        pad_height = img_height + 2 * pad_size

        # 计算裁剪块数量（向上取整，确保覆盖全部）
        num_x = (pad_width + crop_size - 1) // crop_size
        num_y = (pad_height + crop_size - 1) // crop_size

        new_width = num_x * crop_size
        new_height = num_y * crop_size

        # 生成裁剪坐标（确保不越界）
        coords = []
        for y in range(num_y):
            for x in range(num_x):
                x0 = x * crop_size
                y0 = y * crop_size
                x1 = min(x0 + crop_size, new_width)
                y1 = min(y0 + crop_size, new_height)
                coords.append((x0, y0, x1, y1))

        # 原始影像在填充后的有效区域
        valid_region = (pad_size, pad_size, pad_size + img_width, pad_size + img_height)

        return coords, new_width, new_height, valid_region

    def _load_gdal_block_safe(self, img_path, x_off, y_off, x_size, y_size):
        """安全的GDAL分块读取：严格检查边界，避免越界"""
        ds = gdal.Open(img_path, gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(f"无法打开影像文件: {img_path}")

        # 获取影像真实尺寸
        img_width = ds.RasterXSize
        img_height = ds.RasterYSize

        # 修正读取窗口，确保不越界
        y1 = min(y_off + y_size, img_height)
        x1 = min(x_off + x_size, img_width)
        read_height = y1 - y_off
        read_width = x1 - x_off

        if read_height <= 0 or read_width <= 0:
            ds = None
            raise ValueError(
                f"读取窗口无效: ({x_off},{y_off}) 尺寸 {x_size}×{y_size}，影像尺寸 {img_width}×{img_height}")

        # 读取指定块数据（安全读取）
        band_data = []
        for b in range(ds.RasterCount):
            band = ds.GetRasterBand(b + 1)
            # 安全读取：只读取有效区域
            data = band.ReadAsArray(x_off, y_off, read_width, read_height).astype(np.float32)
            band_data.append(data)

        # 堆叠波段并补零到目标尺寸（保持尺寸一致）
        img_block = np.stack(band_data, axis=-1)
        if img_block.shape[0] < y_size or img_block.shape[1] < x_size:
            pad_h = y_size - img_block.shape[0]
            pad_w = x_size - img_block.shape[1]
            img_block = cv2.copyMakeBorder(
                img_block, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )

        # 归一化（和训练保持一致）
        img_block = (img_block / 255.0 - 0.5) / 0.5

        ds = None
        return img_block

    def _generate_tile_array_optimized(self, img_block, overlap_length, tile_size):
        """优化的子块生成：更小的子块+批量友好"""
        h, w = img_block.shape[:2]

        # 减小大块尺寸，提升GPU利用率（原2048→1024）
        step = tile_size - overlap_length
        num_rows = max(1, (h + step - 1) // step)
        num_cols = max(1, (w + step - 1) // step)

        # 生成子块（预分配内存）
        tile_array = []
        for r in range(num_rows):
            row_tiles = []
            y0 = r * step
            y1 = min(y0 + tile_size, h)
            for c in range(num_cols):
                x0 = c * step
                x1 = min(x0 + tile_size, w)
                tile = img_block[y0:y1, x0:x1, :]

                # 快速补零到tile_size（向量化操作）
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = np.pad(
                        tile,
                        ((0, tile_size - tile.shape[0]),
                         (0, tile_size - tile.shape[1]),
                         (0, 0)),
                        mode='constant'
                    )
                row_tiles.append(tile)
            tile_array.append(row_tiles)

        return tile_array, num_rows, num_cols

    def _merge_tiles_optimized(self, tile_preds, tile_size, overlap_length, img_shape):
        """
        优化的无缝拼接函数：
        1. 增大重叠区域，确保完全覆盖
        2. 使用加权平均，避免边缘效应
        3. 精确裁剪，消除缝隙
        """
        h, w = img_shape[:2]
        step = tile_size - overlap_length

        # 初始化累加数组和权重数组（用于加权平均）
        result = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 生成一个中心权重核（中心权重高，边缘权重低）
        def create_weight_kernel(size, overlap):
            kernel = np.ones((size, size), dtype=np.float32)
            # 对重叠区域应用线性衰减
            if overlap > 0:
                # 顶部和底部重叠区域
                kernel[:overlap, :] = np.linspace(0, 1, overlap)[:, np.newaxis]
                kernel[-overlap:, :] = np.linspace(1, 0, overlap)[:, np.newaxis]
                # 左侧和右侧重叠区域
                kernel[:, :overlap] *= np.linspace(0, 1, overlap)[np.newaxis, :]
                kernel[:, -overlap:] *= np.linspace(1, 0, overlap)[np.newaxis, :]
            return kernel

        weight_kernel = create_weight_kernel(tile_size, overlap_length)

        # 逐行逐列拼接
        for r in range(len(tile_preds)):
            for c in range(len(tile_preds[r])):
                pred = tile_preds[r][c]
                if pred is None:
                    continue

                # 计算子块在大图中的原始位置
                y0 = r * step
                x0 = c * step
                y1 = y0 + tile_size
                x1 = x0 + tile_size

                # 计算在大图中的有效区域（不越界）
                y0_img = max(y0, 0)
                x0_img = max(x0, 0)
                y1_img = min(y1, h)
                x1_img = min(x1, w)

                # 计算在子块中的对应裁剪区域
                y0_tile = y0_img - y0
                x0_tile = x0_img - x0
                y1_tile = y1_img - y0
                x1_tile = x1_img - x0

                # 裁剪预测块和权重核
                pred_crop = pred[y0_tile:y1_tile, x0_tile:x1_tile]
                weight_crop = weight_kernel[y0_tile:y1_tile, x0_tile:x1_tile]

                # 加权累加
                result[y0_img:y1_img, x0_img:x1_img] += pred_crop * weight_crop
                weight[y0_img:y1_img, x0_img:x1_img] += weight_crop

        # 安全平均（避免除0）
        weight[weight == 0] = 1
        result = result / weight

        return result

    @torch.no_grad()
    def predict_large_image(self,
                            img_pathA,
                            img_pathB,
                            label_path=None,
                            batch_size=16,  # 匹配训练batch_size，最大化GPU利用率
                            crop_size=256,
                            pad_size=16,
                            out_pred_path='',
                            checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        self.net_G.eval()
        self._clear_cache()

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.net_G = self.net_G.to(self.device, non_blocking=True)
            self.net_G = self.net_G.float()
            self.logger.write(f"[OK] GPU加速已启用: {self.device}\n")
            self.logger.write(f"[INFO] 模型数据类型: float32 (匹配输入类型)\n")
        else:
            self.logger.write("[WARN] 未检测到GPU，使用CPU运行（速度极慢）\n")

        self.logger.write(f"\n===== 开始预测大影像 =====\n")
        self.logger.write(f"时相A: {img_pathA}\n")
        self.logger.write(f"时相B: {img_pathB}\n")

        # 获取影像基础信息
        ds = gdal.Open(img_pathA)
        total_width = ds.RasterXSize
        total_height = ds.RasterYSize
        geotrans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        ds = None

        # 优化分块参数
        block_height = 1024
        overlap_length = pad_size
        nb_blocks = (total_height + block_height - 1) // block_height

        self.logger.write(f"影像信息：{total_width}×{total_height}，分块数：{nb_blocks}\n")
        self.logger.write(f"大块高度：{block_height}，子块尺寸：{crop_size}，批量大小：{batch_size}\n")

        # 初始化最终结果
        final_result = np.zeros((total_height, total_width), dtype=np.uint8)

        # 按行分大块处理
        pbar = tqdm(range(nb_blocks), desc="处理影像大块", unit="块")
        for block_idx in pbar:
            # 计算当前块坐标（边界检查）
            start_y = block_idx * block_height
            this_block_h = min(block_height, total_height - start_y)

            # 加载高度
            load_height = this_block_h + overlap_length
            if block_idx < nb_blocks - 1:
                load_height = min(load_height, block_height)  # 关键：限制加载高度不超过block_height

            try:
                # 安全读取双时相影像块
                imgA_block = self._load_gdal_block_safe(img_pathA, 0, start_y, total_width, load_height)
                imgB_block = self._load_gdal_block_safe(img_pathB, 0, start_y, total_width, load_height)
            except Exception as e:
                self.logger.write(f"\n[ERROR] 读取块{block_idx}失败: {str(e)}\n")
                pbar.update(1)
                continue

            # 生成子块数组
            tile_array_A, num_rows, num_cols = self._generate_tile_array_optimized(
                imgA_block, overlap_length, crop_size
            )
            tile_array_B, _, _ = self._generate_tile_array_optimized(
                imgB_block, overlap_length, crop_size
            )

            # 展平子块用于批量预测
            flat_tiles_A = []
            flat_tiles_B = []
            tile_positions = []
            for r in range(num_rows):
                for c in range(num_cols):
                    flat_tiles_A.append(tile_array_A[r][c])
                    flat_tiles_B.append(tile_array_B[r][c])
                    tile_positions.append((r, c))

            # 批量预测
            tile_preds = [[None for _ in range(num_cols)] for _ in range(num_rows)]

            # 分批次处理
            for batch_start in range(0, len(flat_tiles_A), batch_size):
                batch_end = min(batch_start + batch_size, len(flat_tiles_A))
                batch_A = flat_tiles_A[batch_start:batch_end]
                batch_B = flat_tiles_B[batch_start:batch_end]
                batch_pos = tile_positions[batch_start:batch_end]

                if len(batch_A) == 0:
                    continue

                # 转换为张量
                tensors_A = []
                tensors_B = []
                for a, b in zip(batch_A, batch_B):
                    tensor_A = torch.from_numpy(a.transpose(2, 0, 1)).float()
                    tensor_B = torch.from_numpy(b.transpose(2, 0, 1)).float()

                    # 32倍数填充
                    pad_h = (32 - tensor_A.shape[1] % 32) % 32
                    pad_w = (32 - tensor_A.shape[2] % 32) % 32
                    if pad_h > 0 or pad_w > 0:
                        tensor_A = F.pad(tensor_A, (0, pad_w, 0, pad_h), mode='reflect')
                        tensor_B = F.pad(tensor_B, (0, pad_w, 0, pad_h), mode='reflect')

                    tensors_A.append(tensor_A)
                    tensors_B.append(tensor_B)

                # 堆叠并异步传输到GPU
                batch_A_tensor = torch.stack(tensors_A).to(self.device, non_blocking=True)
                batch_B_tensor = torch.stack(tensors_B).to(self.device, non_blocking=True)

                # 模型推理
                with torch.inference_mode():
                    if self.net_name in ['ChangeDINO', 'BIE_ChangeDINO']:
                        pred, _ = self.net_G(batch_A_tensor, batch_B_tensor)
                    else:
                        pred = self.net_G(batch_A_tensor, batch_B_tensor)[-1]

                    # 去除填充
                    pred = pred[..., :crop_size, :crop_size]
                    pred_binary = self._process_model_output(pred)
                    pred_np = pred_binary.squeeze(1).cpu().numpy()

                # 保存预测结果
                for i, (r, c) in enumerate(batch_pos):
                    if r < num_rows and c < num_cols:
                        tile_preds[r][c] = pred_np[i]

                # 强制清理显存（异步）
                del batch_A_tensor, batch_B_tensor, pred, pred_binary, pred_np
                torch.cuda.empty_cache()

            # 拼接子块
            block_result = self._merge_tiles_optimized(
                tile_preds, crop_size, overlap_length, imgA_block.shape
            )

            # 保存到最终结果
            save_height = min(this_block_h, block_result.shape[0])
            if save_height > 0:
                final_result[start_y:start_y + save_height, :total_width] = (
                            block_result[:save_height, :total_width] > 0.5).astype(np.uint8)

            # 清理内存
            del imgA_block, imgB_block, tile_array_A, tile_array_B, tile_preds, block_result
            gc.collect()

            mem_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            pbar.set_postfix({'显存占用(MB)': f"{mem_usage:.1f}"})

        pbar.close()

        # 保存结果
        if out_pred_path == '':
            out_pred_path = os.path.splitext(img_pathA)[0] + '_pred.tif'

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_pred_path, total_width, total_height, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(geotrans)
        out_ds.SetProjection(proj)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(final_result * 255)
        out_band.SetNoDataValue(0)
        out_band.FlushCache()
        out_ds = None

        self.logger.write(f"\n[OK] 预测完成！结果保存至: {out_pred_path}\n")
        self.logger.write(f"[INFO] 最终结果尺寸: {final_result.shape}\n")

        return final_result
