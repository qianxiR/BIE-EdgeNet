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

from torch.cuda.amp import autocast


# Decide which device we want to run on
class CDEvaluator_fp16():

    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # 指标计算器
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # 日志配置
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        # 训练状态记录
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

        # 创建目录
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.vis_dir):
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading best checkpoint...\n')
            checkpoint = torch.load(
                os.path.join(self.checkpoint_dir, checkpoint_name),
                map_location=self.device,
                weights_only=False
            )
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # 更新最佳状态
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_acc_id = checkpoint['best_epoch_acc_id']
            self.best_val_f1 = checkpoint['best_val_f1']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.logger.write(
                'Eval Historical_best_acc = %.4f (at epoch %d)\nEval Historical_best_f1 = %.4f (at epoch %d)' %
                (self.best_val_acc, self.best_epoch_acc_id, self.best_val_f1, self.best_epoch_id)
            )
            self.logger.write('\n')
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score


    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.dataloader)

        # 日志打印
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # 可视化保存
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        vis_pred = utils.make_numpy_grid(self._visualize_pred())
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)


    def _collect_epoch_states(self):
        scores_dict = self.running_metric.get_scores()
        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)
        self.epoch_acc = scores_dict['F1_1']

        # 保存分数文件
        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)), mode='a') as file:
            pass

        # 日志记录
        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)
        self.logger.write('\n')


    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        # 模型推理（混合精度在调用处控制，此处仅执行推理）
        self.G_pred = self.net_G(img_in1, img_in2)[-1]


    def eval_models(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # -------------------------- 2. 核心修改：常规评估启用混合精度 --------------------------
        # 嵌套 torch.no_grad()（禁止梯度）和 autocast（FP16推理）
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad(), autocast():  # 关键：混合精度上下文
                self._forward_pass(batch)  # 模型推理（自动转为FP16）
            self._collect_running_batch_states()
        self._collect_epoch_states()


    def block_gdal_input(self, img, img_size, crop=512, pad=0): # gdal分块读取
        [img_width, img_height] = img_size
        x_height = x_width = crop
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad

        numBand = 3
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
                    floatData = np.array(
                        img.GetRasterBand(ii + 1).ReadAsArray(x0, y0, x1 - x0, y1 - y0),
                        dtype=np.float32
                    )
                    feature[..., ii] = (floatData/255 - 0.5)/0.5  # 归一化

                # 边界填充
                if (i == 0):
                    feature_pad = cv2.copyMakeBorder(
                        feature, pad, x_height - pad - feature.shape[0], 0, 0,
                        cv2.BORDER_REFLECT_101
                    )
                else:
                    feature_pad = cv2.copyMakeBorder(
                        feature, 0, x_height - feature.shape[0], 0, 0,
                        cv2.BORDER_REFLECT_101
                    )
                if (j == 0):
                    feature_pad = cv2.copyMakeBorder(
                        feature_pad, 0, 0, pad, x_width - pad - feature_pad.shape[1],
                        cv2.BORDER_REFLECT_101
                    )
                else:
                    feature_pad = cv2.copyMakeBorder(
                        feature_pad, 0, 0, 0, x_width - feature_pad.shape[1],
                        cv2.BORDER_REFLECT_101
                    )

                yield feature_pad, [x0, x1, y0, y1]


    def pred_gdal_blocks_write(self, img_pathA, img_pathB, out_path=''):
        self._load_checkpoint()
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # GDAL配置
        batch_size = 1
        pad = 16
        x_width = 256
        x_height = 256
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad

        # 读取输入图像
        datasetname = gdal.Open(img_pathA, gdal.GA_ReadOnly)
        if datasetname is None:
            print('Could not open %s' % img_pathA)
            return
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]

        datasetname2 = gdal.Open(img_pathB, gdal.GA_ReadOnly)
        if datasetname2 is None:
            print('Could not open %s' % img_pathB)
            return
        if img_width != datasetname2.RasterXSize or img_height != datasetname2.RasterYSize:
            print("双时相图像范围不一致")
            return

        # 创建输出图像
        driver = gdal.GetDriverByName('GTiff')
        if out_path == '':
            out_path = img_pathA.rsplit('.', 1)[0] + '_res.tif'
        outRaster = driver.Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
        outband = outRaster.GetRasterBand(1)
        outRaster.SetGeoTransform(datasetname.GetGeoTransform())
        outRaster.SetProjection(datasetname.GetProjection())

        # 分块推理
        num_Xblock = img_width // crop_width + (1 if img_width % crop_width > 0 else 0)
        num_Yblock = img_height // crop_height + (1 if img_height % crop_height > 0 else 0)
        blocks = num_Xblock * num_Yblock
        input_gen = self.block_gdal_input(datasetname, imageSize, x_width, pad)
        input_gen2 = self.block_gdal_input(datasetname2, imageSize, x_width, pad)

        # -------------------------- 3. 核心修改：GDAL分块推理启用混合精度 --------------------------
        for i in tqdm(range(blocks)):
            imgA, xy = next(input_gen)
            imgB, xyB = next(input_gen2)

            # 计算块的起始位置（处理边界）
            xs = xy[0] + pad if xy[0] > 0 else xy[0]
            ys = xy[2] + pad if xy[2] > 0 else xy[2]

            # 图像预处理（转为模型输入格式）
            # 时相1
            imgs = np.array([imgA]).transpose(0, 3, 1, 2)  # (1,3,256,256)
            imgs = V(torch.Tensor(imgs.astype(np.float32)).to(self.device))
            # 时相2
            imgs2 = np.array([imgB]).transpose(0, 3, 1, 2)  # (1,3,256,256)
            imgs2 = V(torch.Tensor(imgs2.astype(np.float32)).to(self.device))

            # 混合精度推理（关键：禁止梯度+FP16）
            with torch.no_grad(), autocast():  # 嵌套上下文，降低显存
                predictions = self.net_G(imgs, imgs2)[-1]  # 模型推理（FP16）

            # 预测结果后处理
            predictions = torch.argmax(predictions, dim=1, keepdim=True)  # (1,1,256,256)
            predictions = np.array(predictions)[0][0]  # (256,256)
            # 裁剪掉边界填充，保留有效区域
            prediction = predictions[pad: pad + crop_height, pad: pad + crop_width]

            # 写入输出图像
            outband.WriteArray((prediction * 255).astype(np.int), xs, ys)
            outband.FlushCache()

        # 释放资源
        datasetname = None
        datasetname2 = None
        outRaster = None
        return