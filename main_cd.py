from argparse import ArgumentParser
import torch
from models.trainer import *
from models.trainer_FP16 import *
import os
import data_config
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import warnings
warnings.filterwarnings('ignore')
# print(torch.cuda.is_available())



"""
the main function for training the CD networks
"""


def train(args, mode=''):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    # model = CDTrainer_fp16(args=args, dataloaders=dataloaders)
    model.train_models(mode=mode)


def test(args, mode=''):
    from models.evaluator import CDEvaluator
    from models.run_evaluator import CDEvaluator_param

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test', dataset=args.dataset)
    test_root = data_config.DataConfig().get_data_config(args.data_name).root_dir
    test_project = args.project_name
    model = CDEvaluator(args=args, dataloader=dataloader, test_name=os.path.join(test_root, test_project))
    # model = CDEvaluator_fp16(args=args, dataloader=dataloader)
    # model = CDEvaluator_param(args=args, dataloader=dataloader, test_name=os.path.join(test_root, test_project))

    model.eval_models(checkpoint_name='best_ckpt.pt', mode=mode)
    # model.eval_models(checkpoint_name='best_acc_ckpt.pt', mode=mode)
    # model.eval_models(checkpoint_name='last_ckpt.pt', mode=mode)

    # model.predict_large_image(
    # img_pathA="E:\CDCD\zyhCD20160607\Level18\zyhCD20160607.tif",
    # img_pathB="E:\CDCD\zyhCD20230308\Level18\zyhCD20230308.tif",
    # label_path="E:\CDCD\CDCDLabel.tif",
    # batch_size=args.batch_size,
    # crop_size=args.img_size,
    # pad_size=64,
    # out_pred_path="E:\CDCD\output_pred.tif",
    # checkpoint_name="best_ckpt.pt"
# )


def test2(args):
    from models.evaluator import CDEvaluator

    model = CDEvaluator(args=args, dataloader='')
    model.pred_gdal_blocks_write(r'E:\bianhuajiance\cq\fengjiexian\nanbu\2020\500115_clip4.tif',
                                 r'E:\bianhuajiance\cq\fengjiexian\nanbu\2021\500115_clip4.tif')


if __name__ == '__main__':
    # ------------
    # args
    # ------------


    parser3 = ArgumentParser()
    parser3.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser3.add_argument('--project_name', default='BIE_EdgeNet-CDCD-repair-7-1-2', type=str)
    parser3.add_argument( '--checkpoint_root', default='checkpoints', type=str)
    parser3.add_argument('--vis_root', default='vis', type=str)
    
    # data
    parser3.add_argument('--num_workers', default=0, type=int)
    parser3.add_argument('--dataset', default='CDDataset', type=str)
    parser3.add_argument('--data_name', default='CDCD-repair-7-1-2', type=str)
    
    parser3.add_argument('--batch_size', default=16, type=int)
    parser3.add_argument('--split', default="train", type=str)
    parser3.add_argument('--split_val', default="val", type=str)
    
    parser3.add_argument('--img_size', default=256, type=int)
    parser3.add_argument('--shuffle_AB', default=False, type=str)
    
    # model4
    parser3.add_argument('--n_class', default=2, type=int)
    parser3.add_argument('--embed_dim', default=32, type=int)
    parser3.add_argument('--pretrain', default=None, type=str)
    parser3.add_argument('--multi_scale_train', default=False, type=str)
    parser3.add_argument('--multi_scale_infer', default=False, type=str)
    parser3.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])
    
    parser3.add_argument('--net_G', default='BIE_EdgeNet', type=str,
                         help='base_resnet18 | base_transformer_pos_s4 | '
                              'base_transformer_pos_s4_dd8 | '
                              'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    parser3.add_argument('--loss', default='eas', type=str)
    # parser3.add_argument('--loss', default='ce', type=str)
    # parser3.add_argument('--loss', default='MambaBCDLoss', type=str)
    
    # optimizer
    parser3.add_argument('--optimizer', default='adamw', type=str)
    parser3.add_argument('--lr', default=0.0001, type=float)
    parser3.add_argument('--max_epochs', default=200, type=int)
    parser3.add_argument('--lr_policy', default='linear', type=str,
                         help='linear | step')
    parser3.add_argument('--lr_decay_iters', default=100, type=int)
    
    args3 = parser3.parse_args()
    utils.get_device(args3)
    # print(args.gpu_ids)
    
    #  checkpoints dir
    args3.checkpoint_dir = os.path.join(args3.checkpoint_root, args3.project_name)
    os.makedirs(args3.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args3.vis_dir = os.path.join(args3.vis_root, args3.project_name)
    os.makedirs(args3.vis_dir, exist_ok=True)
    
    train(args3, mode='ALL')
    test(args3, mode='ALL')


    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='[Ablation]CD_CNN_Tr_BIE-CDCD-repair-7-1-2', type=str)

    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CDCD-repair-7-1-2', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model4
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=32, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='BIE_EdgeNet', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    # parser.add_argument('--loss', default='eas', type=str)
    parser.add_argument('--loss', default='ce', type=str)
    # parser.add_argument('--loss', default='DINO_Loss', type=str)
    # parser.add_argument('--loss', default='AERNet_Loss', type=str)
    # parser.add_argument('--loss', default='HSANet_Loss', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # train(args, mode='CNN_Tr_BIE')
    test(args, mode='Edge')
    #
    # parser2 = ArgumentParser()
    # parser2.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser2.add_argument('--project_name', default='BIE_Conv_ChangeDINO-LEVIR-256-edge', type=str)
    #
    # parser2.add_argument('--checkpoint_root', default='checkpoints', type=str)
    # parser2.add_argument('--vis_root', default='vis', type=str)
    #
    # # data
    # parser2.add_argument('--num_workers', default=0, type=int)
    # parser2.add_argument('--dataset', default='CDDataset', type=str)
    # parser2.add_argument('--data_name', default='LEVIR-256-edge', type=str)
    #
    # parser2.add_argument('--batch_size', default=16, type=int)
    # parser2.add_argument('--split', default="train", type=str)
    # parser2.add_argument('--split_val', default="val", type=str)
    #
    # parser2.add_argument('--img_size', default=256, type=int)
    # parser2.add_argument('--shuffle_AB', default=False, type=str)
    #
    # # model4
    # parser2.add_argument('--n_class', default=2, type=int)
    # parser2.add_argument('--embed_dim', default=32, type=int)
    # parser2.add_argument('--pretrain', default=None, type=str)
    # parser2.add_argument('--multi_scale_train', default=False, type=str)
    # parser2.add_argument('--multi_scale_infer', default=False, type=str)
    # parser2.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])
    #
    # parser2.add_argument('--net_G', default='BIE_ChangeDINO', type=str,
    #                     help='base_resnet18 | base_transformer_pos_s4 | '
    #                          'base_transformer_pos_s4_dd8 | '
    #                          'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    # # parser2.add_argument('--loss', default='eas', type=str)
    # # parser2.add_argument('--loss', default='ce', type=str)
    # parser2.add_argument('--loss', default='MambaBCDLoss', type=str)
    # # parser.add_argument('--loss', default='RCDT_MultiScale_Loss', type=str)
    #
    # # optimizer
    # parser2.add_argument('--optimizer', default='adamw', type=str)
    # parser2.add_argument('--lr', default=0.0001, type=float)
    # parser2.add_argument('--max_epochs', default=200, type=int)
    # parser2.add_argument('--lr_policy', default='linear', type=str,
    #                     help='linear | step')
    # parser2.add_argument('--lr_decay_iters', default=100, type=int)
    #
    # args2 = parser2.parse_args()
    # utils.get_device(args2)
    # # print(args.gpu_ids)
    #
    # #  checkpoints dir
    # args2.checkpoint_dir = os.path.join(args2.checkpoint_root, args2.project_name)
    # os.makedirs(args2.checkpoint_dir, exist_ok=True)
    # #  visualize dir
    # args2.vis_dir = os.path.join(args2.vis_root, args2.project_name)
    # os.makedirs(args2.vis_dir, exist_ok=True)
    #
    # # train(args2, mode='CNN_Tr_Edge')
    # # test(args2, mode='CNN_Tr_Edge')
    #
    # parser1 = ArgumentParser()
    # parser1.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser1.add_argument('--project_name', default='BIE_Conv_ChangeDINO-CDCD-repair-7-1-2', type=str)
    # parser1.add_argument( '--checkpoint_root', default='checkpoints', type=str)
    # parser1.add_argument('--vis_root', default='vis', type=str)
    #
    # # data
    # parser1.add_argument('--num_workers', default=0, type=int)
    # parser1.add_argument('--dataset', default='CDDataset', type=str)
    # parser1.add_argument('--data_name', default='CDCD-repair-7-1-2', type=str)
    #
    # parser1.add_argument('--batch_size', default=16, type=int)
    # parser1.add_argument('--split', default="train", type=str)
    # parser1.add_argument('--split_val', default="val", type=str)
    #
    # parser1.add_argument('--img_size', default=256, type=int)
    # parser1.add_argument('--shuffle_AB', default=False, type=str)
    #
    # # model4
    # parser1.add_argument('--n_class', default=2, type=int)
    # parser1.add_argument('--embed_dim', default=32, type=int)
    # parser1.add_argument('--pretrain', default=None, type=str)
    # parser1.add_argument('--multi_scale_train', default=False, type=str)
    # parser1.add_argument('--multi_scale_infer', default=False, type=str)
    # parser1.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])
    #
    # parser1.add_argument('--net_G', default='BIE_ChangeDINO', type=str,
    #                      help='base_resnet18 | base_transformer_pos_s4 | '
    #                           'base_transformer_pos_s4_dd8 | '
    #                           'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    # # parser1.add_argument('--loss', default='eas', type=str)
    # # parser1.add_argument('--loss', default='ce', type=str)
    # parser1.add_argument('--loss', default='MambaBCDLoss', type=str)
    #
    # # optimizer
    # parser1.add_argument('--optimizer', default='adamw', type=str)
    # parser1.add_argument('--lr', default=0.0001, type=float)
    # parser1.add_argument('--max_epochs', default=200, type=int)
    # parser1.add_argument('--lr_policy', default='linear', type=str,
    #                      help='linear | step')
    # parser1.add_argument('--lr_decay_iters', default=100, type=int)
    #
    # args1 = parser1.parse_args()
    # utils.get_device(args1)
    # # print(args.gpu_ids)
    #
    # #  checkpoints dir
    # args1.checkpoint_dir = os.path.join(args1.checkpoint_root, args1.project_name)
    # os.makedirs(args1.checkpoint_dir, exist_ok=True)
    # #  visualize dir
    # args1.vis_dir = os.path.join(args1.vis_root, args1.project_name)
    # os.makedirs(args1.vis_dir, exist_ok=True)
    #
    # # train(args1, mode='ALL')
    # test(args1, mode='ALL')
    #
    #
    # parser4 = ArgumentParser()
    # parser4.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser4.add_argument('--project_name', default='BIE_Conv_ChangeDINO-WHU-256-edge-7-1-2', type=str)
    # parser4.add_argument( '--checkpoint_root', default='checkpoints', type=str)
    # parser4.add_argument('--vis_root', default='vis', type=str)
    #
    # # data
    # parser4.add_argument('--num_workers', default=0, type=int)
    # parser4.add_argument('--dataset', default='CDDataset', type=str)
    # parser4.add_argument('--data_name', default='WHU-256-edge-7-1-2', type=str)
    #
    # parser4.add_argument('--batch_size', default=16, type=int)
    # parser4.add_argument('--split', default="train", type=str)
    # parser4.add_argument('--split_val', default="val", type=str)
    #
    # parser4.add_argument('--img_size', default=256, type=int)
    # parser4.add_argument('--shuffle_AB', default=False, type=str)
    #
    # # model4
    # parser4.add_argument('--n_class', default=2, type=int)
    # parser4.add_argument('--embed_dim', default=32, type=int)
    # parser4.add_argument('--pretrain', default=None, type=str)
    # parser4.add_argument('--multi_scale_train', default=False, type=str)
    # parser4.add_argument('--multi_scale_infer', default=False, type=str)
    # parser4.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])
    #
    # parser4.add_argument('--net_G', default='BIE_ChangeDINO', type=str,
    #                      help='base_resnet18 | base_transformer_pos_s4 | '
    #                           'base_transformer_pos_s4_dd8 | '
    #                           'base_transformer_pos_s4_dd8_dedim8|ChangeFormerV5|SiamUnet_diff')
    # # parser4.add_argument('--loss', default='eas', type=str)
    # # parser4.add_argument('--loss', default='ce', type=str)
    # parser4.add_argument('--loss', default='MambaBCDLoss', type=str)
    #
    # # optimizer
    # parser4.add_argument('--optimizer', default='adamw', type=str)
    # parser4.add_argument('--lr', default=0.0001, type=float)
    # parser4.add_argument('--max_epochs', default=200, type=int)
    # parser4.add_argument('--lr_policy', default='linear', type=str,
    #                      help='linear | step')
    # parser4.add_argument('--lr_decay_iters', default=100, type=int)
    #
    # args4 = parser4.parse_args()
    # utils.get_device(args4)
    # # print(args.gpu_ids)
    #
    # #  checkpoints dir
    # args4.checkpoint_dir = os.path.join(args4.checkpoint_root, args4.project_name)
    # os.makedirs(args4.checkpoint_dir, exist_ok=True)
    # #  visualize dir
    # args4.vis_dir = os.path.join(args4.vis_root, args4.project_name)
    # os.makedirs(args4.vis_dir, exist_ok=True)
    #
    # # train(args4, mode='ALL')
    # # test(args4, mode='ALL')
