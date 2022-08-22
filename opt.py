# encoding: UTF-8
# author: Anzhu Yu
# data: 20210421 16:22

# 为了方便控制参数，我们将参数放在这里，方便后续修改
import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    # 数据部分
    parser.add_argument('--root_dir', type=str,
                        default= "D:\\程序\\Datasets\\Change Detection\\",
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='WHU-Area',
                        choices=['WHU-Area', 'WHU-Sat'],
                        help='which dataset to train/val')
    # 网络部分
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=2,
                        help='number of gpus')
    # parser.add_argument('--num_gpus', type=int, default=2,
    #                     help='number of gpus')
    # 是否读取权值
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    parser.add_argument('--out_dir', type=str, default='',
                        help='output path')

    # 下面不一定有用，我暂时保留了
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()
