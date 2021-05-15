import argparse
import itertools
import random
import time
import warnings
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from dataset.imbalance_cifar import ImbalanceCIFAR10
from utils import *
from matplotlib import cm
from sklearn.manifold import TSNE

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', choices=['inat', 'imagenet', 'cifar10', 'cifar100'])
parser.add_argument('--data_path', type=str, default='D:/dataset/CIFAR10')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str,
                    choices=['None', 'Resample', 'Reweight', 'DRW'])
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='pretrain_rot',
                    type=str, help='(additional) name to indicate which experiment it is')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=200, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='./checkpoint')
best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    print(args.store_name)
    # prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes,return_features=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    model = load_network(args, model)
    # Data loading code
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':

        train_dataset = ImbalanceCIFAR10(root='D:/dataset/CIFAR10', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_val)
        val_dataset = datasets.CIFAR10(root='D:/dataset/CIFAR10', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(root='D:/dataset', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    features_t, labels_t = extract_feature(args, train_loader, model)
    features_v, labels_v = extract_feature(args, val_loader, model)

    visualize(features_t, labels_t, 'train')
    visualize(features_v, labels_v, 'test')

def visualize(features, label, set):
    tsne = TSNE(perplexity=50, early_exaggeration=15, n_components=2, init='pca', n_iter=1000)  # TSNE降维，降到2
    # 只需要显示前500个
    # plot_only = 500
    # 降维后的数据
    low_dim_embs = tsne.fit_transform(features.numpy())
    # 标签
    labels = label.numpy()
    plt.cla()
    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    color_cycle = ["darkorange", "deeppink", "blue", "brown", "red", "dimgrey", "gold", "green", "darkturquoise","blueviolet"]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 / 9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        # plt.text(x, y, int(s), backgroundcolor=c, fontsize=1)
        plt.scatter(x, y, s=8, color=color_cycle[int(s)])
        # plt.scatter(x, y, s=8, c=c)
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())
    name = 'CE_1'+set + '.jpg'
    plt.savefig(name)

    # plt.savefig("{}.jpg".format(2))


def load_network(args, network):
    filename = 'D:/code/NIPS2021/模型数据备份/model/cifar10_resnet32_CE_None_exp_0.1_original/ckpt.pth.tar'
    # filename = 'checkpoint/cifar10_resnet32_CE_None_exp_0.05_rot/ckpt.pth.tar'
    # filename = 'checkpoint/cifar10_resnet32_CE_None_exp_0.05_/ckpt.pth.tar'
    # filename = 'checkpoint/cifar10_resnet32_CE_Resample_exp_0.01_backbone/ckpt.pth.tar' #pretrian+resample
    # filename = 'checkpoint/cifar10_resnet32_CE_None_exp_0.01_pretrain_rot/ckpt.pth.tar' # pretrain
    # filename = 'checkpoint/cifar10_resnet32_CE_None_exp_0.01_backbone/ckpt.pth.tar' #pretrian+all_backbone
    # filename = 'checkpoint/cifar10_resnet32_CE_Resample_exp_0.01_retrain_classifier/ckpt.pth.tar' #pretrian+all_backbone+resample_classifier

    print(filename)
    pretrained_dict = torch.load(filename)['state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print(torch.load(filename)['state_dict'].keys())
    # new_dict = {}
    # for k, v in pretrained_dict.items():
    #     new_dict.update({"module." + k: v})
    # network.load_state_dict(new_dict)
    network.load_state_dict(pretrained_dict)
    return network

def extract_feature(args, val_loader, model):
    # switch to evaluate mode
    model.eval()

    labels = torch.IntTensor()
    features = torch.FloatTensor()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if input.shape[0]<100:
                break
            ff = torch.FloatTensor(100, 64).zero_().cuda()
            tt = torch.IntTensor(100).zero_().cuda()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, feature = model(input)
            # print(feature.shape)
            ff += feature
            tt += target
            features = torch.cat((features, ff.data.cpu()), 0)
            labels = torch.cat((labels, tt.cpu()), 0)
            # print(labels.shape)
    return features, labels


if __name__ == '__main__':
    main()
