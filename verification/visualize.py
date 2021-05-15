import argparse
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
from imbalance_cifar import IMBALANCECIFAR10
from utils import *
from matplotlib import cm
from sklearn.manifold import TSNE

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--ls', default=False, type=bool, help='use label smooth')
parser.add_argument('--verify', default=False, type=bool, help='using siamese network')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of dataload workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str])
    print(args.store_name)
    prepare_folders(args)
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
    model = models.__dict__[args.arch](num_classes=num_classes)

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

        train_dataset = IMBALANCECIFAR10(root='D:/dataset/CIFAR10', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.rand_number, train=True, download=True,
                                         transform=transform_val)
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

    features, labels = extract_feature(args, train_loader, model)
    visualize(features, labels)

def visualize(features, label):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)  # TSNE降维，降到2
    # 只需要显示前500个
    # plot_only = 500
    # 降维后的数据
    low_dim_embs = tsne.fit_transform(features.numpy())
    # 标签
    labels = label.numpy()
    plt.cla()
    X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 / 9 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.text(x, y, int(s), backgroundcolor=c, fontsize=1)
        # plt.scatter(x, y, s=8, c=c)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max());
    plt.title('Visualize last layer')
    plt.savefig("noise_train.jpg")

    # plt.savefig("{}.jpg".format(2))


def load_network(args, network):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)

    network.load_state_dict(torch.load(filename)['state_dict'])
    return network

def extract_feature(args, val_loader, model):
    # switch to evaluate mode
    model.eval()

    labels = torch.IntTensor()
    features = torch.FloatTensor()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, _) in enumerate(val_loader):
            if input.shape[0]<100:
                break
            ff = torch.FloatTensor(100, 64).zero_().cuda()
            tt = torch.FloatTensor(100).zero_().cuda()
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, _,feature = model(input)
            # print(feature.shape)
            ff += feature
            tt += target
            features = torch.cat((features, ff.data.cpu()), 0)
            labels = torch.cat((labels, tt.cpu()), 0)
            # print(labels.shape)
    return features, labels


if __name__ == '__main__':
    main()
