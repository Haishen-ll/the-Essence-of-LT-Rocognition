import argparse
import random
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from utils import *
from dataset.imbalance_cifar import ImbalanceCIFAR10, ImbalanceCIFAR100
from dataset.imbalance_svhn import ImbalanceSVHN
from losses import LDAMLoss, FocalLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn'])
parser.add_argument('--data_path', type=str, default='D:/dataset/CIFAR10')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--loss_type', default="CE", type=str, choices=['CE', 'Focal', 'LDAM'])
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, choices=['None', 'Resample', 'Reweight', 'DRW'])
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='original', type=str, help='(additional) name to indicate experiment')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=200, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
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
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, which can slow down training considerably! '
                      'You may see unexpected behavior when restarting from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    print(f"===> Creating model '{args.arch}'")
    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset in {'cifar10', 'svhn'}:
        num_classes = 10
    else:
        raise NotImplementedError
    use_norm = True if args.loss_type == 'LDAM' else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # load self-supervised pre-trained model
    if args.pretrained:
        print("load parameters from pretrained model")
        filename = 'checkpoint/cifar10_resnet32_CE_None_exp_0.1_pretrain_rot/ckpt.pth.tar'
        # filename = 'checkpoint/simsiam/simsiam-cifar10-resnet32.pth'
        pretrained_dict = torch.load(filename)['state_dict']

        # print(pretrained_dict)
        # print(model.state_dict().keys())
        new_dict ={}
        for k, v in pretrained_dict.items():
            new_dict.update({"module."+k: v})

        # print(new_dict.keys())
        # pretrained_dict.pop('module.fc.weight')
        # pretrained_dict.pop('module.fc.bias')
        new_dict.pop('module.fc.weight')
        new_dict.pop('module.fc.bias')
        model_dict = model.state_dict()
        # print(new_dict)

        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    mean = [0.4914, 0.4822, 0.4465] if args.dataset.startswith('cifar') else [.5, .5, .5]
    std = [0.2023, 0.1994, 0.2010] if args.dataset.startswith('cifar') else [.5, .5, .5]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.dataset == 'cifar10':
        train_dataset = ImbalanceCIFAR10(
            root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
            rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=args.data_path,
                                       train=False, download=True, transform=transform_val)
        train_sampler = None
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'cifar100':
        train_dataset = ImbalanceCIFAR100(
            root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
            rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root=args.data_path,
                                        train=False, download=True, transform=transform_val)
        train_sampler = None
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'svhn':
        train_dataset = ImbalanceSVHN(
            root=args.data_path, imb_type=args.imb_type, imb_factor=args.imb_factor,
            rand_number=args.rand_number, split='train', download=True, transform=transform_train)
        val_dataset = datasets.SVHN(root=args.data_path,
                                    split='test', download=True, transform=transform_val)
        train_sampler = None
        if args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported!")

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Checkpoint '{args.resume}' loaded, testing...")
        validate(val_loader, model, nn.CrossEntropyLoss(), 0, args)
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"No checkpoint found at '{args.resume}'")



    cudnn.benchmark = True

    if args.dataset.startswith(('cifar', 'svhn')):
        cls_num_list = train_dataset.get_cls_num_list()
        print('cls num list:')
        print(cls_num_list)
        args.cls_num_list = cls_num_list

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == 'Reweight':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            per_cls_weights = None

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % best_acc1
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        target = target.cuda()
        output = model(inputs)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            output = model(inputs)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
            flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        if tf_writer is not None:
            tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
            tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
            tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


if __name__ == '__main__':
    main()
