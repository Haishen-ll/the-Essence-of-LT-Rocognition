import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import models
from tensorboardX import SummaryWriter
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from loss.labelsmooth import LabelSmoothingCrossEntropy
from autoaugment import CIFAR10Policy, Cutout

import math

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--root_log',type=str, default='./logs')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-mb', '--meta-batch', default=10, type=int, metavar='N', help='meta-mini-batch size')
parser.add_argument('--mix_ratio', default=0.1, type=float, help='imbalance factor')

best_acc1 = 0

def main():
    args = parser.parse_args()


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')


    ngpus_per_node = torch.cuda.device_count()



    args.store_name = '_'.join([args.dataset, args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str,  str(args.mix_ratio)])
    prepare_folders(args)
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'tiny':
        num_classes = 200
    else:
        pass
    model = models.__dict__[args.arch](num_classes=num_classes)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer_model = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    # Data loading code
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),    # add AutoAug
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='D:/dataset/CIFAR10', imb_type=args.imb_type, imb_factor=args.imb_factor, mix_ratio = args.mix_ratio, rand_number = args.rand_number, train = True, download = True, transform = transform_train)
        test_dataset = datasets.CIFAR10(root='D:/dataset/CIFAR10', train=False, download=True, transform=transform_test)
        # test_dataset =
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='/home/public_data/liulei/cifar100/', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='/home/public_data/liulei/cifar100/', train=False, download=True, transform=transform_test)
    else:
        warnings.warn('Dataset is not listed')
        return

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:', cls_num_list)
    args.cls_num_list = cls_num_list


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)


    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer_model, epoch, args)
        criterion = nn.CrossEntropyLoss().cuda()
        # train for one epoch
        train(num_classes, train_dataset, train_loader, model, criterion, optimizer_model, epoch, args, log_training, tf_writer)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, epoch, args, log_testing, tf_writer)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer_model.state_dict(),
        }, is_best)


def train(num_classes, train_dataset, train_loader, model, criterion, model_opt, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        model_opt.zero_grad()

        outputs = model(input)
        loss = criterion(outputs, target)

        # measure accuracy and record loss
        # print(outputs.shape)
        acc1, acc5 = accuracy(outputs, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))

        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        model_opt.step()

        # measure elapsed time
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
            data_time=data_time, loss=losses, top1=top1, top5=top5, lr=model_opt.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', model_opt.param_groups[-1]['lr'], epoch)


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
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
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
        out_cls_acc = '%s Class Accuracy: %s' % (flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_' + flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i): x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr





if __name__ == '__main__':
    main()
    # a = torch.tensor([[[1, 2, 3, 4],
    #             [1, 2, 3, 4]]]).float()
    # a0 = torch.norm(a, p=2, dim=0)  #
    # a1 = torch.norm(a, p=2, dim=-1)  #
    # print(a.shape)
    # print(a0)
    # print(a1)    # print(a1)