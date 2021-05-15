import numpy as np
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
x = [[1300, 20, 80, 0.45],[948, 22.5, 69, 2],
[1444, 23, 57, 0.5],[1440, 21.5, 79,2.4],
[786, 26.5, 64, 1.5],[1084, 28.5, 59, 3],
[1652, 23, 84, 0.4],[1844, 26, 73, 1],
[1756, 29.5, 72, 0.9],[1116, 35, 92, 2.8],
[1754, 30, 76, 0.8],[1656, 20, 83, 1.45],
[1200, 22.5, 69, 1.8],[1536, 23, 57, 1.5],
[1500, 21.8, 77, 0.6],[960, 24.8, 67, 1.5]]

y = [[0.066],[0.005],[0.076],[0.011],[0.001],[0.003],
[0.170],[0.140],[0.156],[0.039],[0.120],[0.059],
[0.040],[0.087],[0.120],[0.039]]
clf.fit(x,y)
test_x = np.array([[1436, 28, 68, 2]], dtype='float32')
print(clf.predict(test_x))

from sklearn import tree

X=[[0,0,0],[0,0,1],[1,0,0],
   [2,0,0],[2,1,0],[2,1,1],
   [1,1,1],[0,0,0],[0,1,0],
   [2,1,0],[0,1,1],[1,0,1],
   [1,1,0],[2,0,1] ]
Y=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

data = tree.export_graphviz(clf,feature_names=['Outlook','Humidity','Windy'],class_names=['NotPlay','Play'])



import numpy as np
from sklearn.linear_model import LogisticRegression

Model = LogisticRegression()
x = [[20, 45, 2], [120, 46, 4],
    [90, 55, 10],[81, 56, 19],
    [200, 55, 8]]
y = [1,1,0,0,0]
Model.fit(x,y)

test1 = np.array([[90, 60, 8]])
test2 = np.array([[80, 50, 10]])

print(Model.predict(test1))
print(Model.predict(test2))



import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def rotation(inputs):
    batch = inputs.shape[0]
    target = torch.Tensor(np.random.permutation([0, 1, 2, 3] * (int(batch / 4) + 1)), device=inputs.device)[:batch]
    target = target.long()
    image = torch.zeros_like(inputs)
    image.copy_(inputs)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(inputs[i, :, :, :], target[i], [1, 2])

    return image, target


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def adjust_learning_rate(optimizer, epoch, args):
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 160:
        lr = args.lr * 0.01
    elif epoch > 180:
        lr = args.lr * 0.0001
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ImbalancedTripletSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


def calc_confusion_mat(val_loader, model, args, save_path, name):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes, normalize=True, title=args.confusion_title)
    plt.savefig(os.path.join(save_path, 'LDAM_'+name+'_confusion_matrix.png'))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = [str(i) for i in range(10)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.title(title, fontsize=18)
    plt.xlabel('Predicted label', fontsize=17)
    plt.ylabel('True label', fontsize=17)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    norm = 1000 if normalize else 1
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] / norm, fmt),
                    ha="center", va="center",
                    color="black")  # color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return ax


def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f'Creating folder: {folder}')
            os.mkdir(folder)


def save_checkpoint(args, state, is_best):
    filename = f'{args.root_model}/{args.store_name}/ckpt.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
