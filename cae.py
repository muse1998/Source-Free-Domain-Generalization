import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip
import utils
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import tsne, a_distance
from autoencoder import auto_encoder
from prompts import get_prompts

from PIL import ImageFile
import warnings
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_all(input, output, num_class, num_style, device,lamda1, lamda2):
    """
    The total loss of CAE to generate domain unified prompt representations
    """
    # reconstruction loss
    loss_rec = input @ output.t() 
    eye = torch.eye(loss_rec.shape[0]).to(device)
    loss_rec  = loss_rec  * eye
    loss_rec = -1 * loss_rec.sum()/loss_rec.shape[0]
    # intra-class loss
    output = output.view(num_class, num_style, -1)
    output_mean = torch.mean(output, dim=1).unsqueeze(1).repeat(1,num_style,1)
    loss_intra = -1 * torch.sum(output * output_mean, dim=2).mean()
    # inter-class loss
    loss_inter = 0
    output = output.transpose(0,1)
    for style in range(num_style):
        temp = output[style] @ output[style].t()
        temp = temp * (1-torch.eye(num_class).to(device))
        loss_inter += (temp.sum() / (num_class*(num_class-1)))
    loss_inter = loss_inter/num_style
    loss_all =  loss_rec+ lamda1 * loss_intra + lamda2 * loss_inter

    return loss_all

def encode_t(model, args, device):
    """
    Encode prompts (class_num x domain_num)
    """
    flag = 0
    prompts, class_num, domain_num = get_prompts(args.data, args.bank_type)
    with torch.no_grad():
        for prompt in prompts:
            text = clip.tokenize(prompt).to(device)
            text_features = model.encode_text(text).float()
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            if flag==0:
                # text features have to be placed on cpu beacuse of the limitation of gpu memorys.
                text_features_all = text_features.to('cpu')
                flag = 1
            else:
                text_features_all =  torch.cat((text_features_all, text_features.to('cpu')), dim=0)
    return text_features_all.to(device), class_num, domain_num

def train(cae, model, args, device, optimizer, text_features, class_num, domain_num):
    """
    training process of CAE using text features (no image is involved) 
    """
    model.train()
    out_features = cae(text_features)
    loss = loss_all(text_features, out_features, class_num, domain_num, device,args.intra,args.inter)
    print('loss_all: {:.3f} '.format(loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test(cae, model, args, device, optimizer, text_features, class_num, domain_num, test_loader):
    """
    inference process of the method on testing set
    """
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        # the reshaping is for mean pooling on the domain dimension
        in_features = text_features
        out_features = cae(in_features)
        text_features_rec = out_features.view(class_num, domain_num, -1)
        # mean pooling to generate domain unified prompt representations for each class
        text_features_rec = torch.mean(text_features_rec, dim=1)
        end = time.time()
        for i, (images, target, _) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            image_features = model.encode_image(images).float()

            # the normalization process of image features
            # the normalization process of text features is done in function autoencoder (class AutoEncoder).
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # calculate cosine similarity between the image representations and the domain unified prompt representations for inference
            output = 100 * image_features @ text_features_rec.t()
            
            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
    return top1.avg


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # switch backbones of CLIP
    if args.arch == 'vitb16':
        model, preprocess = clip.load("ViT-B/16", device=device)
    elif args.arch == 'vitb32':
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif args.arch == 'resnet50':
        model, preprocess = clip.load("RN50", device=device)
    
    # init CAE
    autoencoder = auto_encoder(in_shape=model.text_projection.shape[1]).to(device)

    optimizer = AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=1e-4)

    # Data loading code

    test_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',
                                        download=True, transform=preprocess, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print("test_dataset_size: ", len(test_dataset))

    # start training
    best_test_acc1 = 0.
    
    # encode prompts (class_num x domain_num)
    text_features, class_num, domain_num = encode_t(model, args, device)

    for epoch in range(args.epochs):
        # train CAE with no image data
        train(autoencoder, model, args, device, optimizer, text_features, class_num, domain_num)

        # evaluate on testing set
        print("Evaluate on test set...")
        best_test_acc1 = max(best_test_acc1, test(autoencoder, model, args, device, optimizer, text_features, class_num, domain_num, test_loader))
        print(best_test_acc1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: PACS)')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vitb16')
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--freeze-bn', action='store_true', help='whether freeze all bn layers')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='only activated when freeze-bn is True')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--bank_type", type=str, default='Combined')
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--intra', default=0.5, type=float, help='weight of loss intra')
    parser.add_argument('--inter', default=0.05, type=float, help='weight of loss intra')
    args = parser.parse_args()
    main(args)
