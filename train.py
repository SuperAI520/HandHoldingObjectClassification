import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.nn import DataParallel
from model.mobilefacenet import MobileFaceNet
from model.resnet import ResNet50
from model.cbam import CBAMResNet
from model.attention import ResidualAttentionNet_56, ResidualAttentionNet_92
from margin.ArcMarginProduct import ArcMarginProduct
from margin.MultiMarginProduct import MultiMarginProduct
from margin.CosineMarginProduct import CosineMarginProduct
from margin.SphereMarginProduct import SphereMarginProduct
from margin.InnerProduct import InnerProduct
from utils.visualize import Visualizer
from utils.logging import init_log
from dataloader.casia_webface import CASIAWebFace


def calculate_accuracy(predicted, target):
    num_data = target.size()[0]
    predicted = torch.argmax(predicted, dim=1)
    correct_pred = torch.sum(predicted == target)

    accuracy = correct_pred*(100/num_data)

    return accuracy.item()

def train(args):
    # gpu init
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(args.save_dir, args.model_pre + args.backbone.upper() + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataloader loader
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    # validation dataloader
    trainset = CASIAWebFace(args.train_root, args.train_file_list, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4, drop_last=False)
    # # test dataloader
    testset = CASIAWebFace(args.test_root, args.test_file_list, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4, drop_last=False)

    # define backbone and margin layer
    if args.backbone == 'MobileFace':
        net = MobileFaceNet(feature_dim=args.feature_dim)
    elif args.backbone == 'Res50':
        net = ResNet50()
    elif args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Attention_56':
        net = ResidualAttentionNet_56(feature_dim=args.feature_dim)
    elif args.backbone == 'Attention_92':
        net = ResidualAttentionNet_92(feature_dim=args.feature_dim)
    else:
        print(args.backbone, ' is not available!')

    if args.margin_type == 'ArcFace':
        margin = ArcMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'MultiMargin':
        margin = MultiMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'CosFace':
        margin = CosineMarginProduct(args.feature_dim, trainset.class_nums, s=args.scale_size)
    elif args.margin_type == 'Softmax':
        margin = InnerProduct(args.feature_dim, trainset.class_nums)
    elif args.margin_type == 'SphereFace':
        margin = SphereMarginProduct(args.feature_dim, trainset.class_nums)
    else:
        print(args.margin_type, 'is not available!')

    if args.resume:
        print('resume the model parameters from: ', args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70, 120, 160], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)

    best_acc = 0.0
    best_iters = 0
    total_iters = 0
    vis = Visualizer(env=args.model_pre + args.backbone)
    for epoch in range(1, args.total_epoch + 1):
        # train model
        _print(f"Train Epoch: {epoch}/{args.total_epoch} ...")
        net.train()

        since = time.time()
        for data in trainloader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()

            raw_logits = net(img)
            output = margin(raw_logits, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1
            # print train information
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                vis.plot_curves({'softmax loss': total_loss.item()}, iters=total_iters, title='train loss',
                                xlabel='iters', ylabel='train loss')
                vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters',
                                ylabel='train accuracy')

                _print(f"Iters: {total_iters:0>6d}/[{epoch:0>2d}], loss: {total_loss.item():.4f}, train_accuracy: "
                       f"{correct/total:.4f}, time: {time_cur:.2f} s/iter, learning rate: {exp_lr_scheduler.get_lr()[0]}")

            # save model
            if total_iters % args.save_freq == 0:
                msg = f'Saving checkpoint: {total_iters}'
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict},
                    os.path.join(save_dir, f'Iter_{total_iters:06}_net.ckpt'))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict},
                    os.path.join(save_dir, f'Iter_{total_iters:06}_margin.ckpt'))

            # test accuracy
            if total_iters % args.test_freq == 0:
                net.eval()

                epoch_accuracy = []

                for test_data in testloader:
                    test_img, test_label = test_data[0].to(device), test_data[1].to(device)
                    
                    raw_logits = net(test_img)
                    test_output = margin(raw_logits, test_label)
                    
                    _, test_predict = torch.max(test_output.data, 1)
                    total = test_label.size(0)
                    batch_accuracy = (np.array(test_predict.cpu()) == np.array(test_label.data.cpu())).sum() / total
                    epoch_accuracy.append(batch_accuracy)

                _print(f'Ave Accuracy: {np.mean(epoch_accuracy) * 100:.4f}')
                acc = np.mean(epoch_accuracy) * 100
                if best_acc <= acc:
                    best_acc = acc
                    best_iters = total_iters

                _print(f'Current Best Accuracy: {best_acc:.4f} in iters: {best_iters}')

                vis.plot_curves({'test': np.mean(epoch_accuracy)}, iters=total_iters,
                                title='test accuracy', xlabel='iters', ylabel='test accuracy')
                net.train()
            
        exp_lr_scheduler.step()
    _print(f'Finally Best Accuracy: {best_acc:.4f} in iters: {best_iters}')
    print('finishing training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for image classification')
    parser.add_argument('--train_root', type=str, default='datasets/train/', help='train image root')
    parser.add_argument('--train_file_list', type=str, default='datasets/train/train.list', help='train list')
    parser.add_argument('--test_root', type=str, default='datasets/test/', help='test image root')
    parser.add_argument('--test_file_list', type=str, default='datasets/test/test.list', help='test file list')

    parser.add_argument('--backbone', type=str, default='Attention_92', help='MobileFace, Res50_IR, SERes50_IR, Res100_IR, SERes100_IR, Attention_56, Attention_92')
    parser.add_argument('--margin_type', type=str, default='Softmax', help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float, default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=200, help='total epochs')

    parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
    parser.add_argument('--test_freq', type=int, default=100, help='test frequency')
    parser.add_argument('--resume', type=int, default=False, help='resume model')
    parser.add_argument('--net_path', type=str, default='./checkpoints/resnet50_Iter_486000_net.ckpt', help='net weight')
    parser.add_argument('--margin_path', type=str, default='./checkpoints/resnet50_Iter_48600_margin.ckpt', help='margin weight')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='Attention_92_', help='model prefix')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    train(args)
