import os
import argparse
import time
import cv2
from PIL import Image, ImageDraw
from datetime import datetime
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
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



def preprocess_image(pil_im, resize_im=True):
    # mean and std list for channels (Imagenet)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((112, 112), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_ten.requires_grad_()
    im_as_ten = im_as_ten.to(device)
    return im_as_ten


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = ['Holding Nothing', 'Holding Object']
    feature_dim = 512
    scale_size = 32.0

    ori_img = cv2.imread(args.source)
    src_img = ori_img

    if src_img is None:
        print("Cannot read image!")
        return
    
    src_img = cv2.resize(src_img, (112, 112))

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0, 1.0]
    ])

    src_img = test_transform(src_img)
    src_img = torch.unsqueeze(src_img, 0).to(device)

    if args.backbone == 'MobileFace':
        net = MobileFaceNet(feature_dim=feature_dim)
    elif args.backbone == 'Res50':
        net = ResNet50()
    elif args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=feature_dim, mode='ir_se')
    elif args.backbone == 'Attention_56':
        net = ResidualAttentionNet_56(feature_dim=feature_dim)
    elif args.backbone == 'Attention_92':
        net = ResidualAttentionNet_92(feature_dim=feature_dim)
    else:
        print(args.backbone, ' is not available!')

    margin = InnerProduct(feature_dim, len(class_names))

    net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
    margin.load_state_dict(torch.load(args.margin_path)['net_state_dict'])

    net = net.to(device)
    margin = margin.to(device)

    net.eval()
    with torch.no_grad():
        raw_logits = net(src_img)
        test_output = margin(raw_logits, None)        
        _, cls_idx = torch.max(test_output.data, 1)
        softmax_result = F.softmax(test_output, dim=1).squeeze(0)
        cls_idx_cpu = int(cls_idx.cpu().numpy())
        softmax_result_cpu = softmax_result.cpu().numpy()
        print(cls_idx_cpu, softmax_result_cpu[cls_idx_cpu])

        txt_color = [(0, 0, 255 ), (0, 255, 0)]
        text = f'{class_names[cls_idx_cpu]}: {softmax_result_cpu[cls_idx_cpu]}'
        cv2.putText(ori_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color[cls_idx_cpu], 1)
        cv2.imshow('Image with Text', ori_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch for image classification')
    parser.add_argument('--source', type=str, default='images/1.jpg', help='select image')
    parser.add_argument('--backbone', type=str, default='MobileFace', help='MobileFace, Res50_IR, SERes50_IR, Res100_IR, SERes100_IR, Attention_56, Attention_92')
    parser.add_argument('--net_path', type=str, default='./checkpoints/MobileFace/Iter_000800_net.ckpt', help='net weight path')
    parser.add_argument('--margin_path', type=str, default='./checkpoints/MobileFace/Iter_000800_margin.ckpt', help='margin weight path')
    parser.add_argument('--gpus', type=str, default='0', help='model prefix')

    args = parser.parse_args()

    test(args)
