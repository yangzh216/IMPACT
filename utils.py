import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from datetime import datetime
from torchvision.datasets import CIFAR100, CIFAR10

# mu = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]



def denormalize(tensor, mean, std):
    """
    将标准化后的图像反标准化回原始状态。
    Args:
        tensor: 标准化后的张量，形状 [C, H, W]
        mean: 标准化时使用的均值
        std: 标准化时使用的标准差
    Returns:
        反标准化后的张量
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean  # 反标准化公式



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(args):
    # args.mu = mu
    # args.std = std
    valdir = path.join(args.data_dir, 'val')
    # valdir = path.join(args.data_dir, '10_class_val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize(args.img_size),
                                                           transforms.CenterCrop(args.crop_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=args.mu, std=args.std)
                                                           ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    # print(val_dataset.class_to_idx)
    return val_loader



def get_loaders_CIFAR100(args):
    # CIFAR-100 标准均值和标准差
    args.mu = [0.5071, 0.4867, 0.4408]
    args.std = [0.2675, 0.2565, 0.2761]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mu, std=args.std),
    ])

    val_dataset = CIFAR100(root='/home/yzh/adversarial_patch/IMPACT/dataset', train=False, download=True, transform=transform)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    return val_loader

def get_loaders_CIFAR10(args):
    # CIFAR-10 标准均值和标准差
    args.mu = [0.4914, 0.4822, 0.4465]
    args.std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mu, std=args.std),
    ])

    val_dataset = CIFAR10(root='/home/yzh/adversarial_patch/IMPACT/dataset', train=False, download=True, transform=transform)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers,
                                             pin_memory=True)

    return val_loader

'''
@Parameter atten_grad, ce_grad: should be 2D tensor with shape [batch_size, -1]
'''
def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad


'''
random shift several patches within the range
'''
def shift_image(image, range, mu, std, patch_size=16):
    batch_size, channel, h, w = image.shape
    h_range, w_range = range
    new_h = h + 2 * h_range * patch_size
    new_w = w + 2 * w_range * patch_size
    new_image = torch.zeros([batch_size, channel, new_h, new_w]).cuda()
    new_image = (new_image - mu) / std
    shift_h = np.random.randint(-h_range, h_range+1)
    shift_w = np.random.randint(-w_range, w_range+1)
    # shift_h = np.random.randint(-1, 2)
    # shift_w = 0
    new_image[:, :, h_range*patch_size : h+h_range*patch_size, w_range*patch_size : w + w_range*patch_size] = image.detach()
    h_start = (h_range + shift_h) * patch_size
    w_start = (w_range + shift_w) * patch_size
    new_image = new_image[:, :, h_start : h_start+h, w_start : w_start+w]
    return new_image



class my_logger:
    def __init__(self, args):
        name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log".format(datetime.now().strftime("%Y-%m-%d %H:%M"), args.seed, args.network, args.dataset, args.population_size, args.minipatch_num, args.patch_num, args.DE_attack_iters, args.ES_attack_iters, args.target_label)
        args.name = name
        # name = "{}_{}.log".format(args.network, args.dataset)
        # args.name = name
        self.name = path.join(args.log_dir, name)
        with open(self.name, 'w') as F:
            print('\n'.join(['%s:%s' % item for item in args.__dict__.items() if item[0][0] != '_']), file=F)
            print('\n', file=F)

    def info(self, content):
        with open(self.name, 'a') as F:
            print(content)
            print(content, file=F)


class my_meter:
    def __init__(self):
        self.meter_list = {}

    def add_loss_acc_asr(self, model_name, loss_dic: dict, correct_num, suc_att_num, querry, batch_size):
        if model_name not in self.meter_list.keys():
            self.meter_list[model_name] = self.model_meter()
        sub_meter = self.meter_list[model_name]
        sub_meter.add_loss_acc_asr(loss_dic, correct_num, suc_att_num, querry, batch_size)

    def clean_meter(self):
        for key in self.meter_list.keys():
            self.meter_list[key].clean_meter()

    def get_loss_acc_msg(self):
        msg = []
        for key in self.meter_list.keys():
            sub_meter = self.meter_list[key]
            sub_loss_bag = sub_meter.get_loss()
            loss_msg = ["{}: {:.4f}({:.4f})".format(x, sub_meter.last_loss[x], sub_loss_bag[x])
                        for x in sub_loss_bag.keys()]
            loss_msg = " ".join(loss_msg)
            msg.append("model:{} Loss:{} Acc:{:.4f}({:.4f}) Asr:{:.4f}({:.4f}) AQ:{:.4f}({:.4f})".format(
                key, loss_msg, sub_meter.last_acc, sub_meter.get_acc(), sub_meter.last_asr, sub_meter.get_asr(), sub_meter.last_q, sub_meter.get_q()))
        msg = "\n".join(msg)
        return msg

    class model_meter:
        def __init__(self):
            self.loss_bag = {}
            self.acc = 0.
            self.count = 0
            self.last_loss = {}
            self.last_acc = 0.
            self.asr = 0.
            self.last_asr = 0.
            
            self.q = 0
            self.last_q = 0.


        def add_loss_acc_asr(self, loss_dic: dict, correct_num, suc_att_num, query, batch_size):
            for loss_name in loss_dic.keys():
                if loss_name not in self.loss_bag.keys():
                    self.loss_bag[loss_name] = 0.
                self.loss_bag[loss_name] += loss_dic[loss_name] * batch_size
            self.last_loss = loss_dic

            self.last_acc = correct_num / batch_size
            self.acc += correct_num

            self.last_asr = suc_att_num / batch_size
            self.asr += suc_att_num

            self.last_q = query / batch_size
            self.q += query

            self.count += batch_size




        def get_loss(self):
            return {x: self.loss_bag[x] / self.count for x in self.loss_bag.keys()}

        def get_acc(self):
            return self.acc / self.count
        
        def get_asr(self):
            return self.asr / self.count
        
        def get_q(self):
            return self.q / self.count        

        def clean_meter(self):
            self.__init__()
