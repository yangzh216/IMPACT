import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
import numpy as np
import time

from models.bagnet import bagnet17
from models.resnet import ResNet50, ResNet152, ResNet101
from utils import clamp, get_loaders, my_logger, my_meter, denormalize, get_loaders_CIFAR100, get_loaders_CIFAR10

from DE import init_population, mutation, crossover, calculate_fitness, selection,fitness_selection
import DE

from torchvision.models import resnet50

from torchvision.models import vgg16
import timm

import torchvision.transforms.functional as FF
import math



def get_aug():
    parser = argparse.ArgumentParser(description='IMPACT')

    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--dataset', default='ImageNet', type=str,  choices=['ImageNet','CIFAR100', 'CIFAR10'])
    parser.add_argument('--data_dir', default='/data/yzh/ImageNet', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)

    parser.add_argument('--network', default='ResNet50', type=str, choices=['resnet50_cifar10', 'resnet18_cifar100','mixer-b','bagnet17', 'AT_ResNet50',
                                                                           'ResNet152', 'ResNet50', 'ResNet18', 'VGG16','ViT-B'])
    parser.add_argument('--dataset_size', default=10, type=float, help='Use part of Eval set')


    parser.add_argument('--patch_num', default=3, type=int)

    parser.add_argument('--minipatch_num', default=64, type=int)
    parser.add_argument('--tile_size', default=4, type=int)

    parser.add_argument('--population_size', default=25, type=int)
    parser.add_argument('--DE_attack_iters', default=139, type=int)
    parser.add_argument('--ES_attack_iters', default=1500, type=int)

    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--mu', default=[0.485, 0.456, 0.406], nargs='+', type=float, help='The mean value of the dataset') 
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], nargs='+', type=float, help='The std value of the dataset') 

    parser.add_argument('--targeted', action='store_true', help='Whether to perform targeted attack')
    parser.add_argument('--target_label', default=None, type=int, help='Target class label for targeted attack')


    args = parser.parse_args()


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


def main():
    args = get_aug()


    logger = my_logger(args)
    meter = my_meter()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
  
    # load model
    if args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.network == 'ViT-B':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        

    # elif args.network == 'AT_ResNet50':
    #     current_dir = os.getcwd()
    #     robustness_parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'patch_defenses', 'robustness')) 
    #     if os.path.isdir(robustness_parent_dir) and robustness_parent_dir not in sys.path:
    #         sys.path.insert(0, robustness_parent_dir) # insert(0, ...) 优先搜索此路径
    #         print(f"Added directory to sys.path: {robustness_parent_dir}")
    #     elif robustness_parent_dir in sys.path:
    #         print(f"Directory already in sys.path: {robustness_parent_dir}")
    #     else:
    #         print(f"Error: Directory not found: {robustness_parent_dir}")
    #         # 可以选择退出或抛出异常
    #         # sys.exit(1) 

    #     # 4. 现在你可以正常导入 robustness 库及其内容了
    #     try:
    #         import torch as ch # robustness 库常用别名
    #         from robustness.datasets import ImageNet # 或者 CIFAR
    #         from robustness.model_utils import make_and_restore_model
    #     except ImportError as e:
    #         print(f"Error importing robustness library: {e}")
    #         print("Please ensure the path added to sys.path is correct and the library exists.")
    #         sys.exit(1)        
    #     imagenet_path = '/data/yzh/ImageNet'
    #     ds = ImageNet(imagenet_path)
    #     resume_path = '/home/yzh/adversarial_patch/IMP/defense_pt/imagenet_linf_4.pt'
    #     model, checkpoint = make_and_restore_model(arch='resnet50', dataset=ds,
    #                                             resume_path=resume_path) 
    #     print("Robust model loaded successfully!")

    elif args.network == 'bagnet17':
        model = bagnet17(pretrained=True,clip_range=None,aggregation='mean')
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('/home/yzh/adversarial_patch/patch_defenses/PatchGuard/checkpoints/bagnet17_net.pth')
        model.load_state_dict(checkpoint['state_dict'])

    elif args.network == 'resnet50_cifar100':
        # print([m for m in timm.list_models() if 'cifar10' in m])
        # model = timm.create_model('resnet32_cifar100', pretrained=True, num_classes=100)
        model = resnet50(num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # 去掉 maxpool
        checkpoint = torch.load("/home/yzh/adversarial_patch/IMP/models/cifar100_models/resnet50/pytorch_model.bin")
        model.load_state_dict(checkpoint)
    elif args.network == 'resnet50_cifar10':
        # print([m for m in timm.list_models() if 'cifar10' in m])
        # model = timm.create_model('resnet32_cifar100', pretrained=True, num_classes=100)
        model = resnet50(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # 去掉 maxpool
        checkpoint = torch.load("/home/yzh/adversarial_patch/IMP/models/cifar10_models/resnet50/pytorch_model.bin")
        model.load_state_dict(checkpoint)
    else:
        print('Wrong Network')
        raise





    model = model.cuda()
    model.eval()

    # loss
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    # eval dataset
    if args.dataset == 'ImageNet':
        loader = get_loaders(args)
    elif args.dataset == 'CIFAR100':
        loader = get_loaders_CIFAR100(args)
    elif args.dataset == 'CIFAR10':
        loader = get_loaders_CIFAR10(args)

    mu = torch.tensor(args.mu).view(3, 1, 1).cuda()
    std = torch.tensor(args.std).view(3, 1, 1).cuda()


    start_time = time.time()

    '''Original image been classified incorrect but turn to be correct after adv attack'''
    false2true_num = 0

    successful_count = 0

    for i, (X, y) in enumerate(loader):
        if i == args.dataset_size:
            break


        source_image = X[0].detach().cpu()
        # 反标准化
        source_image = denormalize(source_image, args.mu, args.std)
        # 将张量的值限制在 [0, 1] 范围内（确保值在显示图像时有效）
        source_image = torch.clamp(source_image, 0, 1)
        source_image = FF.to_pil_image(source_image)
        source_image.save(f'/home/yzh/adversarial_patch/IMP/print/source_image_{i}_{y[0]}.png')



        DE.query_count = 0


        X, y = X.cuda(), y.cuda()

        model.zero_grad()
        if 'DeiT' in args.network:
            out, atten = model(X)
        else:
            out = model(X)


        classification_result = out.max(1)[1] == y
        correct_num = classification_result.sum().item()


        # 处理目标攻击的损失
        if args.target_label is not None:
            target_y = torch.full_like(y, args.target_label).cuda()  # 固定目标标签
            loss = criterion(out, target_y)
        else:
            loss = criterion(out, y)

        meter.add_loss_acc_asr("Base", {'CE': loss.item()}, correct_num, 0, 0, y.size(0))


        # 初始化种群
        binary_population, rgb_population = init_population(args.batch_size, args.population_size, args.patch_num, args.minipatch_num)
        
        population = [binary_population, rgb_population]
        # # 计算初始适应度
        fitness = calculate_fitness(model, args.minipatch_num, X, population, y)  # 形状 [batch_size, population_size]
                

        attack_successful = torch.zeros(args.batch_size, dtype=torch.bool).cuda()  # 每个样本是否成功攻击
        for step in range(args.DE_attack_iters):
            # 如果所有样本已经成功攻击，跳出循环
            if attack_successful.all():
                break

                
            # 变异操作
            M_binary_population, M_rgb_population = mutation(args.batch_size, args.population_size, [binary_population, rgb_population])  # 按01编码和RGB进行变异
            
            
            # 交叉操作
            C_binary_population, C_rgb_population = crossover(args.batch_size, args.population_size,[M_binary_population, M_rgb_population], [binary_population, rgb_population], args.patch_num, args.minipatch_num)  # 按概率交叉

            # 选择操作，比较适应度
            [binary_population, rgb_population], fitness = selection(
                args.batch_size, args.minipatch_num, args.population_size, model, X, [C_binary_population, C_rgb_population], [binary_population, rgb_population], fitness, y
            )

            # 获取每批次中当前最优的个体
            best_indices, best_fitness_values = fitness_selection(fitness)
            best_binary_individuals = binary_population[np.arange(args.batch_size), best_indices]


            # 输出调试信息
            print(f"Step {step}: Best fitness values: {best_fitness_values}")

            # 构造掩码
            best_masks = best_binary_individuals.reshape(args.batch_size, (args.img_size // args.tile_size), (args.img_size // args.tile_size))
            best_masks = np.kron(best_masks, np.ones((args.tile_size, args.tile_size), dtype=int))  # 扩展到完整 mask
            best_masks = torch.tensor(best_masks, dtype=torch.float32).unsqueeze(1).cuda()  # 添加通道维度
            

            # mu1 = torch.tensor([0.485, 0.456, 0.406]).cuda()  # 通道均值
            # std1 = torch.tensor([0.229, 0.224, 0.225]).cuda()  # 通道标准差

            mu1 = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()  # 通道均值
            std1 = torch.tensor([0.2470, 0.2435, 0.2616]).cuda()  # 通道标准差

            # 构造扰动
            best_perturbation = torch.zeros_like(X).cuda()
            for b in range(args.batch_size):  # 遍历每个批次
                for p in range(args.minipatch_num):  # 遍历每个 patch
                    one_indices = np.where(binary_population[b, best_indices[b]] == 1)[0]
                    if len(one_indices) > p:
                        xx, yy = divmod(one_indices[p], (args.img_size // args.tile_size))
                        patch_rgb = rgb_population[b, best_indices[b], p]  # 提取原始 RGB 值
                        patch_rgb = torch.tensor(patch_rgb, dtype=torch.float32, device='cuda')  # 转为 PyTorch 张量
                        patch_rgb = patch_rgb / 255.0  # 归一化到 [0, 1]
                        patch_rgb = (patch_rgb - mu1) / std1  # 标准化
                        best_perturbation[b, :, xx * args.tile_size:(xx + 1) * args.tile_size, yy * args.tile_size:(yy + 1) * args.tile_size] = patch_rgb.view(3, 1, 1)

            # 获取遮挡图
            # print(best_masks.shape)
            map_image = best_masks[0].detach().cpu()
            max_val = map_image.max()
            map_image = (map_image) / (max_val)
            pil_mask_image = FF.to_pil_image(map_image)
            pil_mask_image.save('mask_image_pil.png')


            # print(best_perturbation.shape)
            # map_image = best_perturbation[0].detach().cpu()
            # max_val = map_image.max()
            # map_image = (map_image) / (max_val)
            # pil_mask_image = FF.to_pil_image(map_image)
            # pil_mask_image.save('DE_image_pil.png')


            # 生成攻击样本
            perturb_x = X * (1 - best_masks) + best_perturbation * best_masks


            # 检测是否有样本攻击成功
            with torch.no_grad():
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)

                for idx, pred in enumerate(out.max(1)[1]):  # 获取模型预测标签
                    if not attack_successful[idx] and pred != y[idx]:  # 检查是否成功攻击
                        attack_successful[idx] = True


        sigma = 0.2  # 高斯噪声标准差

        successful_samples = attack_successful.clone()  # 记录成功攻击的样本
        
        current_loss = criterion(out,y if args.target_label is None else target_y).item()

        for es_step in range(args.ES_attack_iters):
            if successful_samples.all():
                break

            # 为每个样本生成新的扰动
            noise = torch.normal(0, sigma, size=best_perturbation.shape).cuda()  # 高斯噪声
            # noise = torch.empty_like(best_perturbation).uniform_(-sigma, sigma).cuda() #均匀分布噪声
            # momentum = beta * momentum + (1 - beta) * noise
            # new_perturbation = best_perturbation + 2 * momentum
            new_perturbation = best_perturbation + noise
            new_perturbation = clamp(new_perturbation, (0 - mu) / std, (1 - mu) / std)  # 限制范围

            # 使用固定的掩码生成新的攻击样本
            perturb_x = X * (1 - best_masks) + new_perturbation * best_masks
            
            # 计算新的适应度（损失值）
            with torch.no_grad():
                if 'DeiT' in args.network:
                    out, atten = model(perturb_x)
                else:
                    out = model(perturb_x)
                DE.query_count += 1
                loss = criterion(out, y if args.target_label is None else target_y)

            # 比较适应度，选择更优扰动
            for idx in range(args.batch_size):
                if not successful_samples[idx]:  # 仅优化未成功的样本
                    if loss.item() > current_loss:  # 如果新的扰动效果更好
                        best_perturbation[idx] = new_perturbation[idx].clone()
                        current_loss = loss.item()



            print(f"es_Step {es_step}: Best fitness values: {current_loss}")
            # 检测攻击是否成功
            with torch.no_grad():
                for idx, pred in enumerate(out.max(1)[1]):  # 获取预测标签
                    if not successful_samples[idx] and pred != y[idx]:  # 攻击成功条件
                        successful_samples[idx] = True
                        successful_count += 1  # 成功优化样本数量 +1

        # 输出最终结果
        logger.info(f"Optimization completed: {successful_samples.sum().item()}/{args.batch_size} samples successfully attacked.")
        logger.info(f"Total successful samples using 1-1 ES: {successful_count}.")


        '''Eval Adv Attack'''
        with torch.no_grad():


            perturb_x = X * (1 - best_masks) + best_perturbation * best_masks

            # # 计算 L0 范数
            # difference = X - perturb_x  # 计算像素差异
            # non_zero_elements = (difference != 0).float().sum(dim=(1, 2, 3))  # 每张图片的非零差异数
            # l0_norm = non_zero_elements  # 这就是 L0 范数

            # print(f"L0 norms for each image in the batch: {l0_norm}")

            # 假设 perturb_x 是扰动后的图像张量，取第一个样本
            perturb_image = perturb_x[0].detach().cpu()
            # 反标准化
            perturb_image = denormalize(perturb_image, args.mu, args.std)
            # 将张量的值限制在 [0, 1] 范围内（确保值在显示图像时有效）
            perturb_image = torch.clamp(perturb_image, 0, 1)
            pil_image = FF.to_pil_image(perturb_image)
            pil_image.save('adversarial_image.png')


            if 'DeiT' in args.network:
                out, atten = model(perturb_x)
            else:
                out = model(perturb_x)


            # 评估有目标攻击的结果
            if args.target_label is not None:
                classification_result_after_attack = out.max(1)[1] == args.target_label
                acc_suc = out.max(1)[1] == args.target_label
                loss = criterion(out, target_y)
            else:
                classification_result_after_attack = out.max(1)[1] == y
                acc_suc = out.max(1)[1] != y
                loss = criterion(out, y)


            # classification_result_after_attack = out.max(1)[1] == y
            # loss = criterion(out, y)
            meter.add_loss_acc_asr("ADV", {'CE': loss.item()}, (classification_result_after_attack.sum().item()), (acc_suc.sum().item()), DE.query_count, y.size(0))

        '''Message'''
        if i % 1 == 0:
            logger.info("Iter: [{:d}/{:d}] Loss and Acc for all models:".format(i, args.dataset_size))
            msg = meter.get_loss_acc_msg()
            logger.info(msg)

            classification_result_after_attack = classification_result_after_attack[classification_result == False]
            false2true_num += classification_result_after_attack.sum().item()
            logger.info("Total False -> True: {}\n".format(false2true_num))

    end_time = time.time()
    msg = meter.get_loss_acc_msg()
    logger.info("\nFinish! Using time: {}\n{}".format((end_time - start_time), msg))


if __name__ == "__main__":
    main()
