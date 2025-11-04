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
from models.resnet import ResNet50
from utils import clamp, get_loaders, my_logger, my_meter, denormalize, get_loaders_CIFAR100, get_loaders_CIFAR10

from DE import init_population, mutation, crossover, calculate_fitness, selection,fitness_selection,decode_individual
import DE

from torchvision.models import resnet50

from torchvision.models import vgg16

import torchvision.transforms.functional as FF



def get_aug():
    parser = argparse.ArgumentParser(description='IMPACT')

    parser.add_argument('--name', default='IMPACT', type=str)
    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--dataset', default='ImageNet', type=str,  choices=['ImageNet','CIFAR100', 'CIFAR10'])
    parser.add_argument('--data_dir', default='/data/yzh/ImageNet', type=str)
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--workers', default=16, type=int)

    parser.add_argument('--network', default='ResNet50', type=str, choices=['resnet50_cifar10', 'resnet50_cifar100','mixer-b',
                                                                           'ResNet152', 'ResNet50', 'ResNet18', 'VGG16','ViT-B'])
    parser.add_argument('--dataset_size', default=100, type=float, help='Use part of Eval set')


    parser.add_argument('--patch_num', default=3, type=int)

    parser.add_argument('--minipatch_num', default=64, type=int)
    parser.add_argument('--tile_size', default=4, type=int)

    parser.add_argument('--population_size', default=50, type=int)
    parser.add_argument('--DE_attack_iters', default=69, type=int)
    parser.add_argument('--ES_attack_iters', default=1500, type=int)

    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    parser.add_argument('--mu', default=[0.485, 0.456, 0.406], nargs='+', type=float, help='The mean value of the dataset') 
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], nargs='+', type=float, help='The std value of the dataset') 

    parser.add_argument('--targeted', action='store_true', help='Whether to perform targeted attack')
    parser.add_argument('--target_label', default=None, type=int, help='Target class label for targeted attack')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for computation')


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
  
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # load model
    if args.network == 'ResNet50':
        model = ResNet50(pretrained=True)
    elif args.network == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
    elif args.network == 'ViT-B':
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    elif args.network == 'resnet50_cifar100':
        model = resnet50(num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load("/home/yzh/adversarial_patch/IMP/models/cifar100_models/resnet50/pytorch_model.bin")
        model.load_state_dict(checkpoint)
    elif args.network == 'resnet50_cifar10':
        model = resnet50(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load("/home/yzh/adversarial_patch/IMP/models/cifar10_models/resnet50/pytorch_model.bin")
        model.load_state_dict(checkpoint)
    else:
        print('Wrong Network')
        raise





    model = model.to(device)
    model.eval()

    # loss
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    # eval dataset
    if args.dataset == 'ImageNet':
        loader = get_loaders(args)
    elif args.dataset == 'CIFAR100':
        loader = get_loaders_CIFAR100(args)
    elif args.dataset == 'CIFAR10':
        loader = get_loaders_CIFAR10(args)

    mu = torch.tensor(args.mu).view(3, 1, 1).to(device)
    std = torch.tensor(args.std).view(3, 1, 1).to(device)


    start_time = time.time()

    successful_count = 0

    for i, (X, y) in enumerate(loader):
        if i == args.dataset_size:
            break

        DE.query_count = 0


        X, y = X.to(device), y.to(device)

        model.zero_grad()
        if 'DeiT' in args.network:
            out, atten = model(X)
        else:
            out = model(X)


        classification_result = out.max(1)[1] == y
        correct_num = classification_result.sum().item()


        # loss of clean samples
        if args.target_label is not None:
            target_y = torch.full_like(y, args.target_label).to(device)
            loss = criterion(out, target_y)
        else:
            loss = criterion(out, y)

        meter.add_loss_acc_asr("Base", {'CE': loss.item()}, correct_num, 0, 0, y.size(0))



        # initialize population
        binary_population, rgb_population = init_population(args.population_size, args.patch_num, args.minipatch_num, args.img_size, args.tile_size)
        
        population = [binary_population, rgb_population]

        # calculate initial fitness
        fitness = calculate_fitness(model, args.minipatch_num, X, population, y,args.img_size, args.tile_size, device,targeted=False)  # [batch_size, population_size]
                
        attack_successful = torch.zeros(args.batch_size, dtype=torch.bool).to(device)
        for step in range(args.DE_attack_iters):
            # if samples are successfully attacked, break
            if attack_successful.all():
                break

                
            # mutation operation
            M_binary_population, M_rgb_population = mutation(args.population_size, [binary_population, rgb_population])
            
            individual_length = (args.img_size // args.tile_size) * (args.img_size // args.tile_size)
            
            # crossover operation
            C_binary_population, C_rgb_population = crossover(args.population_size,[M_binary_population, M_rgb_population], [binary_population, rgb_population], args.patch_num, args.minipatch_num, individual_length)

            # selection operation, compare fitness
            [binary_population, rgb_population], fitness = selection(
                args.minipatch_num, args.population_size, model, X, [C_binary_population, C_rgb_population], [binary_population, rgb_population], fitness, y, args.img_size, args.tile_size, device
            )

            # get best individuals
            best_indices, best_fitness_values = fitness_selection(fitness)
            best_binary_individuals = binary_population[best_indices]
            best_rgb_individuals = rgb_population[best_indices]


            # print fitness
            print(f"Step {step}: Best fitness values: {best_fitness_values}")

            
            mu1 = torch.tensor([0.485, 0.456, 0.406]).to(device)
            std1 = torch.tensor([0.229, 0.224, 0.225]).to(device)

            
            # mu1 = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
            # std1 = torch.tensor([0.2470, 0.2435, 0.2616]).cuda()


            best_masks, best_perturbation = decode_individual(best_binary_individuals, best_rgb_individuals, (args.img_size // args.tile_size), args.tile_size, 
                      args.minipatch_num, mu1, std1, X, device)

            # save mask
            map_image = best_masks[0].detach().cpu()
            max_val = map_image.max()
            map_image = (map_image) / (max_val)
            pil_mask_image = FF.to_pil_image(map_image)
            pil_mask_image.save('mask_image_pil.png')

            # save perturbation
            map_image = best_perturbation[0].detach().cpu()
            max_val = map_image.max()
            map_image = (map_image) / (max_val)
            pil_mask_image = FF.to_pil_image(map_image)
            pil_mask_image.save('perturbation.png')


            # adv sample
            perturb_x = X * (1 - best_masks) + best_perturbation * best_masks

            # visualize adv sample
            de_image = perturb_x[0].detach().cpu()
            de_image = denormalize(de_image, args.mu, args.std)
            de_image = torch.clamp(de_image, 0, 1)
            pil_image = FF.to_pil_image(de_image)
            pil_image.save('de_image.png')



            # evaluate attack success
            with torch.no_grad():
                out = model(perturb_x)

                for idx, pred in enumerate(out.max(1)[1]):
                    if not attack_successful[idx] and pred != y[idx]:
                        attack_successful[idx] = True


        sigma = 0.3

        successful_samples = attack_successful.clone()
        
        current_loss = criterion(out,y if args.target_label is None else target_y).item()
        print(f"current_loss before ES: {current_loss}")

        for es_step in range(args.ES_attack_iters):
            if successful_samples.all():
                break


            noise = torch.normal(0, sigma, size=best_perturbation.shape).to(device)  
            # noise = torch.empty_like(best_perturbation).uniform_(-sigma, sigma).cuda() 
            # momentum = beta * momentum + (1 - beta) * noise
            # new_perturbation = best_perturbation + 2 * momentum
            new_perturbation = best_perturbation + noise
            new_perturbation = clamp(new_perturbation, (0 - mu) / std, (1 - mu) / std)


            perturb_x = X * (1 - best_masks) + new_perturbation * best_masks
            

            with torch.no_grad():
                out = model(perturb_x)
            DE.query_count += 1
            loss = criterion(out, y if args.target_label is None else target_y)


            for idx in range(args.batch_size):
                if not successful_samples[idx]:
                    if loss.item() > current_loss:
                        best_perturbation[idx] = new_perturbation[idx].clone()
                        current_loss = loss.item()



            print(f"es_Step {es_step}: Best fitness values: {current_loss}")

            with torch.no_grad():
                for idx, pred in enumerate(out.max(1)[1]): 
                    if not successful_samples[idx] and pred != y[idx]:
                        successful_samples[idx] = True
                        successful_count += 1


        logger.info(f"Optimization completed: {successful_samples.sum().item()}/{args.batch_size} samples successfully attacked.")
        logger.info(f"Total successful samples using 1-1 ES: {successful_count}.")


        '''Eval Adv Attack'''
        with torch.no_grad():


            perturb_x = X * (1 - best_masks) + best_perturbation * best_masks

            # best = best_perturbation * best_masks
            # map_image = best[0].detach().cpu()
            # max_val = map_image.max()
            # map_image = (map_image) / (max_val)
            # pil_mask_image = FF.to_pil_image(map_image)
            # pil_mask_image.save('perturbation_ES.png')



            # perturb_image = perturb_x[0].detach().cpu()
            # perturb_image = denormalize(perturb_image, args.mu, args.std)
            # perturb_image = torch.clamp(perturb_image, 0, 1)
            # pil_image = FF.to_pil_image(perturb_image)
            # pil_image.save('adversarial_image.png')



            out = model(perturb_x)


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
            logger.info("\n")

    end_time = time.time()
    msg = meter.get_loss_acc_msg()
    logger.info("\nFinish! Using time: {}\n{}".format((end_time - start_time), msg))


if __name__ == "__main__":
    main()
