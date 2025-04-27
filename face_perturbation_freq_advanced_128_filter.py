
import argparse
import collections
import datetime
import os
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import util
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer_loss import Trainer
mlconfig.register(madrys.MadrysLoss)
# from models.ResNet import ResNet18
from models.ResNet_128 import ResNet18
from models.inception_resnet_v1 import InceptionResnetV1
import torch.nn.functional as F

import torchvision.transforms.functional as TF
import scipy.ndimage as nd
import kornia.filters as KF
import kornia as K
import pywt

import warnings
os.environ["OMP_NUM_THREADS"] = "4"          # cap OpenMP/MKL
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

torch.set_num_threads(4)                     # intra-op
torch.set_num_interop_threads(4)             # inter-op
torch.backends.cudnn.benchmark = True        # autotune fastest convs
warnings.filterwarnings("ignore", category=UserWarning)  # silence weight msg



# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='../datasets')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.2, type=float)
parser.add_argument('--universal_stop_error', default=0.5, type=float)
parser.add_argument('--universal_train_target', default='train_subset', type=str)
parser.add_argument('--train_step', default=10, type=int)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-max', 'random'], help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='noise shape')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)

parser.add_argument('--band_fft', type=str, default='high')
parser.add_argument('--filter_type', type=str, default='sobel')
parser.add_argument('--mode', type=str, default='add')
parser.add_argument('--phi_var', type=float, default=0.03)


args = parser.parse_args()

# Convert Eps
args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255

# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
print("config: ",config)
# config.set_immutable()
from omegaconf import OmegaConf
OmegaConf.set_readonly(config, True)

for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))




def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
    return


def universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    random_noise = random_noise.to(device)
    model = model.to(device)
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i in range(len(labels)):
                class_index = labels[i].item()
                noise = random_noise[class_index]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=images[i].shape, patch_location=args.patch_location)
                # mask_cord, class_noise = noise_generator._patch_noise_extend_to_img_frequency_domain(noise, image_size=images[i].shape, patch_location=args.patch_location, freq_type='high')
                images[i] += class_noise
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, band_fft, filter_type, mode, phi_var):
    # Class-Wise perturbation
    # Generate Data loader
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)

    if args.use_subset:
        data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                               train_shuffle=True, train_drop_last=True)
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)

    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
    if hasattr(model, 'classify'):
        model.classify = True
    while condition:
        if args.attack_type == 'min-min' and not args.load_model:
            # Train Batch for min-min noise
            for j in range(0, args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)

                # # Apply FFT-based perturbation
                perturbed_images = ciper_fft_perturbation_kornia(images, band=band_fft, phi=phi_var, filter_type=filter_type, mode=mode)
                
                train_imgs = perturbed_images
                
                # Train
                model.train()
                for param in model.parameters():
                    param.requires_grad = True

                trainer.train_batch(images, train_imgs, labels, model, optimizer)

        for i, (images, labels) in tqdm(enumerate(data_loader[args.universal_train_target]), total=len(data_loader[args.universal_train_target])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)
            # Add Class-wise Noise to each sample
            batch_noise, mask_cord_list = [], []
            for i, (image, label) in enumerate(zip(images, labels)):
                noise = random_noise[label.item()]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                batch_noise.append(class_noise)
                mask_cord_list.append(mask_cord)
            
        
            # Update universal perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            class_noise_eta = collections.defaultdict(list)
            for i in range(len(eta)):
                x1, x2, y1, y2 = mask_cord_list[i]
                delta = eta[i][:, x1: x2, y1: y2]
                class_noise_eta[labels[i].item()].append(delta.detach().cpu())

            for key in class_noise_eta:
                delta = torch.stack(class_noise_eta[key]).mean(dim=0) - random_noise[key]
                class_noise = random_noise[key]
                class_noise += delta
                random_noise[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon)

        # Eval termination conditions
        loss_avg, error_rate = universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))
        random_noise = random_noise.detach()
        ENV['random_noise'] = random_noise
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error
    return random_noise


def samplewise_perturbation_eval(random_noise, data_loader, model, eval_target='train_dataset', mask_cord_list=[]):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    # random_noise = random_noise.to(device)
    model = model.to(device)
    idx = 0
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i, (image, label) in enumerate(zip(images, labels)):
                if not torch.is_tensor(random_noise):
                    sample_noise = torch.tensor(random_noise[idx]).to(device)
                else:
                    sample_noise = random_noise[idx].to(device)
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                images[i] = images[i] + sample_noise
                idx += 1
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, band_fft, filter_type, mode, phi_var):
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)

    if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
        data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
        data_loader['train_dataset'] = data_loader['train_subset']
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    mask_cord_list = []
    idx = 0
    for images, labels in data_loader['train_dataset']:
        for i, (image, label) in enumerate(zip(images, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    condition = True
    train_idx = 0
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    while condition:
        if args.attack_type == 'min-min' and not args.load_model:
            # Train Batch for min-min noise
            for j in tqdm(range(0, args.train_step), total=args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    train_idx = 0
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)
                
                unperturbted_img = images
                
                # Apply FFT-based perturbation on each image in the batch
                perturbed_images = []
                for i, (image, label) in enumerate(zip(images, labels)):
                    perturbed_image = ciper_fft_perturbation_kornia(image.unsqueeze(0), band=band_fft, phi=phi_var, filter_type=filter_type, mode=mode)
                    perturbed_images.append(perturbed_image)
                perturbed_images = torch.cat(perturbed_images, dim=0)
                unperturbted_img = images
                images = perturbed_images
                
                
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(unperturbted_img, images, labels, model, optimizer)

        # Search For Noise
        idx = 0
        for i, (images, labels) in tqdm(enumerate(data_loader['train_dataset']), total=len(data_loader['train_dataset'])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)

            # Add Sample-wise Noise to each sample
            batch_noise, batch_start_idx = [], idx
            for i, (image, label) in enumerate(zip(images, labels)):
                sample_noise = random_noise[idx]
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                if type(sample_noise) is np.ndarray:
                    mask[:, x1: x2, y1: y2] = sample_noise
                else:
                    mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                batch_noise.append(sample_noise)
                idx += 1
            

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            batch_noise = torch.stack(batch_noise).to(device)
            batch_noise = batch_noise.squeeze(1) 
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                if torch.is_tensor(random_noise):
                    random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(random_noise, data_loader, model, eval_target='train_dataset',
                                                            mask_cord_list=mask_cord_list)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))

        if torch.is_tensor(random_noise):
            random_noise = random_noise.detach()
            ENV['random_noise'] = random_noise
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = image.shape[0], image.shape[1], image.shape[2]
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise
    else:
        return random_noise




def apply_filter(image, filter_type='laplace'):
    """
    Apply predefined filters (CiPer-style) to extract noise from the image.

    Args:
        image (torch.Tensor): Image tensor (H, W) or (C, H, W) in range [0, 1].
        filter_type (str): Type of filter to apply. Options: 'laplace', 'gaussian', 'median', 'sobel', 'prewitt'.

    Returns:
        torch.Tensor: Extracted noise (difference between original and filtered image).
    """
    image_np = image.detach().cpu().numpy()

    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, ...]  # Convert to (1, H, W)

    c, h, w = image_np.shape
    filtered = np.zeros_like(image_np)

    for i in range(c):
        if filter_type == 'laplace':
            filtered[i] = nd.laplace(image_np[i])
        elif filter_type == 'gaussian':
            filtered[i] = nd.gaussian_filter(image_np[i], sigma=1)
        elif filter_type == 'median':
            filtered[i] = nd.median_filter(image_np[i], size=3)
        elif filter_type == 'sobel':
            gx = nd.sobel(image_np[i], axis=0)
            gy = nd.sobel(image_np[i], axis=1)
            filtered[i] = np.hypot(gx, gy)
        elif filter_type == 'prewitt':
            gx = nd.prewitt(image_np[i], axis=0)
            gy = nd.prewitt(image_np[i], axis=1)
            filtered[i] = np.hypot(gx, gy)
        else:
            raise ValueError(f"Unsupported filter: {filter_type}")

    noise = image_np - filtered
    return torch.tensor(noise, dtype=image.dtype)


def ciper_fft_perturbation(image, phi=0.05, band='high', filter_type='laplace', mode='add'):
    """
    Apply CiPer-based filter noise in frequency domain to create perturbation.

    Args:
        image (torch.Tensor): Single image tensor (B, C, H, W)
        phi (float): Strength of noise to apply
        band (str): Frequency band to target: 'high', 'mid', 'low', 'all'
        filter_type (str): Type of filter to extract CiPer noise
        mode (str): 'add' or 'subtract' perturbation in frequency domain

    Returns:
        torch.Tensor: Perturbed image (same shape as input)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    device = image.device
    B, C, H, W = image.shape

    # Apply filter to each image in the batch
    noise_batch = []
    for b in range(B):
        filtered = torch.stack([apply_filter(image[b, c], filter_type) for c in range(C)])
        noise_batch.append(filtered)
    noise = torch.stack(noise_batch).to(device) * phi  # (B, C, H, W)

    # FFT
    image_fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    noise_fft = torch.fft.fftshift(torch.fft.fftn(noise, dim=(-2, -1)), dim=(-2, -1))

    # Frequency mask
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
    freq_radius = torch.sqrt(xx ** 2 + yy ** 2).to(device)

    if band == 'high':
        mask = (freq_radius >= 0.3).float()
    elif band == 'mid':
        mask = ((freq_radius >= 0.15) & (freq_radius < 0.3)).float()
    elif band == 'low':
        mask = (freq_radius < 0.15).float()
    else:
        mask = torch.ones_like(freq_radius)

    mask = mask.view(1, 1, H, W).expand(B, C, H, W)
    
    noise_fft = noise_fft.squeeze(2)

    # Apply masked noise in frequency domain
    if mode == 'add':
        perturbed_fft = image_fft + noise_fft * mask
    elif mode == 'subtract':
        perturbed_fft = image_fft - noise_fft * mask
    else:
        raise ValueError("mode must be either 'add' or 'subtract'")

    # Inverse FFT
    perturbed_fft = torch.fft.ifftshift(perturbed_fft, dim=(-2, -1))
    perturbed_image = torch.fft.ifftn(perturbed_fft, dim=(-2, -1)).real

    return perturbed_image.clamp(0, 1)






def apply_filter_kornia(image: torch.Tensor, filter_type='laplace'):
    """
    Apply Kornia filters (CiPer-style) to extract noise from the image.

    Args:
        image (torch.Tensor): Image tensor (C, H, W) in range [0, 1], on GPU.
        filter_type (str): One of ['laplace', 'gaussian', 'median', 'sobel'].

    Returns:
        torch.Tensor: Noise image (original - filtered), shape (C, H, W), on GPU.
    """
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)

    if filter_type == 'laplace':
        filtered = KF.laplacian(image, kernel_size=3)
    elif filter_type == 'gaussian':
        filtered = KF.gaussian_blur2d(image, (3, 3), (1.0, 1.0))
    elif filter_type == 'median':
        filtered = KF.median_blur(image, (3, 3))
    elif filter_type == 'sobel':
        filtered = KF.sobel(image)
        filtered = torch.sqrt(filtered.pow(2).sum(dim=1, keepdim=True))  # Combine dx and dy
    elif filter_type == 'average':
        filtered = KF.box_blur(image, (3, 3))
    elif filter_type == 'guided':
        # image: (1, C, H, W)
        filtered = K.filters.guided_blur(image, image, kernel_size=(5, 5), eps=1e-2, border_type='reflect')
    elif filter_type == 'integral':
        # Fast approximation: cumulative sum over H and W
        image_np = image.squeeze(0).cpu().numpy()
        integral = np.cumsum(np.cumsum(image_np, axis=1), axis=2)
        integral = torch.tensor(integral, dtype=image.dtype).unsqueeze(0).to(device)
        filtered = F.avg_pool2d(integral, kernel_size=3, stride=1, padding=1)  # crude approx
    elif filter_type == 'wavelet':
        image_np = image.squeeze(0).cpu().numpy()  # (C, H, W)
        high_freq_filtered = []
        for c in range(image_np.shape[0]):
            coeffs2 = pywt.dwt2(image_np[c], 'haar')
            LL, (LH, HL, HH) = coeffs2
            high_freq = LH + HL + HH
            high_freq_filtered.append(torch.tensor(high_freq, dtype=image.dtype))
        high_freq_filtered = torch.stack(high_freq_filtered).unsqueeze(0).to(device)
        filtered = F.interpolate(high_freq_filtered, size=image.shape[-2:], mode='bilinear')
    elif filter_type == 'log':
        # LoG = Gaussian smoothing followed by Laplacian
        image_np = image.squeeze(0).cpu().numpy()
        log_filtered = []
        for c in range(image_np.shape[0]):
            log = nd.gaussian_laplace(image_np[c], sigma=1)
            log_filtered.append(torch.tensor(log, dtype=image.dtype))
        filtered = torch.stack(log_filtered).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unsupported filter: {filter_type}")

    noise = image - filtered
    return noise.squeeze(0)


def ciper_fft_perturbation_kornia(image: torch.Tensor, phi=0.05, band='high', filter_type='laplace', mode='add'):
    """
    Apply CiPer-style frequency-domain perturbation using Kornia and torch.fft.

    Args:
        image (torch.Tensor): Input tensor of shape (B, C, H, W) on GPU.
        phi (float): Perturbation strength.
        band (str): One of ['high', 'mid', 'low', 'all'].
        filter_type (str): One of Kornia-supported filters.
        mode (str): Either 'add' or 'subtract'.

    Returns:
        torch.Tensor: Perturbed image (B, C, H, W), on GPU.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = image.shape
    device = image.device

    # Apply filter to extract noise (in batch)
    noise_list = []
    for b in range(B):
        noise = apply_filter_kornia(image[b], filter_type)
        noise_list.append(noise)
    noise = torch.stack(noise_list).to(device) * phi  # (B, C, H, W)

    # FFT on image and noise
    image_fft = torch.fft.fftshift(torch.fft.fftn(image, dim=(-2, -1)), dim=(-2, -1))
    noise_fft = torch.fft.fftshift(torch.fft.fftn(noise, dim=(-2, -1)), dim=(-2, -1))

    # Create frequency mask
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    freq_radius = torch.sqrt(xx ** 2 + yy ** 2).to(device)

    if band == 'high':
        mask = (freq_radius >= 0.3).float()
    elif band == 'mid':
        mask = ((freq_radius >= 0.15) & (freq_radius < 0.3)).float()
    elif band == 'low':
        mask = (freq_radius < 0.15).float()
    else:
        mask = torch.ones_like(freq_radius)

    mask = mask.view(1, 1, H, W).expand(B, C, H, W)

    # Apply masked noise in frequency domain
    if mode == 'add':
        perturbed_fft = image_fft + noise_fft * mask
    elif mode == 'subtract':
        perturbed_fft = image_fft - noise_fft * mask
    else:
        raise ValueError("mode must be 'add' or 'subtract'")

    # Inverse FFT
    perturbed_fft = torch.fft.ifftshift(perturbed_fft, dim=(-2, -1))
    perturbed_image = torch.fft.ifftn(perturbed_fft, dim=(-2, -1)).real

    return perturbed_image.clamp(0, 1)




def main():
    # Setup ENV
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed)
    data_loader = datasets_generator.getDataLoader()


    # Mapping model name to actual model class (example for ResNet18)
    if config.model.name == 'ResNet18':
        model =  ResNet18(num_classes=config.model.num_classes).to(device)
    elif config.model.name == 'InceptionResnetV1':
        model =  InceptionResnetV1(pretrained=None,classify=True,num_classes=config.model.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.version} and {config.model.name}")

    # model = config.model().to(device)
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    # Get the optimizer configuration
    optimizer_name = config.optimizer['name']
    lr = config.optimizer['lr']
    weight_decay = config.optimizer['weight_decay']
    momentum = config.optimizer['momentum']

    # Select the optimizer based on the config
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    
    # Get the scheduler configuration
    scheduler_name = config.scheduler['name']
    

    # Select the scheduler based on the config
    if scheduler_name == 'CosineAnnealingLR':
        T_max = config.scheduler['T_max']
        eta_min = config.scheduler['eta_min']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'MultiStepLR':
        milestones = config.scheduler['milestones']
        gamma = config.scheduler['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")



    # Get the criterion configuration
    criterion_name = config.criterion['name']

    # Select the criterion based on the config
    if criterion_name == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")


    if args.perturb_type == 'samplewise':
        train_target = 'train_dataset'
    else:
        if args.use_subset:
            data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                                   train_shuffle=True, train_drop_last=True)
            train_target = 'train_subset'
        else:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'

    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        if args.attack_type == 'min-max':
            # min-max noise need model to converge first
            train(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        if args.random_start:
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        else:
            random_noise = torch.zeros(*args.noise_shape)
        if args.perturb_type == 'samplewise':
            noise = sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, args.band_fft, args.filter_type, args.mode, args.phi_var)
        elif args.perturb_type == 'classwise':
            noise = universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, args.band_fft, args.filter_type, args.mode, args.phi_var)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    else:
        raise('Not implemented yet')
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
