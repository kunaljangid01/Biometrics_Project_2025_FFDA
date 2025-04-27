import time
import models
import torch
import util

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import torch.nn.functional as F

# def frequency_aware_loss(model, original_images, perturbed_images, labels):
#     """
#     Compute the frequency-aware loss as KL divergence between model predictions on original and perturbed images.
#     """
#     # Forward pass for original images
#     model_output_original = model(original_images)  # Shape: [batch_size, num_classes]
#     prob_original = F.softmax(model_output_original, dim=1)

#     # Forward pass for perturbed images
#     model_output_perturbed = model(perturbed_images)  # Shape: [batch_size, num_classes]
#     prob_perturbed = F.softmax(model_output_perturbed, dim=1)

#     # Compute KL Divergence
#     kl_loss = F.kl_div(prob_perturbed.log(), prob_original, reduction='batchmean')  # Batch mean of KL divergence

#     return kl_loss

def frequency_aware_loss(clean_images, perturbed_images, lambda_freq=0.1):
    """
    Frequency-aware loss function that combines:
    1. A visual domain loss (to ensure the inverse FFT of perturbed images is similar to the original images).
    2. A frequency domain loss (to ensure the FFT of perturbed images is significantly different from the original images).

    Args:
    - clean_images: Original unperturbed images.
    - perturbed_images: Perturbed images after applying FFT-based perturbations.
    - lambda_freq: Weight for the frequency domain loss.

    Returns:
    - total_loss: A combination of the visual domain loss and frequency domain loss.
    """

    # 1. Visual Domain Loss: Minimize difference between clean images and perturbed images after inverse FFT.
    # perturbed_images_ifft = torch.fft.ifftn(perturbed_images, dim=(-2, -1)).real  # Apply IFFT to perturbed images
    visual_loss = torch.nn.functional.mse_loss(perturbed_images, clean_images)  # MSE between clean and IFFT perturbed image

    # 2. Frequency Domain Loss: Minimize difference between FFT of clean images and perturbed images.
    clean_fft = torch.fft.fftn(clean_images, dim=(-2, -1))  # FFT of the original clean images
    perturbed_fft = torch.fft.fftn(perturbed_images, dim=(-2, -1))  # FFT of the perturbed images
    freq_loss = torch.abs(clean_fft - perturbed_fft).mean()  # Mean absolute difference in FFTs

    # Combine both losses
    total_loss = visual_loss + lambda_freq * freq_loss
    return total_loss



class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer, random_noise=None):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            unchanged_img = images
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            log_payload = self.train_batch(unchanged_img,images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, unperturbed_images, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        
        # Compute the frequency-aware loss
        # freq_loss = frequency_aware_loss(model, unperturbed_images, images, labels)
        # Calculate the frequency-aware loss (combined loss)
        freq_loss = 0#frequency_aware_loss(unperturbed_images, images, lambda_freq=0.7)
            
        # loss += 0.1*freq_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
