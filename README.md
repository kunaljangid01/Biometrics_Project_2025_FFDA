
# Biometrics Project 2025 (DSE-622)   by KUNAL JANGID


# FFDA: Fourier-Based Frequency Domain Attacks for Optimizing Unlearnable Datasets 


Initial Code taken from ["Unlearnable Examples: Making Personal Data Unexploitable "](https://openreview.net/forum?id=iAmZUo0DxC0) by Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.


Download the pretrained checkpoint (20180408-102900-casia-webface.pt) from the link given below and put it in this directory.
https://drive.google.com/file/d/142xmAIzllQh40c22tGSsWH4JfsbI52AY/view?usp=sharing


## Experiments.
Check scripts folder for *.sh for each corresponding experiments.


## Reproduce the Unlearnable Examples papers results on Class-wise noise for generating unlearnable example on CIFAR-10, CIFAR-100, PubFig, Pins-100 datasets. All code commands are written in given below bash file.
```console
bash train_paper_reproduce.sh
```


## Results on the clean dataset on CIFAR-10, CIFAR-100, PubFig, Pins-100 datasets.
```console
bash train_clean.sh
```

## I have applied the proposed framework in two ways:  Case-1: Where I get noise (randomly) from the image in the frequency domain and add in frequency domain itself.  In case-2: Where I get the inherent noise of the image (apply filter and get the features, then remove that feature from the image) in spatial domain, then add/subtract it from the image after convert both (image and noise) in frequency domain.

## Case-1: Results of proposed FFDA method on Class-wise noise for generating unlearnable example using the frequency domain on CIFAR-10, CIFAR-100, PubFig, Pins-100 datasets. In this, we get a get the noise from the frequency domain and add them with the clean image using the fft_perturbation(). This function takes three parameters (images, epsilon, band), where images is the clean image, epsilon is a hyperparameter for multiple with the noise and band is also a hyperparameter for where to add noise in frequency (high, low, mid, randomly). For running code all datasets, commands are given in below bash file.

```console
bash train_FFDA.sh
```

## Case-2: Results of proposed FFDA method on Class-wise noise for generating unlearnable example using the frequency domain on CIFAR-10, CIFAR-100, PubFig, Pins-100 datasets. In this, we use inherent noise of a image to make them unlearnable. This is done in the frequency domain using ciper_fft_perturbation_kornia(). This function takes five parameters (image, phi, band, filter_type, mode), where images is the clean image, phi is a hyperparameter for multiple with the inherent noise get from apply_filter_kornia() function, band is a hyperparameter for where to add noise in frequency (high, low, mid, randomly), filter_type is a hyperparameter that give the type of filter we have to apply to get the inherent noise, and mode is a hyperparameter that gives that we have to add or subtract the noise from image. apply_filter_kornia() function have two parameters (image, filter_type), this function works as first, apply the filter (chosen by a user) on the image to get the features and subtract them from clean image, to get the noise. For running code all datasets, commands are given in below bash file.

```console
bash train_FFDA_inherent_noise.sh
```




@Kunal Jangid


