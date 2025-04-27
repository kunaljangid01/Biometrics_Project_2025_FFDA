#!/bin/bash


# Search Universal Perturbation and build datasets
cd ../ 

#CIFAR-10
python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/cifar10/classwise_noise/cifar10_clean \
                      --config_path             configs/cifar10                \
                      --train_data_type         CIFAR10                 \
                      --test_data_type      CIFAR10        \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --train



#CIFAR-100
python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/cifar100/classwise_noise/cifar100_clean \
                      --config_path             configs/cifar100                \
                      --train_data_type         CIFAR100                  \
                      --test_data_type      CIFAR100        \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --train



#PubFig  32x32
python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/PubFig/classwise_noise/PubFig_clean_32 \
                      --config_path             configs/face                \
                      --train_data_type         PubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/PubFig/classwise_noise/PubFig_clean_32/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  \





# PubFig    128x128
python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/PubFig/classwise_noise/PubFig_clean_128 \
                      --config_path             configs/face                \
                      --train_data_type         PubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath None \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  \




#Pins-105    128x128
python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/pins/classwise_noise/pins105_clean_128 \
                      --config_path             configs/pins                \
                      --train_data_type         PubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath None \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/train  \
                      --test_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/test  \

