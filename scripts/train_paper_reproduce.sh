#!/bin/bash


# Search Universal Perturbation and build datasets
cd ../ 


#CIFAR-10
python3 perturbation.py --config_path             configs/cifar10                \
                        --exp_name                output/cifar10/classwise_noise/reproduce_cifar10 \
                        --version                 resnet18                       \
                        --train_data_type         CIFAR10                       \
                        --noise_shape             10 3 32 32                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            classwise                      \
                        --universal_train_target  'train_subset'                 \
                        --universal_stop_error    0.1                            \
                        --use_subset


python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/cifar10/classwise_noise/reproduce_cifar10 \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/cifar10/classwise_noise/reproduce_cifar10/perturbation.pt \
                      --train




#CIFAR-100
python3 perturbation.py --config_path             configs/cifar100                \
                        --exp_name                output/cifar100/classwise_noise/reproduce_cifar100 \
                        --version                 resnet18                       \
                        --train_data_type         CIFAR100                       \
                        --test_data_type          CIFAR100                    \
                        --noise_shape             100 3 32 32                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            classwise                      \
                        --universal_train_target  'train_subset'                 \
                        --universal_stop_error    0.01                            \
                        --use_subset


python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/cifar100/classwise_noise/reproduce_cifar100 \
                      --config_path             configs/cifar100                \
                      --train_data_type         PoisonCIFAR100                  \
                      --test_data_type          CIFAR100                    \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/cifar100/classwise_noise/reproduce_cifar100/perturbation.pt \
                      --train





#PubFig   32x32
python3 face_perturbation.py --config_path             configs/face                \
                        --exp_name                output/PubFig/classwise_noise/reproduce_PubFig_32 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             150 3 32 32                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            classwise                      \
                        --universal_train_target  'train_subset'                 \
                        --universal_stop_error    0.1                            \
                        --use_subset    \
                        --test_data_type  PubFig        \
                        --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                        --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test     \
                        --train_batch_size        32                 \
                        --eval_batch_size         32                 \
                        --train_step 30  \
                        --universal_stop_error 0.1 \
                        --universal_train_target train_dataset       \


python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/PubFig/classwise_noise/reproduce_PubFig_32 \
                      --config_path             configs/face                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/PubFig/classwise_noise/reproduce_PubFig_32/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  \
                    #   --train_face







#PubFig   128x128
python3 face_perturbation_128.py --config_path             configs/face                \
                        --exp_name                output/PubFig/classwise_noise/reproduce_PubFig_128 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             150 3 128 128                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            classwise                      \
                        --universal_train_target  'train_subset'                 \
                        --universal_stop_error    0.1                            \
                        --use_subset    \
                        --test_data_type  PubFig        \
                        --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                        --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test     \
                        --train_batch_size        32                 \
                        --eval_batch_size         32                 \
                        --train_step 30  \
                        --universal_stop_error 0.1 \
                        --universal_train_target train_dataset


python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/PubFige/classwise_noise/reproduce_PubFig_128 \
                      --config_path             configs/face                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/PubFig/classwise_noise/reproduce_PubFig_128/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  






#Pins-105   128X128
python3 face_perturbation_128.py --config_path             configs/pins                \
                        --exp_name                output/pins/classwise_noise/reproduce_pins105_128 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             105 3 128 128                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                            \
                        --attack_type             min-min                        \
                        --perturb_type            classwise                      \
                        --universal_train_target  'train_subset'                 \
                        --universal_stop_error    0.1                            \
                        --use_subset    \
                        --test_data_type  PubFig        \
                        --train_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/train  \
                        --test_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/test     \
                        --train_batch_size        32                 \
                        --eval_batch_size         32                 \
                        --train_step 30  \
                        --universal_stop_error 0.1 \
                        --universal_train_target train_dataset


python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/pins/classwise_noise/reproduce_pins105_128 \
                      --config_path             configs/pins                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/pins/classwise_noise/reproduce_pins105_128/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/train  \
                      --test_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/test  




