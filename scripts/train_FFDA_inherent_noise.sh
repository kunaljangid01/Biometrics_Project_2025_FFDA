#!/bin/bash


# Search Universal Perturbation and build datasets
cd ../ 


#CIFAR-10   32x32     Noise from Sobel filter, add at high frequency
python3 face_perturbation_freq_advanced_128_filter.py --config_path             configs/cifar10                \
                        --exp_name                output/cifar10/classwise_noise/generate_noise_high_freq_sobel_filter_add_phi0.5_cifar10 \
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
                        --use_subset                        \
                        --band_fft high   \
                        --filter_type sobel                        \
                        --mode add                            \
                        --phi_var  0.5

python3 -u main.py    --version                 resnet18                       \
                      --exp_name                output/cifar10/classwise_noise/generate_noise_high_freq_sobel_filter_add_phi0.5_cifar10 \
                      --config_path             configs/cifar10                \
                      --train_data_type         PoisonCIFAR10                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/cifar10/classwise_noise/generate_noise_high_freq_sobel_filter_add_phi0.5_cifar10/perturbation.pt \
                      --train




#PubFig  128x128      Noise from average filter, add at high frequency
python3 face_perturbation_freq_advanced_128_filter.py --config_path             configs/face                \
                        --exp_name                output/PubFig/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_PubFig_128 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             150 3 128 128                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                           \
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
                        --universal_train_target train_dataset     \
                        --band_fft high   \
                        --filter_type average                        \
                        --mode add                            \
                        --phi_var   0.5


python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/PubFig/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_PubFig_128 \
                      --config_path             configs/face                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/PubFig/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_PubFig_128/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  \





#PubFig  128x128      Noise from laplace filter, subtract at high frequency
python3 face_perturbation_freq_advanced_128_filter.py --config_path             configs/face                \
                        --exp_name                output/PubFig/classwise_noise/generate_noise_high_freq_laplace_filter_subtract_phi0.5_PubFig_128 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             150 3 128 128                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                           \
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
                        --universal_train_target train_dataset     \
                        --band_fft high   \
                        --filter_type laplace                        \
                        --mode subtract                            \
                        --phi_var   0.5
                        


python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/PubFig/classwise_noise/generate_noise_high_freq_laplace_filter_subtract_phi0.5_PubFig_128 \
                      --config_path             configs/face                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/PubFig/classwise_noise/generate_noise_high_freq_laplace_filter_subtract_phi0.5_PubFig_128/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/train  \
                      --test_data_path /home/dse622/kunal/data/PubFig/CelebDataProcessed_split/test  \





#Pins-105  128x128      Noise from average filter, add at high frequency
python3 face_perturbation_freq_advanced_128_filter.py --config_path             configs/pins                \
                        --exp_name                output/pins/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_pins105_128 \
                        --version                 resnet18                       \
                        --train_data_type         PubFig                       \
                        --noise_shape             150 3 128 128                     \
                        --epsilon                 8                              \
                        --num_steps               1                              \
                        --step_size               0.8                           \
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
                        --universal_train_target train_dataset     \
                        --band_fft high   \
                        --filter_type average                        \
                        --mode add                            \
                        --phi_var   0.5

  
python3 -u main_128.py    --version                 resnet18                       \
                      --exp_name                output/pins/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_pins105_128 \
                      --config_path             configs/pins                \
                      --train_data_type         PoisonPubFig                  \
                      --poison_rate             1.0                            \
                      --perturb_type            classwise                      \
                      --perturb_tensor_filepath output/pins/classwise_noise/generate_noise_high_freq_average_filter_add_phi0.5_pins105_128/perturbation.pt \
                      --train   \
                      --test_data_type  PubFig        \
                      --train_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/train  \
                      --test_data_path /home/dse622/kunal/data/105_classes_pins_dataset_split/test  \











