import os
import shutil

# paths = ['./results/DP/IG_CIFAR10_0.5_1e-05',
#          './results/DP/IG_CIFAR10_2.0_1e-05',
#          './results/DP/IG_CIFAR10_8.0_1e-05',
#          './results/GradPrune_0.7/IG_CIFAR10',
#          './results/GradPrune_0.9/IG_CIFAR10',
#          './results/GradPrune_0.99/IG_CIFAR10',]
# paths = ['./results/DP/IG_STL10_0.5_1e-05',
#          './results/DP/IG_STL10_2.0_1e-05',
#          './results/DP/IG_STL10_8.0_1e-05',
#          './results/GradPrune_0.7/IG_STL10_GradPrune_0.7',
#          './results/GradPrune_0.9/IG_STL10_GradPrune_0.9',
#          './results/GradPrune_0.99/IG_STL10_GradPrune_0.99',
#          './results/NoDefense/IG_STL10',]
paths = ['./results/DP/IG_MITBIH_0.5_1e-05',
         './results/DP/IG_MITBIH_2.0_1e-05',
         './results/DP/IG_MITBIH_8.0_1e-05',
         './results/GradPrune_0.7/IG_MITBIH_GradPrune_0.7',
         './results/GradPrune_0.9/IG_MITBIH_GradPrune_0.9',
         './results/GradPrune_0.99/IG_MITBIH_GradPrune_0.99',
         './results/NoDefense/IG_MITBIH',]

name = ['dp-large', 'dp-medium', 'dp-small', 'gp70', 'gp90', 'gp99']

file_idx = 2
for path, n in zip(paths, name):
    # shutil.copyfile(f'{path}/{file_idx}_result.png', f'./results/stl10/{file_idx}_result_{n}.png')
    shutil.copyfile(f'{path}/{file_idx}_result.npy', f'./results/MITBIH/{file_idx}_result_{n}.npy')

shutil.copyfile(f'./results/EISFL/IG_MITBIH/{file_idx}_result.npy', f'./results/MITBIH/{file_idx}-eisfl.npy')