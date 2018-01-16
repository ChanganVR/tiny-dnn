# Cifar example with pruned layers
## build
mkdir build && cd build
cmake ../../.. -DBUILD_EXPERIMENTS=ON
make
cd .. & ./build/experiments/pruned_cifar10_test path/to/image

## get validation accuracy
./build/experiments/pruned_cifar10_test cifar-weights -V /local-scratch/changan-home/dataset/cifar-10-batches-bin
The accuracy of pretrained weights is 70.11%

## fine tune pruned network
./build/experiments/pruned_cifar10_fine_tune \
--data_path /local-scratch/changan-home/dataset/cifar-10-batches-bin \
--weights_path cifar_weights \
--learning_rate 0.01 \
--epochs 2 \
--minibatch_size 10 \
--backend_type internal \
--pruning_percentage 0.01