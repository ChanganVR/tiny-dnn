# Cifar example with pruned layers
## Usage
mkdir build & cd build
cmake ../../.. -DBUILD_EXPERIMENTS=ON
make
cp cifar_weights build/experiments
cd build/experiments
./pruned_cifar10_test path/to/image