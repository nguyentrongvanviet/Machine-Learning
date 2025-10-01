from mnist import MNIST

mndata = MNIST('./dir_with_mnist_data_files') # Specify directory for data files
images, labels = mndata.load_training()