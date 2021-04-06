# Configuration for training ProtoPNets.

import os
import getpass


username = getpass.getuser()

base_architecture = "resnet34"
img_size = 224
if base_architecture in ["resnet34"]:
    num_channels = 256
else:
    num_channels = 128

prototype_shape = (2000, num_channels, 1, 1)
num_classes = 200
prototype_activation_function = "log"
add_on_layers_type = "regular"

<<<<<<< HEAD
experiment_run = "105"
=======
experiment_run = "010"
>>>>>>> 5c02e325b69b51fbb3bc93d411b18bac68ab63a4

JPEG_QUALITY = 20


if "COLAB_GPU" in os.environ:
    data_path = "/content/PPNet/datasets/cub200_cropped/"
    pretrained_model_dir = "/content/PPNet/pretrained_models/"
    colab = True
else:
    data_path = "/scratch/PPNet/datasets/cub200_cropped/"
    pretrained_model_dir = "/cluster/scratch/{}/PPNet/pretrained_models/".format(
        username
    )
    colab = False

train_dir = data_path + "train_cropped_augmented/"
test_dir = data_path + "test_cropped/"
train_push_dir = data_path + "train_cropped/"
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {
    "features": 1e-4,
    "add_on_layers": 3e-3,
    "prototype_vectors": 3e-3,
}
joint_lr_step_size = 5

warm_optimizer_lrs = {"add_on_layers": 3e-3, "prototype_vectors": 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    "crs_ent": 1,
    "clst": 0.8,
    "sep": -0.08,
    "l1": 1e-4,
}

num_train_epochs = 21
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]


# configuration for adversarial training
epsilon = 8.0
alpha = 10.0
pgd_alpha = 2.0
pgd_attack_iters = 10
