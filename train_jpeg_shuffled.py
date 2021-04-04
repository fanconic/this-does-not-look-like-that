# Train ProtoPNets on altered CUB-200-2011 dataset.
# Used for the JPEG Experiment.

import os
import shutil
from io import BytesIO
from PIL import Image, ImageFile
import random
random.seed(42)

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from src.utils.helpers import makedir
from src.models import model
from src.training import push
from src.training import prune
from src.training import train_and_test as tnt

from src.utils import save
from src.utils.log import create_logger
from src.data.preprocess import mean, std, preprocess_input_function



parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])



# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run
from settings import colab, username

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

if colab:
    model_dir = '/content/PPNet/saved_models/' + base_architecture + '/' + experiment_run + '/'
else:
    model_dir = '/cluster/scratch/' + username + '/PPNet/saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'src/models/', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'src/models/', 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'src/training/', 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'



# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size, \
                     JPEG_QUALITY

from src.data.customdataset import CustomImageFolder

def randomJPEGcompression(image):
    """
    Applies JPEG compression of quality qf if the given image is of the given target with propability 0.5
    Args:
        image: single image that is compressed with jpeg noise
    """
    compress = bool(random.getrandbits(1))
    if compress:
        outputIoStream = BytesIO()
        image.save(outputIoStream, "JPEG", quality=JPEG_QUALITY, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)
    else:
        return image

def JPEGcompression(image):
    """
    Applies JPEG compression of quality qf if the given image is of the given target
    Args:
        image: single image that is compressed with jpeg noise
    """
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=JPEG_QUALITY, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

normalize = transforms.Normalize(mean=mean, std=std)

# all datasets

# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Lambda(randomJPEGcompression),
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
    
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# test set
test_dataset_uncompressed = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader_uncompressed = torch.utils.data.DataLoader(
    test_dataset_uncompressed, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

test_dataset_compressed = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Lambda(JPEGcompression),
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader_compressed = torch.utils.data.DataLoader(
    test_dataset_compressed, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)


# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size (uncompressed): {0}'.format(len(test_loader_uncompressed.dataset)))
log('test set size (compressed): {0}'.format(len(test_loader_compressed.dataset)))
log('batch size: {0}'.format(train_batch_size))



# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True



# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)



# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    log("\tUncompressed:")
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader_uncompressed,
                    class_specific=class_specific, log=log)

    log("\tCompressed:")
    _ = tnt.test(model=ppnet_multi, dataloader=test_loader_compressed,
                    class_specific=class_specific, log=log)

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        log("\tUncompressed:")
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader_uncompressed,
                        class_specific=class_specific, log=log)

        log("\tCompressed:")
        _ = tnt.test(model=ppnet_multi, dataloader=test_loader_compressed,
                        class_specific=class_specific, log=log)

        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                log("\tUncompressed:")
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader_uncompressed,
                                class_specific=class_specific, log=log)

                log("\tCompressed:")
                _ = tnt.test(model=ppnet_multi, dataloader=test_loader_compressed,
                                class_specific=class_specific, log=log)
                                
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', 
                                            accu=accu, target_accu=0.70, log=log)


logclose()


