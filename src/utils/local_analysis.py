import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2

import re
import os
import copy

from src.utils.helpers import makedir, find_high_activation_crop, torch2numpy, visualize_image_grid
from src.utils.helpers import get_image_patch_position
from src.utils.log import create_logger
from src.data.preprocess import mean, std, undo_preprocess_input_function

from src.utils.ct import ctx_noparamgrad_and_eval
from src.utils import attack1 as a1
from src.utils import attack3 as a3

import sys
sys.path.insert(0, '.')
from settings import colab, username


##### HELPER FUNCTIONS FOR PLOTTING

def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    # plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(load_img_dir, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    # plt.axis('off')
    # plt.imsave(fname, p_img)
    return p_img

def save_prototype_self_activation(load_img_dir, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    # plt.axis('off')
    # plt.imsave(fname, p_img)
    return p_img
    
def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    # plt.imshow(p_img_rgb)
    # plt.axis('off')
    # plt.imsave(fname, p_img_rgb)
    return p_img_rgb
    
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    # plt.imshow(img_rgb_float)
    # plt.axis('off')
    # plt.imsave(fname, img_rgb_float)
    return img_rgb_float


class LocalAnalysis(object):
    def __init__(self, load_model_dir, load_model_name, test_image_name, image_save_directory = None, attack=None):
        model_base_architecture = load_model_dir.split('/')[-3]
        experiment_run = load_model_dir.split('/')[-2]
        
        
        self.save_analysis_path = '/cluster/scratch/{}/PPNet/local_analysis_attack{}/'.format(username, attack) \
                                    + model_base_architecture + '-' + experiment_run + '/'  + test_image_name[:-4]
        
        if image_save_directory is not None:
            self.save_analysis_path = image_save_directory + test_image_name

        makedir(self.save_analysis_path)

        self.log, self.logclose = create_logger(log_filename=os.path.join(self.save_analysis_path, 'local_analysis.log'))

        load_model_path = os.path.join(load_model_dir, load_model_name)
        epoch_number_str = re.search(r'\d+', load_model_name).group(0)
        self.start_epoch_number = int(epoch_number_str)

        self.log('load model from ' + load_model_path)
        self.log('model base architecture: ' + model_base_architecture)
        self.log('experiment run: ' + experiment_run)

        self.ppnet = torch.load(load_model_path)
        self.ppnet = self.ppnet.cuda()
        self.ppnet_multi = torch.nn.DataParallel(self.ppnet)

        self.img_size = self.ppnet_multi.module.img_size
        prototype_shape = self.ppnet.prototype_shape
        self.max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        self.class_specific = True
        self.normalize = transforms.Normalize(mean=mean, std=std)
        
        # confirm prototype class identity
        self.load_img_dir = os.path.join(load_model_dir, 'img')

        self.prototype_info = np.load(os.path.join(self.load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
        self.prototype_img_identity = self.prototype_info[:, -1]

        self.log('Prototypes are chosen from ' + str(len(set(self.prototype_img_identity))) + ' number of classes.')
        self.log('Their class identities are: ' + str(self.prototype_img_identity))

        # confirm prototype connects most strongly to its own class
        prototype_max_connection = torch.argmax(self.ppnet.last_layer.weight, dim=0)
        self.prototype_max_connection = prototype_max_connection.cpu().numpy()
        if np.sum(self.prototype_max_connection == self.prototype_img_identity) == self.ppnet.num_prototypes:
            self.log('All prototypes connect most strongly to their respective classes.')
        else:
            self.log('WARNING: Not all prototypes connect most strongly to their respective classes.')
        
    
    def logclose(self):
        self.logclose()
    
    
    def local_analysis(self, img_variable, test_image_label, max_prototypes=10, idx=0, pid=None, verbose=False, 
                       show_images=True, normalize_sim_map=None):
        '''
        Perform local analysis.
        Arguments:
            img_variable (torch.Tensor): imput image to test on.
            test_image_label (int): true label of test image.
            max_prototypes (int): number of most similar prototypes to display (fefault: 10).
            idx (int): image id in the batch (default: 0).
            pid (int): prototype id.
            verbose (bool): whether to print (default: False).
            show_images (bool): show images (default: True).
            normalize_sim_map (Tuple(2)): min and max values resp. to be used for normalizing the activation map.
        '''
        img_size = self.ppnet_multi.module.img_size
        images_test = img_variable.cuda()
        labels_test = torch.tensor([test_image_label])

        logits, min_distances = self.ppnet_multi(images_test)
        conv_output, distances = self.ppnet.push_forward(images_test)
        prototype_activations = self.ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = self.ppnet.distance_2_similarity(distances)
        if self.ppnet.prototype_activation_function == 'linear':
            prototype_activations = prototype_activations + self.max_dist
            prototype_activation_patterns = prototype_activation_patterns + self.max_dist

        tables = []
        for i in range(logits.size(0)):
            tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))

        idx = idx
        predicted_cls = tables[idx][0]
        correct_cls = tables[idx][1]
        self.log('Predicted: ' + str(predicted_cls))
        self.log('Actual: ' + str(correct_cls))
        if predicted_cls == correct_cls:
            self.log('Prediction is correct.')
        else:
            self.log('Prediction is wrong.')

        original_img = save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img.png'),
                                             images_test, idx)

        ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
        makedir(os.path.join(self.save_analysis_path, 'most_activated_prototypes'))

        self.log('Most activated 10 prototypes of this image:')
        self.log('--------------------------------------------------------------')

        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
        for i in range(1, max_prototypes+1):
            if pid is not None and pid != sorted_indices_act[-i].item():
                continue
            self.log('top {0} activated prototype for this image:'.format(i))
            p_img = save_prototype(self.load_img_dir, os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                                                'top-%d_activated_prototype.png' % i), 
                                   self.start_epoch_number, sorted_indices_act[-i].item())
            p_oimg_with_bbox = save_prototype_original_img_with_bbox(
                self.load_img_dir,
                fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                                   'top-%d_activated_prototype_in_original_pimg.png' % i), 
                epoch=self.start_epoch_number,
                index=sorted_indices_act[-i].item(),
                bbox_height_start=self.prototype_info[sorted_indices_act[-i].item()][1],
                bbox_height_end=self.prototype_info[sorted_indices_act[-i].item()][2],
                bbox_width_start=self.prototype_info[sorted_indices_act[-i].item()][3],
                bbox_width_end=self.prototype_info[sorted_indices_act[-i].item()][4],
                color=(0, 255, 255))
            p_img_with_self_actn = save_prototype_self_activation(
                self.load_img_dir,
                os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                             'top-%d_activated_prototype_self_act.png' % i), 
                self.start_epoch_number, sorted_indices_act[-i].item())

            self.log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
            self.log('prototype class identity: {0}'.format(self.prototype_img_identity[sorted_indices_act[-i].item()]))
            if self.prototype_max_connection[sorted_indices_act[-i].item()] != self.prototype_img_identity[sorted_indices_act[-i].item()]:
                self.log('prototype connection identity: {0}'.format(self.prototype_max_connection[sorted_indices_act[-i].item()]))
            self.log('activation value (similarity score): {0}'.format(array_act[-i]))
            self.log('last layer connection with predicted class: {0}'.format(self.ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))

            activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC)

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                          high_act_patch_indices[2]:high_act_patch_indices[3], :]
            if verbose:
                self.log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                       high_act_patch)
            if verbose:
                log('most highly activated patch by this prototype shown in the original image:')
            p_img_with_bbox = imsave_with_bbox(
                fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                   'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i), 
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map
            if normalize_sim_map is not None:
                rescaled_activation_pattern = upsampled_activation_pattern - normalize_sim_map[0]
                rescaled_activation_pattern = rescaled_activation_pattern / normalize_sim_map[1]
            else:
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            if verbose:
                self.log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(self.save_analysis_path, 'most_activated_prototypes',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                       overlayed_img)

            if show_images:
                visualize_image_grid(images=[p_oimg_with_bbox, p_img, p_img_with_bbox, overlayed_img], 
                                     titles=['Training Image from which \nprototype is taken', 'Prototype', 
                                             'Test Image + BBox', 'Test Image + Activation Map'], ncols=4)
                plt.tight_layout()
                plt.show()
            self.log('--------------------------------------------------------------')

        return sorted_indices_act, prototype_activation_patterns, \
                [np.amin(upsampled_activation_pattern), np.amax(rescaled_activation_pattern)]
    
    def attack1(self, img_variable, loc, other_loc=None, i=1, idx=0):
        '''
        Perform attack 1 (e.g. make head appear at the stomach).
        Arguments:
            img_variable (torch.Tensor): imput image to attack.
            loc ([(int, int)]): list of coordinates of the patch which we want to perturb.
            other_loc ([(int, int)]): list of coordinates of the patch which we want to perturb but without increasing the 
                                      similarity with patch.
            i (int): index of the prototype for which we want to maximize the similarity.
            idx (int): image id in the batch (default: 0)
        '''
        pid = i # index of the prototype for which we want to maximize the similarity.
        idx = idx # image index

        images_test = img_variable.cuda()
        
        original_img = save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img.png'), 
                                             images_test, idx)

        self.log('Attacking prototype: {}'.format(pid))

        mask = np.zeros((self.img_size, self.img_size, 3))
        loc_img = get_image_patch_position(loc, self.img_size)
        mask[loc_img[0]:loc_img[1], loc_img[2]:loc_img[3], :] = 1

        if other_loc is not None: 
            other_loc_img = get_image_patch_position(other_loc, self.img_size)
            mask[other_loc_img[0]:other_loc_img[1], other_loc_img[2]:other_loc_img[3], :] = 1
        mask = np.moveaxis(mask, 2, 0)
        mask = torch.from_numpy(mask)
        mask = (mask > 0).cuda()

        clip_min, clip_max = 0.0, 1.0
        toTensor = transforms.ToTensor()
        images_test_unprocessed = undo_preprocess_input_function(img_variable).cuda()

        with ctx_noparamgrad_and_eval(self.ppnet_multi):
            sim, lmax, gmax, _ = a1.similarity_score(self.ppnet_multi, self.ppnet, self.normalize, images_test_unprocessed, 
                                                     pid, loc=loc)

            self.log('Similarity with prototype {} before attack is {:2f}.\tMax Similarity: {:2f}'.format(pid, lmax, gmax))
            images_perturbed, pert, score = a1.pgd(images_test_unprocessed, mask, pid, loc, self.ppnet_multi, self.ppnet, 
                                                   self.normalize, attack_steps=40, attack_lr=2/255, attack_eps=8/255, 
                                                   clip_min=clip_min, clip_max=clip_max, minimize=False, idx=0)

            sim, lmax, gmax, _ = a1.similarity_score(self.ppnet_multi, self.ppnet, self.normalize, images_perturbed, 
                                                     pid, loc=loc)
            self.log('Similarity with prototype {} after attack is {:2f}.\tMax Similarity: {:2f}'.format(pid, lmax, gmax))

        visualize_image_grid(torch2numpy, images=[images_test_unprocessed, images_perturbed, pert*127+0.5], 
                             titles=['Original Image', 'Perturbed Image', 'Perturbation'])
        return images_perturbed, pert
    
    
    def attack3(self, img_variable, i=1, pid=None, idx=0):
        '''
        Perform attack 3 (e.g. make head disappear).
        Arguments:
            img_variable (torch.Tensor): imput image to attack.
            i (int): rank of prototype to attack (in decreasing order of similarity e.g. most similar = 1).
            pid (int): prototype id to attack (optional).
            idx (int): image id in the batch (default: 0)
        '''
        i = i # rank of prototype to attack (in decreasing order of similarity e.g. most similar = 1)
        idx = idx # image index

        images_test = img_variable.cuda()
        
        original_img = save_preprocessed_img(os.path.join(self.save_analysis_path, 'original_img.png'), 
                                             images_test, idx)

        logits, min_distances = self.ppnet_multi(images_test)
        conv_output, distances = self.ppnet.push_forward(images_test)
        prototype_activations = self.ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = self.ppnet.distance_2_similarity(distances)
        if self.ppnet.prototype_activation_function == 'linear':
            prototype_activations = prototype_activations + self.max_dist
            prototype_activation_patterns = prototype_activation_patterns + self.max_dist

        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
        if pid is None:
            pid = sorted_indices_act[-i].item()
        print ('Attacking prototype: {}'.format(pid))

        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(self.img_size, self.img_size),
                                                  interpolation=cv2.INTER_CUBIC)

        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1], 
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]

        mask = np.zeros((self.img_size, self.img_size, 3))
        mask[high_act_patch_indices[0]:high_act_patch_indices[1], high_act_patch_indices[2]:high_act_patch_indices[3], :] = 1
        mask = np.moveaxis(mask, 2, 0)
        mask = torch.from_numpy(mask)
        mask = (mask > 0).cuda()

        clip_min, clip_max = 0.0, 1.0
        toTensor = transforms.ToTensor()
        images_test_unprocessed = undo_preprocess_input_function(img_variable).cuda()

        with ctx_noparamgrad_and_eval(self.ppnet_multi):
            sim = a3.similarity_score(self.ppnet_multi, self.ppnet, self.normalize, images_test_unprocessed, pid)
            print ('Similarity with prototype {} before attack is {:2f}'.format(pid, sim))

            images_perturbed, pert, score = a3.pgd(images_test_unprocessed, mask, pid, self.ppnet_multi, self.ppnet, 
                                                   self.normalize, attack_steps=40, attack_lr=2/255, attack_eps=8/255, 
                                                   clip_min=clip_min, clip_max=clip_max, minimize=True)

            sim = a3.similarity_score(self.ppnet_multi, self.ppnet, self.normalize, images_perturbed, pid)
            print ('Similarity with prototype {} after attack is {:2f}'.format(pid, sim))

        visualize_image_grid(torch2numpy, images=[images_test_unprocessed, images_perturbed, pert*127+0.5], 
                             titles=['Original Image', 'Perturbed Image', 'Perturbation'])
        return images_perturbed, pert


    def jpeg_visualization(self, pil_img, img_name, test_image_label, preprocess_clean, preprocess_compressed, 
                       show_images=False, idx=0, top_n = 1):
        '''
        Perform local analysis comparing compressed and uncompressed images.
        Args:
            pil_img: is the PIL test image which is to be inspected
            img_name: name of the image to save it
            test_image_label: label of the test image
            preprocess_clean: first torch vision transform pipeline (here, without compression)
            preprocess_compressed: second torch vision transform pipeline (here, with compression)
            show_images (default = False): Boolean value to show images
            idx (default = 0): Index Value, when retrieving results of the PPNet
            top_n (default = 1): Visualize the n_th most activating prototype

        Returns:
            A list containing 8 images in the following order:
                1. Full picture of most activated prototype with bounding box
                2. Most activated prototype of compressed image
                3. Compressed image passed through, with activated patch in bounding box
                4. Corresponding activation map of the compressed image

                5. Full picture of most activated prototype with bounding box
                6. Most activated prototype of compressed image
                7. Uncompressed image passed through, with activated patch in bounding box
                8. Corresponding activation map of the uncompressed image
        '''
        # How to save the images
        specific_folder = self.save_analysis_path + "/" + img_name
        makedir(specific_folder)

        # Preprocess Clean image
        img_tensor_clean = preprocess_clean(pil_img)
        img_variable_clean = Variable(img_tensor_clean.unsqueeze(0))

        # Preprocess compressed image
        img_tensor_compressed = preprocess_compressed(pil_img)
        img_variable_compressed = Variable(img_tensor_compressed.unsqueeze(0))
        
        # Initialize as none, and will be filled after examining the first image
        inspected_index = None
        inspected_min = None
        inspected_max = None

        img_variables =[img_variable_compressed, img_variable_clean]

        display_images = []
        for img_variable in img_variables:

            # Forward the image variable through the network
            images_test = img_variable.cuda()
            labels_test = torch.tensor([test_image_label])

            logits, min_distances = self.ppnet_multi(images_test)
            conv_output, distances = self.ppnet.push_forward(images_test)
            prototype_activations = self.ppnet.distance_2_similarity(min_distances)
            prototype_activation_patterns = self.ppnet.distance_2_similarity(distances)
            if self.ppnet.prototype_activation_function == 'linear':
                prototype_activations = prototype_activations + max_dist
                prototype_activation_patterns = prototype_activation_patterns + max_dist

            tables = []
            for i in range(logits.size(0)):
                tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))


            idx = idx
            predicted_cls = tables[idx][0]
            correct_cls = tables[idx][1]
            if predicted_cls == correct_cls:
                pred_text = 'Prediction is correct.'
            else:
                pred_text = 'Prediction is wrong.'
            self.log('Predicted: ' + str(predicted_cls) + '\t Actual: '+ str(correct_cls) + '\t ' + pred_text)
            
            original_img = save_preprocessed_img(os.path.join(specific_folder, 'original_img.png'),
                                                images_test, idx)

            ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
            makedir(os.path.join(specific_folder, 'most_activated_prototypes'))
            array_act, sorted_indices_act = torch.sort(prototype_activations[idx])

            i=top_n
            
            if inspected_index is None:
                inspected_index = sorted_indices_act[-i].item()

            self.log("protoype index: " + str(inspected_index))

            p_img = save_prototype(self.load_img_dir, os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                                                'top-%d_activated_prototype.png' % i), 
                                   self.start_epoch_number, inspected_index)

            p_oimg_with_bbox = save_prototype_original_img_with_bbox(
                self.load_img_dir,
                fname=os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                                   'top-%d_activated_prototype_in_original_pimg.png' % i), 
                epoch=self.start_epoch_number,
                index=inspected_index,
                bbox_height_start=self.prototype_info[inspected_index][1],
                bbox_height_end=self.prototype_info[inspected_index][2],
                bbox_width_start=self.prototype_info[inspected_index][3],
                bbox_width_end=self.prototype_info[inspected_index][4],
                color=(0, 255, 255))
            p_img_with_self_actn = save_prototype_self_activation(
                self.load_img_dir,
                os.path.join(self.save_analysis_path, 'most_activated_prototypes', 
                             'top-%d_activated_prototype_self_act.png' % i), 
                self.start_epoch_number, inspected_index)
        
            self.log('prototype class identity: {0}'.format(self.prototype_img_identity[inspected_index]))
            if self.prototype_max_connection[inspected_index] != self.prototype_img_identity[sorted_indices_act[-i].item()]:
                self.log('prototype connection identity: {0}'.format(self.prototype_max_connection[inspected_index]))
            self.log('activation value (similarity score): {0}'.format(prototype_activations[idx][inspected_index]))

            activation_pattern = prototype_activation_patterns[idx][inspected_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(self.img_size, self.img_size),
                                                    interpolation=cv2.INTER_CUBIC)

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            
            plt.imsave(os.path.join(specific_folder, 'most_activated_prototypes',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                    high_act_patch)
            
            p_img_with_bbox = imsave_with_bbox(
                fname=os.path.join(specific_folder, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i), 
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map and use normalization values of first run
            if inspected_min is None:
                inspected_min = np.amin(upsampled_activation_pattern)

            rescaled_activation_pattern = upsampled_activation_pattern - inspected_min

            if inspected_max is None:
                inspected_max = np.amax(rescaled_activation_pattern)
                
            rescaled_activation_pattern = rescaled_activation_pattern / inspected_max
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
    
            plt.imsave(os.path.join(specific_folder, 'most_activated_prototypes',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                    overlayed_img)
            
            display_images += [p_oimg_with_bbox, p_img, p_img_with_bbox, overlayed_img]
            self.log("------------------------------")
            
        # Visualize Images
        if show_images:
            display_titles = ['Training Image from which \nprototype is taken', 'Prototype', 
                                        'Test Image + BBox', 'Test Image + Activation Map']

            display_titles += [x + "\n(uncompressed)" for x in display_titles]

            visualize_image_grid(images=display_images, 
                                titles= display_titles, ncols=8)
            plt.tight_layout()
            plt.show()

        return display_images