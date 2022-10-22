from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

import json
import glob


mapx = np.load('src/calib_eqc/mapx_gated_left.npz')['arr_0']
mapy = np.load('src/calib_eqc/mapy_gated_left.npz')['arr_0']

def passive_loader(path, crop_size_h, crop_size_w,
                    img_ext='png',
                    num_bits=10, data_type='real',
                    scale_images=False,
                    scaled_img_width=None, scaled_img_height=None):
    normalizer = 2 ** num_bits - 1.

    
    assert os.path.exists(path), "No such file : %s" % path
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)
    if data_type == 'real':
        img = img[crop_size_h:(img.shape[0] - crop_size_h),
              crop_size_w:(img.shape[1] - crop_size_w)
              ]

        img = img.copy()
        img[img > 2 ** 10 - 1] = normalizer

    img = np.float32(img / normalizer)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img


def gated_loader(paths, crop_size_h, crop_size_w, 
                 img_ext='png',   
                 num_bits=10, data_type='real',   
                 scale_images=False,
                 scaled_img_width=None, scaled_img_height=None):
    gated_imgs = []
    normalizer = 2 ** num_bits - 1.

    for path in paths:
        assert os.path.exists(path),"No such file : %s"%path 
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)
        if data_type == 'real':
            img = img[ crop_size_h:(img.shape[0] - crop_size_h),
                       crop_size_w:(img.shape[1] - crop_size_w)
                     ]
            
            img = img.copy()
            img[img > 2 ** 10 - 1] = normalizer
        
        img = np.float32(img / normalizer)
        gated_imgs.append(np.expand_dims(img, axis=2))
    img = np.concatenate(gated_imgs, axis=2)
    if scale_images:
        img = cv2.resize(img, dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA)
    return img

class GatedStereoDataset(data.Dataset):

    def __init__(self,
                 gated_dir,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='tiff',
                 load_passive = False):
        super(GatedStereoDataset, self).__init__()
        
        self.root_dir = gated_dir
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.img_ext = img_ext
        self.json_path = 'src/docs/recording_info_gatedstereo.json'

        print("Loading gated type info from JSON file {} ....".format(self.json_file))
        with open(self.json_file, 'r') as fh:
            self.json_data = json.load(fh)
        print("Loading complete.")

        print("Loading Filenames ...")
        self.filenames_dicts = self.get_filenames_dicts()
        print("Loading complete.")


        self.full_res_shape = (1280, 720)
        self.crop_size_h, self.crop_size_w = int((self.full_res_shape[1]-self.height)/2), int((self.full_res_shape[0]-self.width)/2),

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = gated_loader
        self.interp = Image.ANTIALIAS
        self.load_passive = load_passive
        if self.load_passive:
            self.passive_loader = passive_loader
            

        self.to_tensor = transforms.ToTensor()

        self.resize = {}

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.K = np.array([[2327.21974 / 1280., 0, 668.001905 / 1280., 0],
                           [0, 2321.645198 / 720., 261.468935 / 720., 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        
        self.load_depth = self.check_depth()

    def get_filenames_dicts(self):
        filenames_dicts = []
        for filename in self.filenames:
            rec = filename.split(',')[0]
            idx = filename.split(',')[1]

            filenames_dict = {}
            filenames_dict['rec'] = rec
            filenames_dict['idx'] = idx
            filenames_dict['lidar'] = glob.glob(os.path.join(self.root_dir, rec, self.json_data[rec]['lidar'][1:], idx + '*'))[0]
            filenames_dict['passive'] = glob.glob(os.path.join(self.root_dir, rec, self.json_data[rec]['gated_passive_topic'][1:].format('left'), idx + '*'))[0]

            filenames_dict['gated'] = {}
            for frame_idx in self.frame_idxs:
                curr_idx = str(int(idx) + frame_idx).zfill(5) 
                slice_paths = []
                for slice in self.json_data['slices']:
                    slice_paths.append(glob.glob(os.path.join(self.root_dir, rec, slice[1:].format('left'), curr_idx + '*'))[0])
                filenames_dict['gated'][str(frame_idx)] = slice_paths
        return filenames_dicts


    def __getitem__(self, index):
        
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5

        # line = self.filenames[index].split()
        self.filenames_dict = self.filenames_dicts[index]
        

        inputs['frame_info'] = "{}-{}".format(self.filenames_dict['rec'],self.filenames_dicts['idx'])

        for frame_idx in self.frame_idxs:
            inputs[("gated", i, -1)] = self.get_gated(frame_idx,do_flip)


        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        gated_aug = (lambda x: x)
        self.preprocess(inputs, gated_aug)

        for i in self.frame_idxs:
            del inputs[("gated", i, -1)]
            del inputs[("gated_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(do_flip)
            inputs["depth_gt"] = torch.from_numpy(depth_gt)

        if self.load_passive:
            passive = self.get_passive(do_flip)
            inputs["passive"] = torch.from_numpy(passive)

        

        return inputs        

    def preprocess(self, inputs, gated_aug):
        """
            Resize gated images to the required scales and augment if required

            We create the gated_aug object in advance and apply the same augmentation to all
            images in this item. This ensures that all images input to the pose network receive the
            same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "gated" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
                    s = 2 ** i
                    scaled_img_width, scaled_img_height = self.width // s, self.height // s
                    inputs[(n, im, i)] = cv2.resize(inputs[(n, im, i - 1)], dsize=(scaled_img_width, scaled_img_height), interpolation=cv2.INTER_AREA) 

        for k in list(inputs):
            f = inputs[k]
            if "gated" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(gated_aug(f))

    def __len__(self):
        return len(self.filenames)

    def get_gated(self, frame_index, do_flip):
        gated_paths = self.filenames_dict['gated'][str(frame_index)]
        gated = self.loader(gated_paths, self.crop_size_h, self.crop_size_w, img_ext=self.img_ext)
        if do_flip:
            gated = np.fliplr(gated).copy()
        return gated

    def get_passive(self, do_flip):
        passive_path = self.filenames_dicts['passive']
        passive = self.passive_loader(passive_path, self.crop_size_h, self.crop_size_w, img_ext=self.img_ext)
        passive = np.clip((passive - 87./1023.)* self.json_data[self.filenames_dicts['rec']]['passive_factor'], 0., 1.)
        if do_flip:
            passive = np.fliplr(passive).copy()
        passive = np.expand_dims(passive, 0).astype(np.float32)
        return passive

    def get_depth(self, do_flip):
        lidar_filename = self.filenames_dict['lidar']
        depth_gt = np.load(lidar_filename)['arr_0']
        depth_gt = depth_gt[self.crop_size_h:self.full_res_shape[1] - self.crop_size_h, self.crop_size_w:self.full_res_shape[0] - self.crop_size_w]
        
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt).copy()

        depth_gt = np.expand_dims(depth_gt, 0).astype(np.float32)
        return depth_gt



    def check_depth(self):
        sample = self.filenames[0].split(',')
        rec = sample[0]
        frame_index = sample[1]
        lidar_filename = glob.glob(os.path.join(self.root_dir, rec, self.json_data['lidar'][1:], frame_index + '*'))[0]
        return os.path.isfile(lidar_filename)


    
        
