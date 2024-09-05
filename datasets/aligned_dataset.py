import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
# from .image_folder import make_dataset
from .diff_quat import *
from PIL import Image

import torchvision
import blobfile as bf

from glob import glob

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
 

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic', align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


class EdgesDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir) # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:

            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory
            
            self.AB_paths = make_dataset(self.test_dir) # get image paths
            
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params =  get_params(A.size, self.resize_size, self.crop_size)

        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =self.random_crop, flip=self.random_flip)

        A = transform_image(A)
        B = transform_image(B)

        if not self.train:
            return  B, A, index, AB_path
        else:
            return B, A, index

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)




class DIODE(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True,  img_size=256, random_crop=False, random_flip=True, down_sample_img_size = 0, cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size
        
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.train = train

        self.filenames = [l for l in os.listdir(self.image_root) if not l.endswith('.pth') and not l.endswith('_depth.png') and not l.endswith('_normal.png')]

        self.cache_path = os.path.join(self.image_root, cache_name+f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        fn = self.filenames[index]
        img_path = os.path.join(self.image_root, fn)
        label_path = os.path.join(self.image_root, fn[:-4]+'_normal.png')

        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()
        pil_label = pil_label.convert("RGB")

        # apply the same transform to both A and B
        params =  get_params(pil_image.size, self.resize_size, self.crop_size)

        transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop =False, flip=self.random_flip)
        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =False, flip=self.random_flip)

        cond = transform_label(pil_label)
        img = transform_image(pil_image)

        # if self.down_sample_img:
        #     image_pil = np.array(image_pil).astype(np.uint8)
        #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
        #     down_sampled_image = get_tensor()(down_sampled_image)
        #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
        #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

        #     return image_tensor, data_dict
        if not self.train:
            return img, cond, index, fn
        else:
            return img, cond, index
        
    

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)
    


import joblib

class MotionDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, recycle_data_path, retarget_data_path, train=True, human_data_path = None, load_pose = False, norm = False, overlap=False,):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        # data_list_A = joblib.load(data_path_A)
        # data_list_B = joblib.load(data_path_B)
        self.jt_A = []
        self.jt_B = []
        self.jt_C = []
        self.root_A = []
        self.root_B = []
        self.root_C = []
        motion_length = []
        self.names = []
        self.load_pose = load_pose
        h1_num_bodies = 22
        human_num_bodies = 24
        self.window_size = 24
        self.overlap = overlap
        # start_id = []
        # current_id = 0
        # for (name, data_A), data_B in zip(data_list_A.items(), data_list_B):
        #     if data_B is not None:
        #         target_jt_A = torch.from_numpy(data_A['jt'])#.to(device)
        #         target_global_pos_A = torch.from_numpy(data_A['global'])[:,:3]#.to(device)
        #         target_global_ori_A = torch.from_numpy(data_A['global'])[:,20*3:20*3+6]#.to(device)
        #         target_jt_B = torch.from_numpy(data_B['jt'])#.to(device)
        #         target_global_pos_B = torch.from_numpy(data_B['global'])[:,:3]#.to(device)
        #         target_global_ori_B = torch.from_numpy(data_B['global'])[:,20*3:20*3+6]#.to(device)
        #         self.jt_A.append(target_jt_A)
        #         self.jt_B.append(target_jt_B)
        #         self.root_A.append(torch.concat([target_global_pos_A, target_global_ori_A], dim=1))
        #         self.root_B.append(torch.concat([target_global_pos_B, target_global_ori_B], dim=1))
        #         motion_length.append(target_jt_A.shape[0])
        # # target_jt = torch.cat(target_jt, dim=0).to(device)
        # # target_global = torch.cat(target_global, dim=0).to(device)
        # start_id = torch.zeros_like(target_length, dtype=torch.long)
        # start_id[1:] = torch.cumsum(target_length[:-1], dim=0)
        recycle_data_dict = joblib.load(recycle_data_path)
        retarget_data_path = joblib.load(retarget_data_path)
        self.load_human = human_data_path is not None
        if human_data_path is not None:
            human_data_dict = joblib.load(human_data_path)
            # import pytorch_kinematics as pk
            # chain = pk.build_chain_from_urdf(open("/home/ubuntu/workspace/H1_RL/HST/legged_gym/resources/robots/h1/urdf/h1_add_hand_link_for_pk.urdf","rb").read())
            # human_node_names=['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        for name, recycle_data in recycle_data_dict.items():
            # if data_pair is None:
            #     continue
            # name = data_pair['name']
            self.names.append(name)
        
            retarget_jt = torch.from_numpy(retarget_data_path[name]['jt'])[:,:19]#.to(device)
            retarget_global_pos = torch.from_numpy(retarget_data_path[name]['global'])[:,:3]#.to(device)
            retarget_global_ori = torch.from_numpy(retarget_data_path[name]['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)
            retarget_root = torch.concat([retarget_global_pos, retarget_global_ori], dim=1)
                # breakpoint()
            recycle_jt = torch.from_numpy(recycle_data['jt'])[:,:19]#.to(device)
            recycle_global_pos = torch.from_numpy(recycle_data['global'])[:,:3]#.to(device)
            recycle_global_ori = torch.from_numpy(recycle_data['global'])[:,h1_num_bodies*3:h1_num_bodies*3+6]#.to(device)

            # ret = chain.forward_kinematics(target_jt_B)
            # look up the transform for a specific link
            # left_hand_link = ret['left_hand_link']
            # left_hand_tg = left_hand_link.get_matrix()[:,:3,3]
            # right_hand_link = ret['right_hand_link']
            # right_hand_tg = right_hand_link.get_matrix()[:,:3,3]
            # left_ankle_link = ret['left_ankle_link']
            # left_ankle_tg = left_ankle_link.get_matrix()[:,:3,3]
            # right_ankle_link = ret['right_ankle_link']
            # right_ankle_tg = right_ankle_link.get_matrix()[:,:3,3]
            # get transform matrix (1,4,4), then convert to separate position and unit quaternion

            if human_data_path is not None:
                human_jt = human_data_dict[name]['local_rotation'].reshape(-1,(human_num_bodies-1)*6)
                # target_jt_A[:,0] = 0
                human_root = human_data_dict[name]['root_transformation']
                self.jt_C.append(human_jt)
                self.root_C.append(human_root)
                # breakpoint()


            self.jt_A.append(retarget_jt)
            self.jt_B.append(recycle_jt)
            self.root_A.append(retarget_root)
            self.root_B.append(torch.concat([recycle_global_pos, recycle_global_ori], dim=1))
            motion_length.append(retarget_jt.shape[0])
            assert retarget_jt.shape[0] == recycle_jt.shape[0]


            
        self.jt_A = torch.cat(self.jt_A, dim=0)
        self.jt_B = torch.cat(self.jt_B, dim=0)
        self.root_A = torch.cat(self.root_A, dim=0)
        self.root_B = torch.cat(self.root_B, dim=0)
        self.jt_root_A = torch.concat([self.jt_A, self.root_A], dim=1).type(torch.float32)
        self.jt_root_B = torch.concat([self.jt_B, self.root_B], dim=1).type(torch.float32)
        if human_data_path is not None:
            self.jt_C = torch.cat(self.jt_C, dim=0)
            self.root_C = torch.cat(self.root_C, dim=0)
            self.jt_root_C = torch.concat([self.jt_C, self.root_C], dim=1).type(torch.float32)
            del self.jt_C, self.root_C
        # normalize
        self.mean_A = self.jt_root_A.mean(dim=0, keepdim=True)
        self.mean_B = self.jt_root_B.mean(dim=0, keepdim=True)
        self.std_A = self.jt_root_A.std(dim=0, keepdim=True)
        self.std_B = self.jt_root_B.std(dim=0, keepdim=True)
        if norm:
            self.jt_root_A = (self.jt_root_A - self.mean_A) / self.std_A
            self.jt_root_B = (self.jt_root_B - self.mean_B) / self.std_B
            
        
        # self.cov_xy = 0*(self.jt_root_B * self.jt_root_B).mean() + 0.5 #dim=0, keepdim=True
        self.cov_xy=None
        # breakpoint()
        del self.jt_A, self.jt_B, self.root_A, self.root_B
        self.motion_length = torch.tensor(motion_length, dtype=torch.long)
        self.length = len(motion_length)
        self.max_length = self.motion_length.max()
        # self.pad = torch.zeros((max_length, 19+3+6)).to(self.jt_A)
        self.start_id = torch.zeros_like(self.motion_length, dtype=torch.long)
        self.start_id[1:] = torch.cumsum(self.motion_length[:-1], dim=0)
                    
        self.train = train


    def __getitem__(self, index):
        # index=index%16
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B
            A (tensor) - - an motion in the input jt_A and root_A
            B (tensor) - - its corresponding target motion
        """
        motion_length = self.motion_length[index]
        while motion_length < self.window_size:
            index += 1
            index %= self.length
            motion_length = self.motion_length[index]
        # jt_A = self.jt_A[self.start_id[index]:self.start_id[index]+motion_length]
        # jt_B = self.jt_B[self.start_id[index]:self.start_id[index]+motion_length]
        # root_A = self.root_A[self.start_id[index]:self.start_id[index]+motion_length]
        # root_B = self.root_B[self.start_id[index]:self.start_id[index]+motion_length]
        # jt_root_A = torch.concat([jt_A, root_A], dim=1)
        # jt_root_B = torch.concat([jt_B, root_B], dim=1)
        jt_root_A = self.jt_root_A[self.start_id[index]:self.start_id[index]+motion_length]
        jt_root_B = self.jt_root_B[self.start_id[index]:self.start_id[index]+motion_length]
        if self.load_human:
            jt_root_C = self.jt_root_C[self.start_id[index]:self.start_id[index]+motion_length]
        if self.load_pose:
            # random choose a pose 
            pose_id = torch.randint(motion_length, (1,))
            A = jt_root_A[pose_id]
            B = jt_root_B[pose_id]
            if self.load_human:
                C = jt_root_C[pose_id]
        else:
            pose_id = torch.randint(motion_length - self.window_size+1, (1,))
            if self.overlap:
                breakpoint()
                pose_id = pose_id//(self.window_size-self.overlap) * (self.window_size-self.overlap)
            # zero_pad_A = torch.zeros((self.max_length-motion_length, jt_root_A.shape[-1])).to(jt_root_A)
            # zero_pad_B = torch.zeros((self.max_length-motion_length, jt_root_B.shape[-1])).to(jt_root_A)
            # A = torch.concat([jt_root_A, zero_pad_A], dim=0)
            # B = torch.concat([jt_root_B, zero_pad_B], dim=0)
            A = jt_root_A[pose_id:pose_id+self.window_size]
            B = jt_root_B[pose_id:pose_id+self.window_size]
            if self.load_human:
                # zero_pad_C = torch.zeros((self.max_length-motion_length, jt_root_C.shape[-1])).to(jt_root_A)
                # C = torch.concat([jt_root_C, zero_pad_C], dim=0)
                C = jt_root_C[pose_id:pose_id+self.window_size]
        if self.load_human:
            return B, A, C
        return B, A, index#self.names[index]

        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length
