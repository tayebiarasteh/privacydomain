"""
Created on Jan 20, 2023.
data_provider_privdom.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15




class UKA_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA/chest_radiograph')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "labels/original_novalid_UKA_master_list.csv"), sep=',') # 150,188 train / 39,021 test images
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "labels/temporiginal_novalid_UKA_master_list.csv"), sep=',') # 150,188 train / 39,021 test images

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.mode = mode

        # ageinterval = [0, 40]
        # ageinterval = [40, 70]
        # ageinterval = [70, 100]
        # self.subset_df = self.subset_df[self.subset_df['age'] >= ageinterval[0]]
        # self.subset_df = self.subset_df[self.subset_df['age'] < ageinterval[1]]

        # self.subset_df = self.subset_df[self.subset_df['gender'] > 0] # female
        # self.subset_df = self.subset_df[self.subset_df['gender'] < 1] # male

        # print(len(self.subset_df))


        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed')

        self.file_path_list = list(self.subset_df['image_id'])
        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'healthy'] # 4 labels
        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'healthy'] # 5 labels



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]['subset'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, subset, str(self.file_path_list[idx]) + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]

        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            if self.chosen_labels[idx] == 'cardiomegaly':
                if int(label_df[self.chosen_labels[idx]].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) == 4:
                    label[idx] = 1
                elif int(label_df[self.chosen_labels[idx]].values[0]) > 4:
                    label[idx] = 0
                elif int(label_df[self.chosen_labels[idx]].values[0]) < 3:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'pleural_effusion':
                if int(label_df['pleural_effusion_right'].values[0]) == 3 or int(label_df['pleural_effusion_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['pleural_effusion_right'].values[0]) == 4 or int(label_df['pleural_effusion_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'atelectasis':
                if int(label_df['atelectasis_right'].values[0]) == 3 or int(label_df['atelectasis_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['atelectasis_right'].values[0]) == 4 or int(label_df['atelectasis_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'pneumonia':
                if int(label_df['pneumonic_infiltrates_right'].values[0]) == 3 or int(label_df['pneumonic_infiltrates_left'].values[0]) == 3:
                    label[idx] = 1
                elif int(label_df['pneumonic_infiltrates_right'].values[0]) == 4 or int(label_df['pneumonic_infiltrates_left'].values[0]) == 4:
                    label[idx] = 1
                else:
                    label[idx] = 0

            elif self.chosen_labels[idx] == 'healthy':
                if int(label_df['healthy'].values[0]) == 1:
                    label[idx] = 1
                else:
                    label[idx] = 0


        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            if diseases == 'pleural_effusion':
                disease_length = sum(train_df['pleural_effusion_right'].values == 3)
                disease_length += sum(train_df['pleural_effusion_left'].values == 3)
                disease_length += sum(train_df['pleural_effusion_right'].values == 4)
                disease_length += sum(train_df['pleural_effusion_left'].values == 4)
            elif diseases == 'atelectasis':
                disease_length = sum(train_df['atelectasis_right'].values == 3)
                disease_length += sum(train_df['atelectasis_left'].values == 3)
                disease_length += sum(train_df['atelectasis_right'].values == 4)
                disease_length += sum(train_df['atelectasis_left'].values == 4)
            elif diseases == 'pneumonia':
                disease_length = sum(train_df['pneumonic_infiltrates_right'].values == 3)
                disease_length += sum(train_df['pneumonic_infiltrates_left'].values == 3)
                disease_length += sum(train_df['pneumonic_infiltrates_right'].values == 4)
                disease_length += sum(train_df['pneumonic_infiltrates_left'].values == 4)
            elif diseases == 'healthy':
                disease_length = sum(train_df['healthy'].values == 1)
            else:
                disease_length = sum(train_df[diseases].values == 3)
                disease_length += sum(train_df[diseases].values == 4)

            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class padchest_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'padchest')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "padchest_master_list_20percenttest.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        APhorizview = self.subset_df[self.subset_df['view'] == 'AP_horizontal']
        self.subset_df = PAview.append(APview)
        self.subset_df = self.subset_df.append(APhorizview)


        # ageinterval = [0, 40]
        # ageinterval = [40, 70]
        # ageinterval = [70, 100]
        # self.subset_df = self.subset_df[self.subset_df['age'] >= ageinterval[0]]
        # self.subset_df = self.subset_df[self.subset_df['age'] < ageinterval[1]]

        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'F'] # female
        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'M'] # male


        self.file_path_list = list(self.subset_df['ImageID'])

        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'no_finding']
        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']


    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        subset = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]['ImageDir'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, str(subset), self.file_path_list[idx])) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['ImageID'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class mimic_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size256=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.size256 = size256
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, "MIMIC")
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "nothree_master_list_20percenttest.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        self.subset_df = PAview.append(APview)
        self.file_path_list = list(self.subset_df['jpg_rel_path'])

        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'no_finding']
        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        if self.size256:
            img_path = img_path.replace("/files/", "/preprocessed256/")
        else:
            img_path = img_path.replace("/files/", "/preprocessed/")

        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class chexpert_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.size224 = size224
        self.file_base_dir = self.params['file_path']
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "nothree_master_list_20percenttest.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.subset_df = self.subset_df[self.subset_df['view'] == 'Frontal']


        # ageinterval = [0, 40]
        # ageinterval = [40, 70]
        # ageinterval = [70, 100]
        # self.subset_df = self.subset_df[self.subset_df['age'] >= ageinterval[0]]
        # self.subset_df = self.subset_df[self.subset_df['age'] < ageinterval[1]]

        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'Female'] # female
        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'Male'] # male


        self.file_path_list = list(self.subset_df['jpg_rel_path'])

        # self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'no_finding']
        self.chosen_labels = ['cardiomegaly', 'pleural_effusion', 'pneumonia', 'atelectasis', 'no_finding']





    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        if self.size224:
            img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed224/")
        else:
            img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class cxr14_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'NIH_ChestX-ray14')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_cxr14_master_list.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']


        # ageinterval = [0, 40]
        # ageinterval = [40, 70]
        # ageinterval = [70, 100]
        # self.subset_df = self.subset_df[self.subset_df['age'] >= ageinterval[0]]
        # self.subset_df = self.subset_df[self.subset_df['age'] < ageinterval[1]]

        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'F'] # female
        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'M'] # male


        self.file_path_list = list(self.subset_df['img_rel_path'])

        # self.chosen_labels = ['cardiomegaly', 'effusion', 'pneumonia', 'no_finding']
        self.chosen_labels = ['cardiomegaly', 'effusion', 'pneumonia', 'atelectasis', 'no_finding']





    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx])) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['img_rel_path'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class vindr_data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False, size224=False):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-cxr1')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_master_list.csv"), sep=',')

        if size224:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed224')
        else:
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']
            self.file_base_dir = os.path.join(self.file_base_dir, 'test')


        # ageinterval = [0, 40]
        # ageinterval = [40, 70]
        # ageinterval = [70, 100]
        # self.subset_df = self.subset_df[self.subset_df['age'] >= ageinterval[0]]
        # self.subset_df = self.subset_df[self.subset_df['age'] < ageinterval[1]]

        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'F'] # female
        # self.subset_df = self.subset_df[self.subset_df['gender'] == 'M'] # male


        self.file_path_list = list(self.subset_df['image_id'])

        # self.chosen_labels = ['Cardiomegaly', 'Pleural effusion', 'Pneumonia', 'No finding']
        self.chosen_labels = ['Cardiomegaly', 'Pleural effusion', 'Pneumonia', 'Atelectasis', 'No finding']




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx] + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.4),
                                        transforms.RandomRotation(degrees=8), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]
        label = torch.zeros((len(self.chosen_labels)))  # (h,)

        for idx in range(len(self.chosen_labels)):
            label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
        label = label.float()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor
