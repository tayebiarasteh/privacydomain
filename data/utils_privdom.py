"""
Created on Jan 31, 2023.
utils_privdom.py
heatmap figures generator

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import seaborn as sns
import pdb
import matplotlib.pylab as plt
import pandas as pd




class main_manuscript():
    def __init__(self):
        pass

    def all_figs_epsbelowone(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1)
        plt.suptitle('AUC values for ε < 1', fontsize=20)

        ########## average #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.72,	0.80],
                             [0.88,	0.78,	0.76,	0.69,	0.83],
                             [0.85,	0.75,	0.82,	0.68,	0.83],
                             [0.86,	0.67,	0.72,	0.88,	0.81],
                             [0.89,	0.74,	0.75,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(231)
        plt.title('(A) Average', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## average #########

        ########## cardiomegaly #########
        auc_list = np.array([[0.96, 0.71, 0.73, 0.72, 0.80],
                             [0.90, 0.88, 0.77, 0.74, 0.86],
                             [0.90, 0.84, 0.87, 0.75, 0.88],
                             [0.94, 0.74, 0.78, 0.85, 0.84],
                             [0.93, 0.80, 0.77, 0.75, 0.93]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(232)
        plt.title('(B) Cardiomegaly', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## cardiomegaly #########

        ########## effusion #########
        auc_list = np.array([[0.98,	0.77,	0.84,	0.76,	0.93],
                             [0.95,	0.83,	0.86,	0.73,	0.95],
                             [0.96,	0.81,	0.88,	0.79,	0.95],
                             [0.90,	0.74,	0.83,	0.91,	0.93],
                             [0.94,	0.80,	0.85,	0.79,	0.96]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(233)
        plt.title('(C) Pleural Effusion', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## effusion #########

        ########## pneumonia #########
        auc_list = np.array([[0.92,	0.66,	0.59,	0.77,	0.76],
                             [0.88,	0.71,	0.66,	0.78,	0.78],
                             [0.79,	0.68,	0.76,	0.79,	0.76],
                             [0.84,	0.68,	0.63,	0.92,	0.79],
                             [0.88,	0.67,	0.63,	0.77,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(234)
        plt.title('(D) Pneumonia', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## pneumonia #########

        ########## atelectasis #########
        auc_list = np.array([[0.90,	0.64,	0.60,	0.62,	0.74],
                             [0.84,	0.76,	0.65,	0.49,	0.78],
                             [0.74,	0.71,	0.70,	0.36,	0.75],
                             [0.82,	0.54,	0.56,	0.86,	0.72],
                             [0.82,	0.72,	0.65,	0.55,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(235)
        plt.title('(E) Atelectasis', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## atelectasis #########

        ########## Healthy #########
        auc_list = np.array([[0.91,	0.69,	0.84,	0.74,	0.78],
                             [0.83,	0.72,	0.85,	0.72,	0.78],
                             [0.85,	0.71,	0.88,	0.72,	0.80],
                             [0.79,	0.66,	0.82,	0.86,	0.76],
                             [0.87,	0.70,	0.85,	0.70,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(236)
        plt.title('(F) Healthy', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## Healthy #########

        plt.show()


    def eps_differences(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1.1)
        plt.suptitle('Average AUC values for different ε values', fontsize=20)

        ########## 0 < ε < 1 #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.72,	0.80],
                             [0.88,	0.78,	0.76,	0.69,	0.83],
                             [0.85,	0.75,	0.82,	0.68,	0.83],
                             [0.86,	0.67,	0.72,	0.88,	0.81],
                             [0.89,	0.74,	0.75,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(221)
        plt.title('(A) 0 < ε < 1', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## ε < 1 #########

        ########## 2 < ε < 4 #########
        auc_list = np.array([[0.94,	0.69,	0.71,	0.73,	0.80],
                             [0.87,	0.78,	0.76,	0.68,	0.83],
                             [0.85,	0.75,	0.82,	0.68,	0.83],
                             [0.86,	0.67,	0.72,	0.88,	0.81],
                             [0.89,	0.74,	0.74,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(222)
        plt.title('(B) 2 < ε < 3.5', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## 2 < ε < 4 #########

        ########## 4 < ε < 10 #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.70,	0.80],
                             [0.88,	0.78,	0.76,	0.68,	0.83],
                             [0.84,	0.75,	0.82,	0.68,	0.83],
                             [0.85,	0.67,	0.72,	0.88,	0.80],
                             [0.89,	0.74,	0.74,	0.70,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(223)
        plt.title('(C) 3.5 < ε < 10', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## 4 < ε < 10 #########

        ########## ε = ∞ #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.71,	0.81],
                             [0.86,	0.78,	0.78,	0.68,	0.84],
                             [0.86,	0.76,	0.82,	0.71,	0.84],
                             [0.85,	0.69,	0.72,	0.88,	0.80],
                             [0.90,	0.76,	0.77,	0.70,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(224)
        plt.title('(D) ε = ∞', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## ε = ∞ #########

        plt.show()


    def genderplots_belowone(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1.1)
        plt.suptitle('Average AUC values for different genders using DP-DT (ε < 1) and non-DP-DT (ε = ∞)', fontsize=20)

        ########## Female (ε < 1) #########
        auc_list = np.array([[0.93,	0.68,	0.72,	0.69,	0.81],
                             [0.81,	0.77,	0.78,	0.69,	0.84],
                             [0.80,	0.75,	0.82,	0.71,	0.83],
                             [0.84,	0.68,	0.73,	0.88,	0.79],
                             [0.88,	0.75,	0.77,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(221)
        plt.title('(A) Female (ε < 1)', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## Female (ε < 1) #########

        ########## Male (ε < 1) #########
        auc_list = np.array([[0.92,	0.69,	0.72,	0.71,	0.81],
                             [0.80,	0.79,	0.77,	0.68,	0.84],
                             [0.84,	0.77,	0.82,	0.70,	0.84],
                             [0.84,	0.69,	0.72,	0.88,	0.81],
                             [0.87,	0.77,	0.76,	0.69,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(222)
        plt.title('(B) Male (ε < 1)', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Blues', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## Male (ε < 1) #########

        ########## Female (ε = ∞) #########
        auc_list = np.array([[0.93,	0.68,	0.73,	0.71,	0.80],
                             [0.83,	0.77,	0.76,	0.69,	0.82],
                             [0.80,	0.74,	0.82,	0.68,	0.83],
                             [0.83,	0.66,	0.73,	0.88,	0.81],
                             [0.86,	0.73,	0.76,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(223)
        plt.title('(C) Female (ε = ∞)', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## Female (ε = ∞) #########

        ##########  male (ε = ∞) #########
        auc_list = np.array([[0.92,	0.70,	0.71,	0.72,	0.80],
                             [0.85,	0.79,	0.75,	0.69,	0.83],
                             [0.80,	0.76,	0.82,	0.68,	0.83],
                             [0.81,	0.68,	0.72,	0.88,	0.80],
                             [0.83,	0.74,	0.75,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(224)
        plt.title('(D) Male (ε = ∞)', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Blues', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## male (ε = ∞) #########

        plt.show()


    def age_plots_belowone(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1)
        plt.suptitle('Average AUC values for different age groups using DP-DT (ε < 1) and non-DP-DT (ε = ∞)', fontsize=20)

        ########## [0, 40) years (ε < 1) #########
        auc_list = np.array([[0.94,	0.71,	0.75,	0.73,	0.81],
                            [0.89,	0.79,	0.78,	0.72,	0.81],
                            [0.85,	0.76,	0.84,	0.71,	0.83],
                            [0.87,	0.68,	0.75,	0.90,	0.81],
                            [0.89,	0.75,	0.77,	0.75,	0.87]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(231)
        plt.title('(A) [0, 40) years (ε < 1)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## [0, 40) years (ε < 1) #########

        ########## [40, 70) years (ε < 1) #########
        auc_list = np.array([[0.91,	0.69,	0.72,	0.73,	0.80],
                            [0.84,	0.77,	0.75,	0.70,	0.82],
                            [0.80,	0.75,	0.81,	0.69,	0.83],
                            [0.84,	0.67,	0.72,	0.89,	0.80],
                            [0.83,	0.74,	0.75,	0.72,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(232)
        plt.title('(B) [40, 70) years (ε < 1)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Blues', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## [40, 70) years (ε < 1) #########

        ########## [70, 100) years (ε < 1) #########
        auc_list = np.array([[0.86,	0.66,	0.69,	0.70,	0.76],
                            [0.78,	0.74,	0.72,	0.68,	0.80],
                            [0.79,	0.73,	0.79,	0.67,	0.79],
                            [0.77,	0.63,	0.69,	0.87,	0.76],
                            [0.78,	0.71,	0.72,	0.69,	0.85]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(233)
        plt.title('(C) [70, 100) years (ε < 1)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Reds', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## [70, 100) years (ε < 1) #########



        ########## [0, 40) years (ε = ∞) #########
        auc_list = np.array([[0.94,	0.70,	0.74,	0.73,	0.80],
                            [0.86,	0.79,	0.80,	0.72,	0.81],
                            [0.87,	0.77,	0.84,	0.75,	0.82],
                            [0.86,	0.70,	0.74,	0.90,	0.80],
                            [0.91,	0.77,	0.80,	0.74,	0.87]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(234)
        plt.title('(D) [0, 40) years (ε = ∞)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## [0, 40) years (ε = ∞) #########

        ########## [40, 70) years (ε = ∞) #########
        auc_list = np.array([[0.91,	0.69,	0.71,	0.72,	0.81],
                            [0.83,	0.77,	0.77,	0.69,	0.84],
                            [0.84,	0.76,	0.81,	0.71,	0.83],
                            [0.80,	0.68,	0.72,	0.89,	0.79],
                            [0.86,	0.76,	0.77,	0.70,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(235)
        plt.title('(E) [40, 70) years (ε = ∞)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Blues', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## [40, 70) years (ε = ∞) #########

        ########## [70, 100) years (ε = ∞) #########
        auc_list = np.array([[0.86,	0.64,	0.68,	0.68,	0.77],
                            [0.85,	0.74,	0.75,	0.67,	0.80],
                            [0.84,	0.74,	0.79,	0.69,	0.80],
                            [0.77,	0.65,	0.70,	0.87,	0.77],
                            [0.83,	0.73,	0.73,	0.68,	0.85]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(236)
        plt.title('(F) [70, 100) years (ε = ∞)', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Reds', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## [70, 100) years (ε = ∞) #########

        plt.show()


    def ondomain_DP_AUC_plot(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        plt.suptitle('AUC over ε in on-domain setting', fontsize=20)
        plt.rcParams['font.size'] = '14'

        ########## vindr #########
        x = [0.29, 0.44, 0.58, 0.99, 2.01, 2.72, 3.13, 3.71, 13.43]  # vindr
        y = [0.8800, 0.8818, 0.8846, 0.8936, 0.8977, 0.9007, 0.9015, 0.9044, 0.9079]
        plt.subplot(231)
        plt.title('(A) VDR', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        # plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        plt.ylim([0.8780, 0.909])
        plt.xlim([0, 10])
        ########## VinDr #########

        ########## cxr14 #########
        x = [0.57, 0.61, 0.77, 0.98, 1.50, 3.27, 3.85, 6.62, 7.62]  # C14
        y = [0.728, 0.7315, 0.7393, 0.7435, 0.7487, 0.7505, 0.7512, 0.7526, 0.7537]
        plt.subplot(232)
        plt.title('(B) C14', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        # plt.xlabel('ε', fontsize=16)
        # plt.ylabel('AUROC', fontsize=16)
        plt.ylim([0.724, 0.756])
        plt.xlim([0, 10])
        ########## cxr14 #########

        ########## chexpert #########
        x = [0.36, 0.41, 0.51, 0.58, 0.68, 0.90, 0.97, 1.10, 3.29, 6.47] # AUC
        y = [0.779, 0.7819, 0.7857, 0.788, 0.7905, 0.7927, 0.7931, 0.7935, 0.7949, 0.7958]

        plt.subplot(233)
        plt.title('(C) CPT', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.ylim([0.777, 0.797])
        # plt.xlabel('ε', fontsize=16)
        plt.xlim([0, 10])
        # plt.ylabel('AUROC', fontsize=16)
        ########## chexpert #########

        ########## UKA #########
        x = [0.70, 0.83, 0.98, 1.20, 3.45, 3.82, 5.64, 6.14, 7.49] # UKA
        y = [0.8072, 0.83, 0.8398, 0.846, 0.8478, 0.8503, 0.8523, 0.8539, 0.8574]
        plt.subplot(234)
        plt.title('(D) UKA', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.ylim([0.80, 0.86])
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        plt.xlim([0, 10])
        ########## UKA #########


        ########## padchest #########
        x = [0.54, 0.60, 0.69, 1.04, 2.03, 2.51, 3.29, 3.49, 5.22, 5.69, 7.20, 8.69] # padchest
        y = [0.8385, 0.8436, 0.8444, 0.8449, 0.8481, 0.8514, 0.855, 0.8555, 0.8566, 0.8576, 0.8598, 0.8602]
        plt.subplot(235)
        plt.title('(E) PCH', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.xlabel('ε', fontsize=16)
        # plt.ylabel('AUROC', fontsize=16)
        plt.xlim([0, 10])
        ########## padchest #########

        ########## all #########
        x1 = [0.29, 0.44, 0.58, 0.99, 2.01, 2.72, 3.13, 3.71, 13.43]  # vindr
        y1 = [0.8800, 0.8818, 0.8846, 0.8936, 0.8977, 0.9007, 0.9015, 0.9044, 0.9079]
        x2 = [0.57, 0.61, 0.77, 0.98, 1.50, 3.27, 3.85, 6.62, 7.62]  # C14
        y2 = [0.728, 0.7315, 0.7393, 0.7435, 0.7487, 0.7505, 0.7512, 0.7526, 0.7537]
        x3 = [0.36, 0.41, 0.51, 0.58, 0.68, 0.90, 0.97, 1.10, 3.29, 6.47] # AUC
        y3 = [0.779, 0.7819, 0.7857, 0.788, 0.7905, 0.7927, 0.7931, 0.7935, 0.7949, 0.7958]
        x4 = [0.70, 0.83, 0.98, 1.20, 3.45, 3.82, 5.64, 6.14, 7.49] # UKA
        y4 = [0.8072, 0.83, 0.8398, 0.846, 0.8478, 0.8503, 0.8523, 0.8539, 0.8574]
        x5 = [0.54, 0.60, 0.69, 1.04, 2.03, 2.51, 3.29, 3.49, 5.22, 5.69, 7.20, 8.69] # padchest
        y5 = [0.8385, 0.8436, 0.8444, 0.8449, 0.8481, 0.8514, 0.855, 0.8555, 0.8566, 0.8576, 0.8598, 0.8602]
        plt.subplot(236)
        plt.title('(F) All', fontsize=16)
        plt.plot(x1, y1, linewidth=2, label="VDR", color='blue', marker='o')
        plt.plot(x2, y2, linewidth=2, label="C14", color='red', marker='s')
        plt.plot(x3, y3, linewidth=2, label="CPT", color='green', marker='P')
        plt.plot(x4, y4, linewidth=2, label="UKA", color='orange', marker='*')
        plt.plot(x5, y5, linewidth=2, label="PCH", color='brown', marker='p')
        plt.xlabel('ε', fontsize=16)
        # plt.ylabel('AUROC', fontsize=16)
        plt.legend()
        plt.xlim([0, 10])
        # plt.ylim([0.72, 0.91])
        ########## all #########

        plt.show()


class supplements():
    def __init__(self):
        pass

    def all_figs_epsaboveone(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1.1)
        plt.suptitle('AUC values for 2 < ε < 3.5', fontsize=20)

        ########## average #########
        auc_list = np.array([[0.94,	0.69,	0.71,	0.73,	0.80],
                            [0.87,	0.78,	0.76,	0.68,	0.83],
                            [0.85,	0.75,	0.82,	0.68,	0.83],
                            [0.86,	0.67,	0.72,	0.88,	0.81],
                            [0.89,	0.74,	0.74,	0.71,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(231)
        plt.title('(A) Average', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## average #########

        ########## cardiomegaly #########
        auc_list = np.array([[0.96,	0.70,	0.72,	0.73,	0.80],
                            [0.89,	0.88,	0.78,	0.74,	0.88],
                            [0.91,	0.85,	0.87,	0.74,	0.88],
                            [0.94,	0.75,	0.79,	0.85,	0.84],
                            [0.94,	0.82,	0.75,	0.74,	0.93]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(232)
        plt.title('(B) Cardiomegaly', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## cardiomegaly #########

        ########## effusion #########
        auc_list = np.array([[0.98,	0.76,	0.83,	0.77,	0.93],
                            [0.95,	0.83,	0.86,	0.74,	0.94],
                            [0.96,	0.81,	0.88,	0.80,	0.95],
                            [0.91,	0.74,	0.82,	0.91,	0.93],
                            [0.94,	0.80,	0.85,	0.79,	0.96]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(233)
        plt.title('(C) Pleural Effusion', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## effusion #########

        ########## pneumonia #########
        auc_list = np.array([[0.92,	0.65,	0.57,	0.78,	0.75],
                            [0.86,	0.71,	0.65,	0.76,	0.76],
                            [0.79,	0.69,	0.76,	0.79,	0.76],
                            [0.85,	0.68,	0.63,	0.92,	0.79],
                            [0.87,	0.66,	0.63,	0.76,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(234)
        plt.title('(D) Pneumonia', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## pneumonia #########

        ########## atelectasis #########
        auc_list = np.array([[0.90,	0.64,	0.60,	0.64,	0.73],
                            [0.84,	0.76,	0.65,	0.47,	0.78],
                            [0.72,	0.71,	0.70,	0.32,	0.75],
                            [0.81,	0.54,	0.56,	0.86,	0.71],
                            [0.84,	0.72,	0.64,	0.55,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(235)
        plt.title('(E) Atelectasis', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## atelectasis #########

        ########## Healthy #########
        auc_list = np.array([[0.91,	0.69,	0.84,	0.74,	0.79],
                            [0.82,	0.72,	0.85,	0.71,	0.77],
                            [0.86,	0.71,	0.88,	0.74,	0.80],
                            [0.79,	0.66,	0.82,	0.86,	0.76],
                            [0.86,	0.70,	0.85,	0.69,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(236)
        plt.title('(F) Healthy', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## Healthy #########

        plt.show()


    def all_figs_epsten(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1.1)
        plt.suptitle('AUC values for 3.5 < ε < 10', fontsize=20)

        ########## average #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.70,	0.80],
                            [0.88,	0.78,	0.76,	0.68,	0.83],
                            [0.84,	0.75,	0.82,	0.68,	0.83],
                            [0.85,	0.67,	0.72,	0.88,	0.80],
                            [0.89,	0.74,	0.74,	0.70,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(231)
        plt.title('(A) Average', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## average #########

        ########## cardiomegaly #########
        auc_list = np.array([[0.96,	0.70,	0.72,	0.72,	0.81],
                            [0.90,	0.88,	0.80,	0.74,	0.89],
                            [0.89,	0.85,	0.87,	0.75,	0.88],
                            [0.94,	0.77,	0.81,	0.85,	0.85],
                            [0.94,	0.84,	0.77,	0.74,	0.93]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(232)
        plt.title('(B) Cardiomegaly', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## cardiomegaly #########

        ########## effusion #########
        auc_list = np.array([[0.98,	0.77,	0.84,	0.73,	0.93],
                            [0.96,	0.83,	0.86,	0.74,	0.94],
                            [0.97,	0.81,	0.88,	0.80,	0.95],
                            [0.90,	0.73,	0.82,	0.91,	0.92],
                            [0.95,	0.80,	0.85,	0.78,	0.96]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(233)
        plt.title('(C) Pleural Effusion', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## effusion #########

        ########## pneumonia #########
        auc_list = np.array([[0.92,	0.65,	0.57,	0.76,	0.75],
                            [0.87,	0.71,	0.64,	0.75,	0.76],
                            [0.75,	0.69,	0.76,	0.79,	0.76],
                            [0.84,	0.68,	0.62,	0.92,	0.79],
                            [0.86,	0.65,	0.62,	0.76,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(234)
        plt.title('(D) Pneumonia', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## pneumonia #########

        ########## atelectasis #########
        auc_list = np.array([[0.90,	0.63,	0.60,	0.61,	0.73],
                            [0.85,	0.76,	0.66,	0.45,	0.78],
                            [0.71,	0.71,	0.70,	0.33,	0.74],
                            [0.81,	0.53,	0.55,	0.86,	0.71],
                            [0.86,	0.72,	0.63,	0.56,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(235)
        plt.title('(E) Atelectasis', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## atelectasis #########

        ########## Healthy #########
        auc_list = np.array([[0.91,	0.69,	0.84,	0.71,	0.79],
                            [0.84,	0.72,	0.85,	0.70,	0.77],
                            [0.86,	0.71,	0.88,	0.73,	0.80],
                            [0.77,	0.65,	0.81,	0.86,	0.76],
                            [0.87,	0.70,	0.85,	0.67,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(236)
        plt.title('(F) Healthy', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## Healthy #########

        plt.show()



    def all_figs_nondp(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        sns.set(font_scale=1.1)
        plt.suptitle('AUC values for non-DP-DT (ε = ∞)', fontsize=20)

        ########## average #########
        auc_list = np.array([[0.94,	0.69,	0.72,	0.71,	0.81],
                            [0.86,	0.78,	0.78,	0.68,	0.84],
                            [0.86,	0.76,	0.82,	0.71,	0.84],
                            [0.85,	0.69,	0.72,	0.88,	0.80],
                            [0.90,	0.76,	0.77,	0.70,	0.89]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(231)
        plt.title('(A) Average', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        ########## average #########

        ########## cardiomegaly #########
        auc_list = np.array([[0.96,	0.76,	0.76,	0.71,	0.86],
                            [0.87,	0.88,	0.80,	0.73,	0.90],
                            [0.86,	0.86,	0.87,	0.77,	0.87],
                            [0.96,	0.82,	0.84,	0.85,	0.87],
                            [0.92,	0.85,	0.80,	0.75,	0.93]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(232)
        plt.title('(B) Cardiomegaly', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## cardiomegaly #########

        ########## effusion #########
        auc_list = np.array([[0.98,	0.76,	0.83,	0.70,	0.92],
                            [0.96,	0.83,	0.87,	0.79,	0.95],
                            [0.97,	0.81,	0.88,	0.82,	0.95],
                            [0.89,	0.75,	0.82,	0.91,	0.93],
                            [0.96,	0.81,	0.86,	0.79,	0.96]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(233)
        plt.title('(C) Pleural Effusion', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## effusion #########

        ########## pneumonia #########
        auc_list = np.array([[0.92,	0.66,	0.59,	0.80,	0.76],
                            [0.79,	0.71,	0.69,	0.79,	0.76],
                            [0.80,	0.69,	0.76,	0.77,	0.75],
                            [0.83,	0.69,	0.62,	0.92,	0.78],
                            [0.84,	0.69,	0.67,	0.79,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(234)
        plt.title('(D) Pneumonia', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.ylabel('Training on', fontsize=14)
        plt.xlabel('Test on', fontsize=14)
        ########## pneumonia #########

        ########## atelectasis #########
        auc_list = np.array([[0.90,	0.57,	0.58,	0.64,	0.73],
                            [0.82,	0.76,	0.67,	0.41,	0.79],
                            [0.84,	0.74,	0.70,	0.43,	0.79],
                            [0.82,	0.52,	0.53,	0.86,	0.69],
                            [0.89,	0.74,	0.64,	0.47,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(235)
        plt.title('(E) Atelectasis', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## atelectasis #########

        ########## Healthy #########
        auc_list = np.array([[0.91,	0.68,	0.83,	0.68,	0.80],
                            [0.84,	0.72,	0.86,	0.69,	0.81],
                            [0.86,	0.71,	0.88,	0.74,	0.81],
                            [0.75,	0.66,	0.81,	0.86,	0.74],
                            [0.89,	0.72,	0.85,	0.69,	0.86]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(236)
        plt.title('(F) Healthy', fontsize=12)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        plt.xlabel('Test on', fontsize=14)
        ########## Healthy #########

        plt.show()



    def sample_sizes(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for different subsets of each test benchmark', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [3000, 25596, 29320, 39824, 22045]

        ax = plt.subplot(231)
        plt.title('(A) Full Set', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [552, 10714, 11436, 14457, 10636]

        ax = plt.subplot(232)
        plt.title('(B) Female', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [702, 14882, 17884, 25367, 11408]

        ax = plt.subplot(233)
        plt.title('(C) Male', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [149, 8410, 4391, 2297, 3612]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [250, 15495, 16420, 19197, 10788]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        height = [69, 1690, 8509, 18328, 7503]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years', fontsize=18)
        plt.ylim([0, 44000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()



    def sample_sizes_counter(self):
        path = "DATASET_CSV.csv"

        df = pd.read_csv(path, sep=',', low_memory=False)

        # df1 = df[df['view']== 'PA']
        # df2 = df[df['view']== 'AP']
        # df3 = df[df['view']== 'AP_horizontal']
        # df = df1.append(df2)
        # df = df.append(df3)

        # df = df[df['view']== 'Frontal']

        df = df[df['split'] == 'test']

        print('total:', len(df))

        df_f = df[df['gender'] == 'F']
        print('female:', len(df_f))

        df_m = df[df['gender'] == 'M']
        print('male:', len(df_m))

        df_1 = df[df['age'] > 0]
        df_1 = df_1[df_1['age'] < 40]
        print('[0 40]:', len(df_1))

        df_2 = df[df['age'] >= 40]
        df_2 = df_2[df_2['age'] < 70]
        print('[40 70]:', len(df_2))

        df_3 = df[df['age'] >= 70]
        df_3 = df_3[df_3['age'] < 100]
        print('[70 100]:', len(df_3))


    def sample_sizes_counter_individuallabels(self):
        path = "DATASET_CSV.csv"

        df = pd.read_csv(path, sep=',', low_memory=False)

        # df1 = df[df['view']== 'PA']
        # df2 = df[df['view']== 'AP']
        # df3 = df[df['view']== 'AP_horizontal']
        # df = df1.append(df2)
        # df = df.append(df3)

        # df = df[df['view']== 'Frontal']

        df = df[df['split'] == 'test']

        # disease = 'cardiomegaly'
        # disease = 'pleural_effusion'
        # disease = 'pneumonia'
        # disease = 'atelectasis'
        disease = 'no_finding'

        zero = len(df[df[disease] == 1])
        print('total:', zero)

        df_f = df[df['gender'] == 'Female']
        first = len(df_f[df_f[disease] == 1])
        print('female:', first)

        df_m = df[df['gender'] == 'Male']
        sec = len(df_m[df_m[disease] == 1])
        print('male:', sec)

        df_1 = df[df['age'] > 0]
        df_1 = df_1[df_1['age'] < 40]
        third = len(df_1[df_1[disease] == 1])
        print('[0 40]:', third)

        df_2 = df[df['age'] >= 40]
        df_2 = df_2[df_2['age'] < 70]
        fourth = len(df_2[df_2[disease] == 1])
        print('[40 70]:', fourth)

        df_3 = df[df['age'] >= 70]
        df_3 = df_3[df_3['age'] < 100]
        fifth = len(df_3[df_3[disease] == 1])
        print('[70 100]:', fifth)



    def individual_label_sample_sizes_VDR(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for individual positive labels for different subsets of VDR test set', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [309, 111, 246, 86, 2051]

        ax = plt.subplot(231)
        plt.title('(A) Full Set VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [124, 23, 37, 18, 308]

        ax = plt.subplot(232)
        plt.title('(B) Female VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [58, 43, 77, 29, 392]

        ax = plt.subplot(233)
        plt.title('(C) Male VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [6, 2, 8, 2, 125]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [37, 11, 26, 14, 115]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [32, 4, 9, 3, 4]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years VDR', fontsize=18)
        plt.ylim([0, 500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()



    def individual_label_sample_sizes_C14(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for individual positive labels for different subsets of C14 test set', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1069, 4658, 555, 3279, 9861]

        ax = plt.subplot(231)
        plt.title('(A) Full Set C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [528, 1921, 221, 1363, 4151]

        ax = plt.subplot(232)
        plt.title('(B) Female C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [541, 2737, 334, 1916, 5710]

        ax = plt.subplot(233)
        plt.title('(C) Male C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [415, 1336, 229, 801, 3488]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [575, 2884, 305, 2157, 5782]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [79, 438, 21, 321, 590]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years C14', fontsize=18)
        plt.ylim([0, 10000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()



    def individual_label_sample_sizes_CPT(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for individual positive labels for different subsets of CPT test set', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [3944, 11438, 816, 4522, 3540]

        ax = plt.subplot(231)
        plt.title('(A) Full Set CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1541, 4520, 307, 1670, 1422]

        ax = plt.subplot(232)
        plt.title('(B) Female CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [2403, 6918, 509, 2852, 2118]

        ax = plt.subplot(233)
        plt.title('(C) Male CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [460, 1188, 131, 488, 1018]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1955, 6288, 394, 2647, 2036]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1529, 3962, 291, 1387, 486]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years CPT', fontsize=18)
        plt.ylim([0, 12000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()




    def individual_label_sample_sizes_UKA(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for individual positive labels for different subsets of UKA test set', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [18616, 5049, 5844, 5575, 15273]

        ax = plt.subplot(231)
        plt.title('(A) Full Set UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [5748, 1828, 1846, 1931, 6408]

        ax = plt.subplot(232)
        plt.title('(B) Female UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [12868, 3221, 3998, 3644, 8865]

        ax = plt.subplot(233)
        plt.title('(C) Male UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [637, 228, 387, 332, 1151]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [8262, 2269, 3041, 2702, 7864]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [9713, 2552, 2415, 2539, 6248]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years UKA', fontsize=18)
        plt.ylim([0, 19000])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()




    def individual_label_sample_sizes_PCH(self):

        sns.set(font_scale=1.2)
        plt.suptitle('Total sample sizes for individual positive labels for different subsets of PCH test set', fontsize=20)
        plt.rcParams['font.size'] = 14

        ########## Full Set #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1954, 1373, 992, 1240, 7216]

        ax = plt.subplot(231)
        plt.title('(A) Full Set PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height, color='green')
        ax.bar_label(bars)
        ########## Full Set #########

        ########## Females #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1050, 504, 426, 529, 4083]

        ax = plt.subplot(232)
        plt.title('(B) Female PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## Females #########

        ########## males #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [904, 869, 566, 711, 3133]

        ax = plt.subplot(233)
        plt.title('(C) Male PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height, color='red')
        ax.bar_label(bars)
        ########## males #########

        ########## 0, 40 #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [29, 100, 292, 87, 2171]

        ax = plt.subplot(234)
        plt.title('(E) [0, 40) Years PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## 0, 40 #########

        ########## [40, 70] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [560, 491, 331, 546, 4081]

        ax = plt.subplot(235)
        plt.title('(E) [40, 70) Years PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [40, 70] #########

        ########## [70 100] #########
        x = ['CM', 'PE', 'PN', 'AT', 'HE']
        height = [1354, 780, 359, 601, 942]

        ax = plt.subplot(236)
        plt.title('(F) [70, 100) Years PCH', fontsize=18)
        plt.ylim([0, 7500])
        bars = plt.bar(x, height)
        ax.bar_label(bars)
        ########## [70 100] #########


        plt.show()





if __name__ == '__main__':
    manuscript = main_manuscript()
    supplement = supplements()
    # manuscript.all_figs_epsbelowone()
    # manuscript.eps_differences()
    # manuscript.genderplots_belowone()
    # manuscript.age_plots_belowone()
    # manuscript.ondomain_DP_AUC_plot()
    supplement.all_figs_epsaboveone()
    supplement.all_figs_epsten()
    supplement.all_figs_nondp()
    # supplement.sample_sizes()
    # supplement.individual_label_sample_sizes_VDR()
    # supplement.individual_label_sample_sizes_C14()
    # supplement.individual_label_sample_sizes_CPT()
    # supplement.individual_label_sample_sizes_UKA()
    # supplement.individual_label_sample_sizes_PCH()

