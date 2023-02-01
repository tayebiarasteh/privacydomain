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
        sns.set(font_scale=1.1)
        plt.suptitle('AUROC values for ε < 1', fontsize=16)

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
        plt.title('(A) Average', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
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
        plt.title('(B) Cardiomegaly', fontsize=16)
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
        plt.title('(C) Pleural Effusion', fontsize=16)
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
        plt.title('(D) Pneumonia', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
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
        plt.title('(E) Atelectasis', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
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
        plt.title('(F) Healthy', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0.5, vmax=1, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
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
        sns.set(font_scale=1.3)
        plt.suptitle('Differences of average AUROC values from the on-domain settings for different ε values', fontsize=20)

        ########## 0 < ε < 1 #########
        auc_list = np.array([[0.0,	0.09,	0.10,	0.16,	0.09],
                             [0.06,	0.0,	0.06,	0.19,	0.06],
                             [0.09,	0.03,	0.0,	0.20,	0.06],
                             [0.08,	0.11,	0.10,	0.0,	0.08],
                             [0.05,	0.04,	0.07,	0.17,	0.0]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(221)
        plt.title('(A) 0 < ε < 1', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0, vmax=0.25, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## ε < 1 #########

        ########## 2 < ε < 4 #########
        auc_list = np.array([[0.0,	0.08,	0.09,	0.15,	0.09],
                             [0.07,	0.0,	0.06,	0.20,	0.06],
                             [0.09,	0.03,	0.0,	0.20,	0.06],
                             [0.08,	0.11,	0.10,	0.0,	0.08],
                             [0.04,	0.04,	0.08,	0.17,	0.0]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(222)
        plt.title('(B) 2 < ε < 4', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0, vmax=0.25, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## 2 < ε < 4 #########

        ########## 4 < ε < 10 #########
        auc_list = np.array([[0.0,	0.09,	0.10,	0.18,	0.09],
                             [0.06,	0.0,	0.06,	0.20,	0.06],
                             [0.10,	0.03,	0.0,	0.20,	0.06],
                             [0.09,	0.11,	0.10,	0.0,	0.09],
                             [0.05,	0.04,	0.08,	0.18,	0.0]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(223)
        plt.title('(C) 4 < ε < 10', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0, vmax=0.25, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## 4 < ε < 10 #########

        ########## ε = ∞ #########
        auc_list = np.array([[0.0,	0.09,	0.10,	0.17,	0.08],
                             [0.08,	0.0,	0.04,	0.20,	0.05],
                             [0.08,	0.02,	0.0,	0.17,	0.05],
                             [0.09,	0.09,	0.10,	0.0,	0.09],
                             [0.04,	0.02,	0.05,	0.18,	0.0]])

        df = pd.DataFrame.from_records(auc_list)
        df.index = ['VDR', 'C14', 'CPT', 'UKA', 'PCH']
        df = df.rename(columns={0: 'VDR', 1: 'C14', 2: 'CPT', 3:'UKA', 4:'PCH'})
        plt.subplot(224)
        plt.title('(D) ε = ∞', fontsize=16)
        ax = sns.heatmap(data=df, vmin=0, vmax=0.25, annot=True, fmt=".2f", cmap='Greens', xticklabels=True, yticklabels=True)
        ax.tick_params(axis='y', left=False, labelleft=True)
        ax.tick_params(axis='x', bottom=False, labelbottom=True)
        ########## ε = ∞ #########

        plt.show()


    def ondomain_DP_AUC_plot(self):

        # order: vindr, cxr14, chexpert, UKA, padchest

        # order:

        # vindr,
        # cxr14,
        # chexpert,
        # UKA,
        # padchest
        plt.suptitle('AUROC values for ε < 1', fontsize=20)
        plt.rcParams['font.size'] = '14'

        ########## chexpert #########
        x = [0.90, 0.97, 1.10, 3.29, 6.47] # AUC
        y = [0.7927, 0.7931, 0.7935, 0.7949, 0.7958]

        plt.subplot(231)
        plt.title('(A) CPT', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        # plt.ylim([0.792, 0.796])
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        ########## chexpert #########

        ########## UKA #########
        x = [0.98, 1.20, 3.45, 8.80] # UKA
        y = [0.8398, 0.846, 0.8578, 0.8597]
        plt.subplot(232)
        plt.title('(B) UKA', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        ########## UKA #########

        ########## UKA #########
        x = [0.98, 1.20, 3.45, 8.80] # UKA
        y = [0.8398, 0.846, 0.8578, 0.8597]
        plt.subplot(233)
        plt.title('(B) UKA', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        ########## UKA #########

        ########## UKA #########
        x = [0.98, 1.20, 3.45, 8.80] # UKA
        y = [0.8398, 0.846, 0.8578, 0.8597]
        plt.subplot(234)
        plt.title('(B) UKA', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        ########## UKA #########

        ########## UKA #########
        x = [0.98, 1.20, 3.45, 8.80] # UKA
        y = [0.8398, 0.846, 0.8578, 0.8597]
        plt.subplot(235)
        plt.title('(B) UKA', fontsize=16)
        plt.plot(x, y, linewidth=2, markersize=12)
        plt.xlabel('ε', fontsize=16)
        plt.ylabel('AUROC', fontsize=16)
        ########## UKA #########

        plt.show()


class supplements():
    def __init__(self):
        pass








if __name__ == '__main__':
    manuscript = main_manuscript()
    # manuscript.all_figs_epsbelowone()
    manuscript.eps_differences()
    # manuscript.ondomain_DP_AUC_plot()

