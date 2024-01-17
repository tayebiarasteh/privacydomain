# Securing Collaborative Medical AI by Using Differential Privacy: Domain Transfer for Classification of Chest Radiographs



Overview
------

* This is the official repository of the paper [**Securing Collaborative Medical AI by Using Differential Privacy: Domain Transfer for Classification of Chest Radiographs**](https://pubs.rsna.org/doi/10.1148/ryai.230212).


Introduction
------
...

### Prerequisites

The software is developed in **Python 3.10**. For the deep learning, the **PyTorch 1.13** framework is used. The differential privacy was developed using **Opacus 1.3**.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate privacy_domainDP
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for federated learning as well as training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. Everything can be run from *./main_privdom.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./Train_Valid_privdom.py* contains the training and validation processes.
* *./Prediction_privdom.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper:

S. Tayebi Arasteh, M. Lotfinia, T. Nolte et al. *Securing Collaborative Medical AI by Using Differential Privacy: Domain Transfer for Classification of Chest Radiographs*. Radiology: Artificial Intelligence, https://doi.org/10.1148/ryai.230212, 2024, 6(1), e230212. RSNA

### BibTex

    @article {dpdo2024,
      author = {Tayebi Arasteh, Soroosh and Lotfinia, Mahshad and Nolte, Teresa and Saehn, Marwin and Isfort, Peter and Kuhl, Christiane and Nebelung, Sven and Kaissis, Georgios and Truhn, Daniel},
      title = {Securing Collaborative Medical AI by Using Differential Privacy: Domain Transfer for Classification of Chest Radiographs},
      year = {2024},
      pages = {e230212},
      doi = {10.1148/ryai.230212},
      publisher = {RSNA},
      URL = {https://pubs.rsna.org/doi/10.1148/ryai.230212},
      journal = {Radiology: Artificial Intelligence}
    }
