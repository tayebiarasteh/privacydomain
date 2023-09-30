# Preserving privacy in domain transfer of medical AI models comes at no performance costs: The integral role of differential privacy



Overview
------

* This is the official repository of the paper [**Preserving privacy in domain transfer of medical AI models comes at no performance costs: The integral role of differential privacy**](https://arxiv.org/abs/2306.06503).
* Pre-print version: [https://arxiv.org/abs/2306.06503](https://arxiv.org/abs/2306.06503)


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

S. Tayebi Arasteh et al. *Preserving privacy in domain transfer of medical AI models comes at no performance costs: The integral role of differential privacy*. arxiv.2306.06503, https://doi.org/10.48550/arXiv.2306.06503, 2023.

### BibTex

    @article {dpdo2023,
      author = {Tayebi Arasteh, Soroosh and Lotfinia, Mahshad and Nolte, Teresa and Saehn, Marwin and Isfort, Peter and Kuhl, Christiane and Nebelung, Sven and Kaissis, Georgios and Truhn, Daniel},
      title = {Preserving privacy in domain transfer of medical AI models comes at no performance costs: The integral role of differential privacy},
      year = {2023},
      doi = {10.48550/arXiv.2306.06503},
      publisher = {arXiv},
      URL = {https://arxiv.org/abs/2306.06503},
      journal = {arXiv}
    }
