# Autoencoder: Efficiency vs Tradeoffs

![Author](https://img.shields.io/badge/author-aaneloy-blue) 
![Author](https://img.shields.io/badge/turgeonmaxime-red)
[![MIT](https://img.shields.io/badge/license-MIT-5eba00.svg)](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs/blob/main/LICENSE.txt)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs)

## Update Logs:
The reproduction is still **ongoing**. Please refer the following logs for current and future updates:
1. Individual implimentation using official and partial references as mentioned in the paper is updated.
2. Single-Click result implimentation is ongoing.
3. Other autoencoder models will be added in the future.

## Overview
This repository introduces the paper **"A Comprehensive Study of Auto-Encoders for
Anomaly Detection: Efficiency and Trade-Offs"** by **[Asif Ahmed Neloy](https://aaneloy.github.io/)** and **[Dr. Max Turgeon](https://www.maxturgeon.ca/)**.


The following autonencoders are reproduced in this repository:
* Denoising Auto-Encoder **(DAE)**
* Sparse Auto-Encoder **(SAE)**
* Contractive Auto-Encoder **(CAE)**
* Variational Auto-Encoder **(VAE)**
* Conditional Variational Auto-Encoder **(CVAE)**
* **beta-VAE**
* Adversarial Variational Auto-Encoder **(adVAE)**
* Importance Weighted Auto-Encoder **(IWAE)**
* Probabilistic Auto-Encoder **(PAE)**
* Robust Deep Auto-Encoders **(RDA)**
* Vector Quantised-Variational Auto-Encoder **(VQ-VAE)**


## Reproducibility

The reproducibility process is divided into two process- 
1. Reproduce individual results using the [Official repos](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs/tree/main/Official%20Repos) or [Particial reference repos](https://github.com/UMDimReduction/autoencoder-Efficiency_vs_Tradeoffs/tree/main/Partial%20References) 
2. Reproduce all results using the modified models.


### Install Requirements

The ``requirements.txt`` file contains python packages for reproducing all models. **There is no individual requirements.txt file**. 

#### Prerequisite:
* Anaconda
* Python 3.7.11
* Windows 10/11 (*also can be reproduced using any linux machine. However, the package requirements needs to be adjusted*)
* GPUs (***PyTorch** and **TensorFlow** are created using gpu models*)

#### Installation steps:
* create new env: ``conda create -name autoencoder python=3.7 -y``
* activate env: ``conda activate autoencoder``
* open the directory and install the **``requirements.txt``**: ``pip install -r requirements.txt``


### Individual Results

#### 1. DAE
1. Navigate to the Directory: ``\Partial References\DAE\ ``
2. Run: `python main.py`

#### 2. SAE
1. Navigate to the Directory: ``\Partial References\SAE\ ``
2. Run: `python main.py`
3. Optional Result: `python sparseae_generic.py`

#### 3. CAE
1. Navigate to the Directory: ``\Partial References\CAE\ ``
2. Run: `python main.py`
3. Optional: Change parameters in Fashion-MNIST implementation ``python main_FashionMNIST.py``

#### 4. VAE
1. Navigate to the Directory: ``Official Repos\VAE ``
2. Run: `python run_main.py --dim_z 20`
3. Required Arguments `--dim_z`: Dimension of latent vector. *Default*: `20`

4. Optional:  
* `--results_path`: File path of output images. *Default*: `results`
* `--add_noise`: Boolean for adding salt & pepper noise to input image. *Default*: `False`
* `--n_hidden`: Number of hidden units in MLP. *Default*: `500`
* `--learn_rate`: Learning rate for Adam optimizer. *Default*: `1e-3`
* `--num_epochs`: The number of epochs to run. *Default*: `20`
* `--batch_size`: Batch size. *Default*: `128`
* `--PRR`: Boolean for plot-reproduce-result. *Default*: `True`
* `--PRR_n_img_x`: Number of images along x-axis. *Default*: `10`
* `--PRR_n_img_y`: Number of images along y-axis. *Default*: `10`
* `--PRR_resize_factor`: Resize factor for each displayed image. *Default*: `1.0`
* `--PMLR`: Boolean for plot-manifold-learning-result. *Default*: `False`
* `--PMLR_n_img_x`: Number of images along x-axis. *Default*: `20`
* `--PMLR_n_img_y`: Number of images along y-axis. *Default*: `20`
* `--PMLR_resize_factor`: Resize factor for each displayed image. *Default*: `1.0`
* `--PMLR_n_samples`: Number of samples in order to get distribution of labeled data. *Default*: `5000`

#### 5. CVAE
1. Navigate to the Directory: ``\Partial References\CVAE\``
2. Train and generate CVAE result: `` python main.py ``
3. Train and generate VAE result: `` python VAE.py ``
4. Testing Results: ``  python test_generation.py``

#### 6. beta-VAE
1. Navigate to the Directory: ``Official Repos\Beta_VAE``
2. Train: `` python train.py --z_dim 10 --beta 0.1 --dataset mnist --model conv_mnist --experiment_name z10_beta0.1_mnist_conv ``
3. Alternative: `` train.py --z_dim 10 --beta 0.1 --dataset mnist --model mlp_mnist --experiment_name z10_beta0.1_mnist_mlp ``

#### 7. adVAE

1. Go to the directory: ``cd Official Repos\adVAE``
2. Train model: ``python run.py``
3. optional parameters: 

    * `--datnorm`, type=bool, *default=`True`*, `help='Data normalization'`
    * `--z_dim`, type=int, *default=`128`*, `help='Dimension of latent vector'`
    * `--mx`, type=int, *default=`1`*, `help='Positive margin of MSE target.'`
    * `--mz`, type=int, *default=`1`*, `help='Positive margin of KLD target.'`
    * `--lr`, type=int, *default=`1e-4`*, `help='Learning rate for training'`
    * `--epoch`, type=int, *default=`100`*, `help='Training epoch'`
    * `--batch`, type=int, *default=`32`*, `help='Mini batch size'`


#### 8. IWAE
1. Navigate to the Directory: ``\Partial References\IWAE ``
2. For implementing the result mentioned in the reference paper: `python experiments.py --model [model] --dataset [dataset] --k [k] --layers [l]` *(where _model_ is vae or iwae; _dataset_ is one of BinFixMNIST, MNIST, OMNI; _k_ is 1, 5, or 50; _l_ is 1 or 2.)*
3. Optional implementation:` python experiments.py --exp iwae_to_vae`


4. Optional: `python download_mnist.py`: Download MNIST Dataset. *Default*: `mnist`

#### 9. PAE
1. Navigate to the Directory: `` cd \Official Repos\PAE``
2. Run the python file: ```python main.py --helpfull```
3. Optional Training: ```python main.py```


#### 10. RDA
1. Open the Directory: ``cd \Official Repos\RDA``
2. Train and regenerate: ``python main.py``
3. Compare to VAE: ``python VAE_SRDA.py``


#### 11. VQ-VAE
1. Navigate to the Directory: ``Official Repos\VQ_VAE ``
2. Run: `python main.py --MNIST`
3. Required Arguments `--dataset`: Dataset. *Default*: `CIFAR-10`

4. Optional:

* `--batch_size`, type=int, *default*=`32`*)
* `--n_updates`, type=int, *default*=`5000`)
* (`--n_hiddens`, type=int, *default*=`128`)
* (`--n_residual_hiddens`, type=int, *default*=`32`)
* (`--n_residual_layers`, type=int, *default*=`2`)
* (`--embedding_dim`, type=int, *default*=`64`)
* (`--n_embeddings`, type=int, *default*=`512`)
* (`--beta`, type=float, *default*=`.25`)
* (`--learning_rate`, type=float, *default*=`3e-4`)
* (`--log_interval`, type=int, *default*=`50`)


### All Results
*ALL RESULTS* is implemented to visualize the ROC-AUC, Latent Space Representation and Manifolds for *MNIST*, *FASHION_MNIST* datasets for all models regardless of the training parameters and hyper-parameters.
**(On-going)**



## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
