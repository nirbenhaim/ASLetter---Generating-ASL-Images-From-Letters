<h1 align="center">ASLetter – Generating ASL Images From Letters</h1>
<h2 align="center">Final project for the Technion's EE Deep Learning course (046211)
</h2> 

  <p align="center">
    Nir Ben Haim
  <br>
    David Levit
  </p>


## Background
 deep learning project - GAN which receives as input a letter in English and produces an image containing the letter as it is indicated in sign language (American Sign Language). 

### An illustration for the model: 
![Image of ViT-BYOL](./assets/our_vit_byol.png)

## Results
### original and generated images
![Image of predictions](./results/images_with_predictions_sigma=0.8.png)

## Files in the repository

| File name                                                     | Purpose                                                                                                                                       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `main.py`                                                     | general purpose main application                                                                                                              |
| `model.py`                                                    | generator amd discriminator models                                                                                                            |
| `train.py`                                                    | main application for training the model                                                                                                       |
| `util.py`                                                     | contain print results function and loss function                                                                                              |
| `data.py`                                                     | data loader                                                                                                                                   |
| `import_dataset.py`                                           | download data                                                                                                                                 |

## Running Example

| Arguments                                                     | Purpose                                                                                                                                       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `--eval`                                                      | Model evaluation if true. Model training and evaluating if false                                                                              |
| `--model_path`                                                | Model save/load directory                                                                                                                     |
| `--data_path`                                                 | Dataset save/load directorys                                                                                                                  |
| `--chosen_letter`                                             | Letter chosen to be printed, only when purely evaluating                                                                                      |

* In order to use the GAN without traning eval = True, model_path and data_path contains the wanted directory, and in order to get specific letter implement it in chosen_letter. 
* In order to retarin the GAN eval = False, model_path and data_path contains the wanted directory.

### Inference & Visual Results
* In order  
* run `python main.py`


## TensorBoard

TensorBoard logs are written dynamically during the runs, and it possible to observe the training progress using the graphs. In order to open TensoBoard, navigate to the source directory of the project and in the terminal/cmd:

`tensorboard --logdir=./runs`

* make sure you have the correct environment activated (`conda activate env-name`) and that you have `tensorboard`, `tensorboardX` installed.
## Sources & References
### Sources
* The BYOL code was adapted from the following pytorch implementation of [PyTorch-BYOL](https://github.com/sthalles/PyTorch-BYOL).
* The ViT (Not pretrained) model code was adapted from the following implementation of [vision transformers CIFAR10](https://github.com/kentaroy47/vision-transformers-cifar10) 
### Refrences
* [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733). 
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
