# Adversarial Example in CNNs using Reproducing FGSM

## Honor Code
We pledge on our honor that we have not given or received  any unauthorized assistance in this project.


## Project Overview

This project investigates how small perturbations to input images can
mislead a Convolutional Neural Network (CNN).

It uses the MNIST handwritten-digit dataset and the Fast Gradient Sign
Method (FGSM) to generate adversarial examples. It also explores whether
adversarial training can improve the model's robustness.


## Objectives

- Train a CNN to classify handwritten digits from 0 to 9.
- Generate adversarial images using FGSM.
- Evaluate the effect of different perturbation strengths.
- Train a model using clean and adversarial images.
- Compare the robustness of both models.
- Visualize original and adversarial examples.


## How FGSM Works

The Fast Gradient Sign Method (FGSM) is an adversarial attack that
uses the gradient of the model's loss with respect to the input image.

The adversarial image is calculated as:

$$
x_{adv} = \operatorname{clip}_{[0,1]}
\left(x + \epsilon \operatorname{sign}
\left(\nabla_x J(\theta,x,y)\right)\right)
$$

Here:

- `x` is the original image.
- `epsilon` controls the perturbation strength.
- `J` is the classification loss.
- The input gradient indicates how pixel changes affect the loss.
- Clipping keeps pixel values between 0 and 1.



## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Jupyter Notebook
- MNIST dataset
