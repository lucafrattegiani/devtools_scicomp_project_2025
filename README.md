# Pressure regression with Universal Physics Transformer (UPT)
**Student:** Luca Frattegiani (lfratteg@sissa.it)

## General Description
UPT (https://arxiv.org/abs/2402.12365) is a transformer-based architecture able to perform neural operator learning in the context of a given spatio-temporal problem. In this project I tried to replicate the particular UPT structure used for steady state flows prediction, training and testing the model on the ShapeNet-Car dataset (https://visualcomputing.ist.ac.at/publications/2018/LearningFlow/).

## Dataset
Data are represented by a collection of 889 cars which surfaces consist of 3586 points in $\mathbb{R}^3$