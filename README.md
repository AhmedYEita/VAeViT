# VAeViT
This repository contains the code implementation of VAeViT, our novel Deep Learning 3D multi-views object recognition model. This approach leverages the strengths of the Variational Autoencoder (VAE) and Vision Transformer (ViT) in capturing semantic features from sequential data for 3D multi-view object recognition. VAeViT learns distinct representations: VAE encodes 3D object views into a lower-dimensional latent space, while ViT captures deep feature representations across views for global object perception.

This implementation was evaluated on 4 different variants of 3D multi-view objects:
- ModelNet10 - 12 views (provided in this repo)
- ModelNet10 - 20 views (https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet10v2png_ori2.tar)
- ModelNet40 - 12 views (http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1png.tar)
- ModelNet40 - 20 views (https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar)

The preprocessing part in VAe_embedding.py assumes that the dataset directory is organized as follows:
path -> 10 / 40 category folders -> "train" and "test" -> images

How to run the code:
1. Install the required libraries (tensorflow, keras, tensorflow_addons, sklearn, numpy, os, opencv, matplotlib, seaborn, time, pandas, pickle)
2. Download the whole repo folder, unzip it

*VAe_embedding.py*

3. Insert the dataset directory on your local machine at "path = "      # default will assume that you are running your project from within the repo folder after downloading and unzipping it, it will also assume that you are using the ModelNet10 - 12 views dataset
4. Run the script for the desired number of epochs                      # 100 epochs is recommended and is the default

*ViT_perception.py*

5. If you not planning to train your own VAE model, you can use my trained models by adding "runs/" before both "z_train" and "y_train" loading model directories       # default will assume that you have trained and saved your own models
6. Make sure the correct models of the dataset variant you are implementing are chosen      # default will assume that you are implementing ModelNet10 - 12 views
7. Change the "num_views = " according to the number of views in the dataset variant you are using      # default is set to 12
8. Change the "num_classes = " according to the number of categories in the dataset variant you are using      # default is set to 10

*bar_plot.py*

9. Repeat the processes from 3 to 8 for the other 3 dataset variants and save the obtained results of each (VAE and VAeViT accuracies)
10. After you have trained the 4 variants and evaluated the accuracy of both VAE only and VAeViT, you can use bar_plot.py to plot a bar plot of all values and compare your results.
