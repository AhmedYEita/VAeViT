# VAeViT
This repository contains the code implementation of our novel Deep Learning 3D multi-views object recognition model, VAeViT, which consists of a Variational Autoencoder preceding a Vison Transformer.

This implementation was evaluated on 4 different variants of 3D multi-view objects:
- ModelNet10 - 12 views (provided in this repo)
- ModelNet10 - 20 views ([http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1png.tar](https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet10v2png_ori2.tar))
- ModelNet40 - 12 views ()
- ModelNet40 - 20 views ()

The preprocessing part in VAe_embedding.py assumes that the dataset directory is organized as follows:
path -> 10 / 40 category folders -> "train" and "test" -> images

Before you run the code, make sure to:
1. Install the required libraries (tensorflow, keras, tensorflow_addons, sklearn, numpy, os, opencv, matplotlib, seaborn, time, pandas, pickle)
2. Download the whole repo folder, unzip it
3. Insert the dataset directory on your local machine in VAe_embedding.py at "path = "      # default will assume that you are running your project from within the repo folder after downloading and extracting
4. If you not planning to train your own VAE model, you can use my trained models by adding "runs/" before both "z_train" and "y_train" loading model directories in ViT_perception.py       # default will assume that you are training your own model
5. Make sure the correct models of the dataset variant you are implementing are chosen      # default will assume that you are implementing ModelNet10 - 12 views
6. Change the "num_views = " in ViT_perception.py according to the number of views in the dataset variant you are using      # default is set to 12
7. Change the "num_classes = " in ViT_perception.py according to the number of categories in the dataset variant you are using      # default is set to 10

After you have trained the 4 variants and evaluated the accuracy of both VAE only and VAeViT, you can use bar_plot.py to plot a bar plot of all values and compare your results.
