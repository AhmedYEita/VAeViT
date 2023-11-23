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
1. Download the dataset and insert its directory on your local machine in VAe_embedding.py at "path="
2. 
