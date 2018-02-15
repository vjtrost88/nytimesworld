#!/usr/bin/env python

import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold


## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def main():
    #read in data
    dat = pd.read_csv('../../work/vgg16_feature_map.csv', delimiter=',')
    ## Computing t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, verbose=1)
    t0 = time()
    X_tsne = tsne.fit_transform(dat)
    plot_embedding(X_tsne, "t-SNE embedding of the NYT Images - VGG16 (time %.2fs)" % (time() - t0))
    plt.savefig('vgg16_embedding.png')

if __name__ == '__main__':
    main()

