#!/usr/bin/python

import numpy as np
import pandas as pd
from time import time
from sklearn import manifold


def main():
    #read in data
    print('Reading in data...')
    dat = pd.read_csv('../../work/NYTimesWorld/feature-maps/vgg16_feature_map_V1.csv', delimiter=',')
    ## Computing t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, perplexity = 50, learning_rate = 500, init='pca', random_state=0, verbose=1)
    print('Time recording starting now...')
    t0 = time()
    X_tsne = tsne.fit_transform(dat)
    print('X_tsne fit completed, exporting to data frame')
    #FINISH, FIGURE OUT
    np.savetxt("../../work/NYTimesWorld/feature-maps/t-sne-out-vgg16-feature-map_V1.csv", X_tsne, delimiter=',')

if __name__ == '__main__':
    main()

