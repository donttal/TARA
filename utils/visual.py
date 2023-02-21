import os
import random
import urllib.request
import zipfile
from collections import OrderedDict
from random import sample

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn import datasets, manifold
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import manifold

random.seed(42)


class visual(object):
    def __init__(self, data_labels, data_preds, data_logits, labels) -> None:
        self.data_labels = data_labels
        self.data_preds = data_preds
        self.data_logits = data_logits
        confusion_mat = confusion_matrix(data_labels, data_preds)
        print("confusion_mat.shape : {}".format(confusion_mat.shape))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat, display_labels=labels)

        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(
            include_values=True,
            cmap="YlGnBu",
            ax=ax,
            xticks_rotation="vertical",
            values_format="d"
        )
        plt.show()

    def show_model_tSNE(self, storage_detail):
        '''t-SNE  with legend'''
        # https://stackoverflow.com/questions/52297857/t-sne-scatter-plot-with-legend
        index_sample = sample(range(1, len(self.data_logits)), 1500)
        X, y = [], []
        for index in index_sample:
            X.append(self.data_logits[index])
            y.append(self.data_labels[index])

            X, y = np.array(X), np.array(y)
            print(X.shape)

            # imgae color list
            hex_list = []
            for name, hex in matplotlib.colors.cnames.items():
                # print(name, hex)
                hex_list.append(str(hex))

            tsne = manifold.TSNE(
                n_components=2, learning_rate='auto', init='pca', random_state=42)
            X_tsne = tsne.fit_transform(X)

            print("Org data dimension is {}. Embedded data dimension is {}".format(
                X.shape[-1], X_tsne.shape[-1]))
            X_norm = X_tsne

            plt.style.use("default")
            plt.figure(figsize=(6, 6))

            plt.grid(True, zorder=0)
            for i in range(X_norm.shape[0]):
                plt.scatter(X_norm[i, 0], X_norm[i, 1],
                            s=15, c="#FF3333", zorder=100)

            plt.xlabel("X", size=17)
            plt.ylabel("Y", size=17)
            plt.xlim((-150, 150))
            plt.xticks(size=17)
            plt.ylim((-150, 150))
            plt.yticks(size=17)

            pic_name_tSNE = "output/fig_TSNE/"+storage_detail+"_legend.pdf"
            plt.title('(a) Vanilla Transformer', y=-0.21, fontsize=18)
            plt.savefig(pic_name_tSNE, dpi=200, bbox_inches='tight')
            plt.show()

    def show_word2vec_tNSE(self, storage_detail):
        plt.style.use("default")
        X, y = np.array(self.data_logits), self.data_labels
        emmbed_dict = {}
        with open('../data/glove.6B.300d.txt','r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:],'float32')
                emmbed_dict[word]=vector
        
        tsne_ = manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)

        y_ = [self.labels[i] for i in y]
        wordvec_ = np.array([emmbed_dict[item] for item in y_])
        X_tsne_ = tsne_.fit_transform(wordvec_)

        print("Org data dimension is {}. Embedded data dimension is {}".format(wordvec_.shape[-1], X_tsne_.shape[-1]))

        # x_min_, x_max_ = X_tsne_.min(0), X_tsne_.max(0)
        # X_norm_ = (X_tsne_ - x_min_) / (x_max_ - x_min_)  # 归一化
        X_norm_ = X_tsne_
        plt.figure(figsize=(8, 8))
        plt.grid(True, zorder=0)

        for i in range(X_norm_.shape[0]):
            # if y[i] == 27:
            #   continue
            plt.scatter(X_norm_[i, 0], X_norm_[i, 1], s=15, color="#0000FF",zorder=100)

        plt.xlabel("X",size=19)
        plt.ylabel("Y",size=19)


        plt.xlim((-150, 150))
        plt.xticks(size=18)
        plt.ylim((-150, 150))
        plt.yticks(size=18)

        pic_name_tSNE = "output/fig_TSNE/"+storage_detail+"_legend.pdf"
        plt.title('(b) Wod2Vec',y=-0.18, fontsize=20)

        plt.savefig(pic_name_tSNE,dpi=200,bbox_inches='tight')
        plt.show()