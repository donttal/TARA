from collections import OrderedDict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, manifold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

random.seed(42)


class visual(object):
    def __init__(self,data_labels, data_preds, labels) -> None:
        confusion_mat = confusion_matrix(data_labels, data_preds)
        print("confusion_mat.shape : {}".format(confusion_mat.shape))

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
        

        fig, ax = plt.subplots(figsize=(12,12))
        disp.plot(
            include_values=True,            
            cmap="YlGnBu",                 
            ax=ax,                        
            xticks_rotation="vertical",  
            values_format="d"              
        )
        plt.show()

    def show_tSNE(self, ):
        pass
        

