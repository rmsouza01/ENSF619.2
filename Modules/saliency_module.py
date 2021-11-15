import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["CUDA_VISIBLE_DEVICES"] = " "

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Saliency:

    def __init__(self, model):

        self.model = model

    def plot_heatmap(self, R, x = None, name_class='', save_path=''):
        R = R.numpy().sum(axis=3)
        R = np.abs(R)
        R = R/np.max(R)
        #coolwarm
        #bwr
        #jet
        cmap = cm.Greys
        #cmap = 'seismic'
        img = np.max(R[0]) - R[0]
        norm = Normalize(vmin=0.0, vmax=1.0)
        mScalar = cm.ScalarMappable(norm=norm, cmap=cmap)
        map_img = mScalar.to_rgba(img)#applying colormap in the image

        if not x is None:
            x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
            if x.shape[3] == 1:
                rgb_img = np.empty((x.shape[1], x.shape[2], 3), dtype=float)
                rgb_img[:,:,0] = x_norm[0,:,:,0]
                rgb_img[:,:,1] = x_norm[0,:,:,0]
                rgb_img[:,:,2] = x_norm[0,:,:,0]
            else:
                rgb_img = x_norm[0]

            #print(np.min(map_img), np.max(map_img))
            plt.figure(figsize=(10,5), dpi=350)
            plt.subplot(1,3,1)
            plt.title('Original Image')
            plt.imshow(rgb_img[...,::-1])#for vgg16
            plt.text(10, map_img.shape[0]-10, name_class, color='red', bbox=dict(fill=False, edgecolor='red', linewidth=2))
            plt.axis('off')

            ax = plt.subplot(1,3,2)
            plt.title('Heatmap')
            plt.imshow(map_img)
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(mScalar, cax=cax)

            plt.subplot(1,3,3)
            plt.title('Image + Heatmap')
            plt.imshow(0.3*rgb_img[...,::-1]+0.7*map_img[:,:,0:3])
            plt.axis('off')
            plt.savefig(save_path)

            plt.figure(figsize=(3,3), dpi=350)
            mask = np.zeros(rgb_img.shape)
            mask[:,:,0] = R>0.3
            mask[:,:,1] = R>0.3
            mask[:,:,2] = R>0.3
            plt.imshow((rgb_img*mask)[...,::-1])
            plt.axis('off')
            plt.savefig('C:\\Users\\rober\\OneDrive\\Desktop\\mapas\\salient_map_overlap.png')

        else:
            plt.figure(figsize=(3,3), dpi=350)
            plt.imshow(map_img, interpolation='none', cmap=cmap)
            plt.axis('off')
            plt.colorbar(mScalar)
            plt.savefig(save_path)

    def run(self, x, classIdx, name_class = ''):

        print('class id:', classIdx)
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as g:
            g.watch(x)
            ypred = self.model(x)
            loss = ypred[:, classIdx]

        grads = g.gradient(loss, x)
        self.plot_heatmap(grads, x, name_class, save_path='C:\\Users\\rober\\OneDrive\\Desktop\\mapas\\salient_map.png')

        return grads
