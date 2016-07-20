from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pdb
import pylab

# pickle.dump( favorite_color, open( "save.p", "wb" ) )
bside = pickle.load(open("backwardOut.p", "rb"))['data']


# nx = 100
# ny = 100
# nz = 100
# nx = 3
# ny = 277
# nz = 277
#
# odata = np.random.rand(nx,ny,nz)
# data = bside[0]#np.random.rand(nx,ny,nz)
#
# fig = plt.figure(1, figsize=(6,6))
# main_ax = fig.add_axes([0.1,0.2,0.8,0.7])
# slider_ax  = fig.add_axes([0.1,0.1,0.8,0.05])
#
# main_ax.imshow(data[:,:,0], aspect='auto')
#
#
# my_slider = Slider(slider_ax, 'layer', 0, nz, valinit=0, valfmt='%d')
#
# def update(val):
#     main_ax.imshow(data[:,:,int(val)], aspect='auto')
#     plt.draw()
#
# my_slider.on_changed(update)
# plt.show()

# pdb.set_trace()
# print odata.shape
print bside[0].shape


#figure show: backward network activation on data layer
bside2 = bside[0].reshape(227,227,3)
bside2 *= 255.0/bside2.max()
pylab.imshow(bside2)
pylab.show()
