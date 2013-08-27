#!/usr/bin/python

# Z-axis Intensity-based Signal Matching Analysis (ZISMA)

import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from glob import glob
from matplotlib.widgets import Button
import cv2
from collections import Counter
import math
from scipy.ndimage.filters import gaussian_filter

def fetcher(folderdir):
	files = sorted(glob(os.path.join(folderdir, '*.jpg')))
	topimgs = {} 
	for ind in range(len(files)):
		topimgs[ind] = mpimg.imread(files[ind])
	return topimgs.values()

# represents each image with an bins-d array
# applies gaussian filter
def rep (img, th=250, bins=60, sigma=1):
	rmean = float(0)
	cmean = float(0)

	arr = []
	for r,c in np.ndindex(img.shape):
		if img[r,c] > th:
			arr.append((r,c))
			rmean += r
			cmean += c
	rmean /= len(arr)
	cmean /= len(arr)
	
	angles = []
	for r,c in arr:
		dr = r-rmean
		dc = c-cmean
		# add pi if 2nd quarter; subtract if 3rd quarter
		angle = math.atan(dr/dc)
		if dc < 0:
			if dr > 0:
				angle += math.pi
			else:
				angle -= math.pi
		angles.append(angle)
		rng = (-math.pi,math.pi)

	y,be = np.histogram(angles, bins=bins, range=rng, normed=True)
	fy = gaussian_filter(y,sigma)
	return fy

def cmp(y1, y2,ex=3):
	dys = np.sort(np.abs(y2-y1))
	return np.mean((dys[ex:len(dys)-ex]))

class hmm:
	# dom is the domain of x
	# px is a function that defines prob for x
	# pxx defines prob for x given prev x
	# pex defines prob for e given x
	def __init__(self, dom, px, pxx, pex):
		self.dom = dom
		self.px = px
		self.pxx = pxx
		self.pex = pex
		self.loc = -1 

	def time_update(self):
		newpx = np.zeros(len(self.dom))
		for currx in self.dom:
			out = 0
			for prevx in self.dom:
				out += self.pxx(currx,prevx)*self.px(prevx)
			newpx[currx] = out
		self.px = newpx
			
	def evid_update(self, evid):
		newpx = np.zeros(len(self.dom))
		for currx in self.dom:
			newpx[currx] = self.px[currx]*self.pex(evid)	
		self.px = newpx/np.sum(newpx)
		
	# returns current loc, and probabilty
	# current loc's evidence given
	def next(self, evid):
		if self.loc >= 0:
			self.time_update()
		self.loc += 1
		self.evid_update(evid)
		return self.px,self.loc
		


topimgs = fetcher(sys.argv[1])
botimgs = fetcher(sys.argv[2])

ind = 30
init = 0
end = len(botimgs)
fltimg = topimgs[ind]
fltrep = rep(fltimg)
results = []


for i in np.arange(init,end):
	refimg = botimgs[i]
	refrep = rep(refimg)
	results.append(cmp(fltrep,refrep))
	
plt.plot(results)
plt.show()
