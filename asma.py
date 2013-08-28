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

# hidden markov model
class hmm:
	# dom is the domain of x
	# px is a function that defines prob for x
	# pxx defines prob for x given prev x
	# pex defines prob for e given x
	def __init__(self, box):
		self.dom = box.dom
		self.px = box.px
		self.pxx = box.pxx
		self.pex = box.pex
		self.loc = -1 
		self.len = box.len

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
			newpx[currx] = self.px(currx)*self.pex(evid, currx, self.loc)	
		self.px = newpx/np.sum(newpx)
		
	# returns current loc, and probabilty
	# current loc's evidence given
	def next(self, evid):
		if self.loc >= 0:
			self.time_update()
		self.loc += 1
		self.evid_update(evid)
		return self.px,self.loc
		
class asma:
	def __init__(self, topfs, botfs, ol=0.5):
		self.len = len(topfs)
		self.dom = np.arange(-1,self.len+1)
		self.ol = ol
		self.oi = ol/2
		self.oe = ol/2
		self.xpd = {}
		self.topfs = topfs
		self.botfs = botfs

	def px(self, x):
		if x == -1:
			return self.oi
		elif x == self.len:
			return self.oe
		else:
			return (1-self.ol)/self.len

	def pxx(self, currx, prevx):
		if prevx == -1:
			n = self.oi*self.len
			if currx == -1:
				return (n-1)/n
			if currx == 0:
				return 0.9/n
			if currx == 1:
				return 0.1/n
			else:
				return 0
		elif prevx == self.len-2:
			if currx == self.len-2:
				return 0.1
			elif currx == self.len-1:
				return 0.8
			elif currx == self.len:
				return 0.1
			else:
				return 0
		elif prevx == self.len-1:
			if currx == self.len-1:
				return 0.1
			if currx == self.len:
				return 0.9
			else:
				return 0
		elif prevx == self.len:
			if currx == self.len:
				return 1
			else:
				return 0
		else:
			if currx == prevx:
				return 0.1
			elif currx == prevx+1:
				return 0.8
			elif currx == prevx+2:
				return 0.1
			else:
				return 0

	def pex(self, evid, x, loc):
		if loc not in self.xpd:
			self.xpd[loc] = self.pexh(evid)
		return self.xpd[loc][x]
		
	def pexh(self, evid):
		# stores temp prob values
		out = np.zeros(self.len+2)
		out[self.len+1] = 1 #just to get correct median
		# yeah i know i am lazy
		for ind in np.arange(1,self.len+1):
			product = 1
			z = self.topfs[ind-1]
			scale = np.std(z)
			for fi in range(len(evid)):
				scale = np.std(z)
				product *= norm.pdf(evid[fi],loc=z[fi],scale=scale)
			out[ind] = product
		med = np.median(out)
		out[0] = med
		out[self.len+1] = med
		out /= np.sum(out)
		return out

def getfs(imgs):
	out = []
	for ind in range(len(imgs)):
		out.append(rep(imgs[ind]))	
		print "image %d features extracted." %ind
	return out

topimgs = fetcher(sys.argv[1])[0:3]
botimgs = fetcher(sys.argv[2])[0:3]

topfs = getfs(topimgs)
botfs = getfs(botimgs)
		
myhmm = hmm(asma(topfs, botfs))
for ind in range(len(botfs)):
	px, loc = myhmm.next(botfs[ind])
	plt.plot(px)
	plt.show()
