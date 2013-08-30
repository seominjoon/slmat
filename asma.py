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
	x = np.linspace(-180,180,bins)
	# plt.plot(x,y)
	fy = gaussian_filter(y,sigma)
	# plt.plot(x,fy)
	# plt.xlim(-180,180)
	# plt.show()
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
		self.time = -1 
		self.len = box.len

	def time_update(self):
		newpx = {}
		if self.time < 0:
			for currx in self.dom:
				newpx[currx] = self.px(currx)		 
		else:
			for nextx in self.dom:
				out = 0
				for currx in self.dom:
					out += self.pxx(nextx,currx)*self.prob[currx]
				newpx[nextx] = out
		self.prob = newpx
		self.time += 1 # update time
			
	def evid_update(self, evid):
		newpx = {}
		sumval = 0
		for currx in self.dom:
			newpx[currx] = self.prob[currx]*self.pex(evid, currx, self.time)	
			sumval += newpx[currx]
		for key in newpx.keys():
			newpx[key] = newpx[key]/sumval	
		self.prob = newpx

	# returns current loc, and probabilty
	# current loc's evidence given
	def next(self, evid):
		self.time_update()
		self.evid_update(evid)
		return self.prob,self.time
		
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
		out = {}
		for ind in range(self.len):
			product = 1
			z = self.topfs[ind]
			scale = np.std(z)
			x = np.linspace(-180,180,60)
			plt.plot(x,evid)
			plt.plot(x,z)
			plt.xlabel('angle (degrees)')
			plt.ylabel('bone density')
			plt.show()
			for fi in range(len(evid)):
				scale = np.std(z)
				product *= norm.pdf(evid[fi],loc=z[fi],scale=scale)
			out[ind] = product
		out[-1] = np.mean(out.values()) 
		out[self.len] = out[-1] 
		return out

def getfs(imgs):
	out = []
	for ind in range(len(imgs)):
		out.append(rep(imgs[ind]))	
		print "image %d features extracted." %ind
	return out

topi = int(sys.argv[2])
tope = int(sys.argv[3])
boti = int(sys.argv[5])
bote = int(sys.argv[6])
topimgs = fetcher(sys.argv[1])[topi:tope]
botimgs = fetcher(sys.argv[4])[boti:bote]

def nz(img, r, c):
	maxr,maxc = img.shape
	if r < 0 or c < 0 or r >= maxr or c >= maxc:
		return False
	if img[r,c] > 0:
		return True
	return False

# edge-based feature
img = topimgs[0]
f = {}
f['w'] = 0
f['e'] = 0
f['n'] = 0
f['s'] = 0
f['sw'] = 0
f['nw'] = 0
f['ne'] = 0
f['se'] = 0

dom = range(len(f.keys()))
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
r = 0
c = 0

def inc(d, key):
	d[key] += 1
	ax2.plot(r,c,'x')

for r,c in np.ndindex(img.shape):
	if img[r,c] > 0:
		up = nz(img,r-1,c)
		down = nz(img,r+1,c)
		left = nz(img,r,c-1)
		right = nz(img,r,c+1)
		if up and down and right and (not left):
			inc(f,'w')
		elif up and down and left and (not right):
			inc(f,'e')
		elif left and right and down and (not up):
			inc(f,'n')
		elif left and right and up and (not down):
			inc(f,'s')
		elif up and right and (not down) and (not left):
			inc(f,'sw')
		elif right and down and (not left) and (not up):
			inc(f,'nw')
		elif down and left and (not up) and (not right):
			inc(f,'ne')
		elif left and up and (not right) and (not down):
			inc(f,'se')



ax.bar(dom,f.values())
ax.set_xticks(dom)
ax.set_xticklabels(f.keys())
plt.show()
		

# run angle-based bone detection
topfs = getfs(topimgs)
botfs = getfs(botimgs)

myasma = asma(topfs, botfs)
myhmm = hmm(myasma)
for ind in range(len(botfs)):
	prob, loc = myhmm.next(botfs[ind])
	plt.plot(prob.keys(), prob.values(), 'o')
	plt.ylim(0,1)
	plt.title("evidence #%d" %ind)
	plt.show()
