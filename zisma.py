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

# drawing application
class drawer:
	def __init__(self, topimgs, botimgs):
		self.topimgs = topimgs
		self.topylim, self.topxlim = self.topimgs[0].shape

		self.botimgs = botimgs
		self.botylim, self.botxlim = self.botimgs[0].shape

		# set up analyzer
		self.anlz = analyzer(topimgs, botimgs)

		# set up canvas
		self.fig = plt.figure()
		self.topleft = self.fig.add_subplot(221)
		self.topleft.set_autoscaley_on(False)
		self.topleft.set_autoscalex_on(False)
		plt.ylim([0,self.topylim])
		plt.xlim([0,self.topxlim])

		self.topright = self.fig.add_subplot(222)
		self.topright.set_autoscaley_on(False)
		self.topright.set_autoscalex_on(False)
		plt.ylim([0,255])
		plt.xlim([0,len(self.topimgs)])

		#self.toptrans = self.fig.add_subplot(233)

		self.botleft = self.fig.add_subplot(223)
		self.botleft.set_autoscaley_on(False)
		self.botleft.set_autoscalex_on(False)
		plt.ylim([0,self.botylim])
		plt.xlim([0,self.botxlim])

		self.botright = self.fig.add_subplot(224)
		self.botright.set_autoscaley_on(False)
		self.botright.set_autoscalex_on(False)
		plt.ylim([0,255])
		plt.xlim([0,len(self.botimgs)])

		#self.bottrans = self.fig.add_subplot(236)

		plt.subplots_adjust(bottom=0.2)
		self.axok = plt.axes([0.4,0.05,0.1,0.075])
		self.bok = Button(self.axok, 'OK')
		self.bok.on_clicked(self.onok)

		self.axcmp = plt.axes([0.6,0.05,0.1,0.075])
		self.bcmp = Button(self.axcmp, 'Compare')
		self.bcmp.on_clicked(self.oncmp)

		# variable to store marker, line, and vline
		self.topmarker = None
		self.topline = None
		self.topvline = None
		self.botmarker = None
		self.botline = None
		self.botvline = None

		self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.topleft.imshow(self.topimgs[0])
		self.botleft.imshow(self.botimgs[0])
		plt.show()

	def finddel(self, arr, element):
		for ind in range(len(arr)):
			if arr[ind] == element:
				del arr[ind]
				return

	def onclick(self, event):
		if event.inaxes == self.topleft:
			self.movetopxy(event.xdata, event.ydata)
		if event.inaxes == self.topright:
			self.movetopz(event.xdata)
		if event.inaxes == self.botleft:
			self.movebotxy(event.xdata, event.ydata)
		if event.inaxes == self.botright:
			self.movebotz(event.xdata)
				
	def onok(self, event):
		self.movebotz(self.anlz.trnsl(self.topz))

	# invoked when "Compare" button is clicked
	def oncmp(self, event):
		# calculates optimal translation and similitude measure
		ot, dist = self.topseq.trnsl(self.botseq,self.topz)
		self.movebotz(ot)
		print dist

	def movetopxy(self, x, y):
		if self.topline:
			self.finddel(self.topright.lines, self.topline)
		if self.topmarker:
			self.finddel(self.topleft.lines, self.topmarker)
		self.topx = round(x)
		self.topy = round(y)
		self.topseq = analyzer.tempseq(self.topimgs, self.topx, self.topy)
		self.topline = self.topright.plot(self.topseq.data)[0]
		self.topmarker = self.topleft.plot(x, y, '-ro')[0]
		plt.show()
		
	def movebotxy(self, x, y):
		if self.botline:
			self.finddel(self.botright.lines, self.botline)
		if self.botmarker:
			self.finddel(self.botleft.lines, self.botmarker)
		self.botx = round(x)
		self.boty = round(y)
		self.botseq = analyzer.tempseq(self.botimgs, self.botx, self.boty)
		self.botline = self.botright.plot(self.botseq.data)[0]
		self.botmarker = self.botleft.plot(x, y, '-ro')[0]
		plt.show()

	def movetopz(self, z):
		if self.topvline:
			self.finddel(self.topright.lines, self.topvline)
		self.topz = int(round(z))
		self.topvline = self.topright.axvline(x=self.topz)
		self.topleft.imshow(self.topimgs[self.topz])
		plt.show()

	def movebotz(self, z):
		if self.botvline:
			self.finddel(self.botright.lines, self.botvline)
		self.botz = int(round(z))
		self.botvline = self.botright.axvline(x=self.botz)
		self.botleft.imshow(self.botimgs[self.botz])
		plt.show()

# descriptor for z-axis wave
class wave:
	def __init__(self, data):
		self.data = np.array(data)

	# optimal translation, calculated via metric function
	# the function is normalized by norm function around z
	def trnsl(self, another, z, trim=0.1):
		slen = len(self.data)
		alen = len(another.data)
		tlen = int(min(slen,alen)*trim) # trim length
		sumlen = slen+alen
		vals = np.zeros(sumlen-2*tlen)
		for t in np.arange(tlen, sumlen-tlen):
			init = max(t,alen)
			end = min(t,slen)+alen
			vals[t-tlen] = self.metric(another,init-alen,end-alen,init-t,end-t)
		# optimal translation for another
		am = int(np.argmin(vals))
		dist = vals[am]
		ot =  am + tlen - alen
		return (z-ot, dist)

	# measures the distance between two waves
	# the greater, the more different are the waves
	def metric(self, another, si, se, ai, ae):
		num = se-si
		if num != ae-ai:
			raise NameError('need to be same length')
		sumval = 0.0
		for ind in range(num): 
			sumval += abs(self.data[ind+si]-another.data[ai+ind])
		return sumval / num

	# perform gaussian normalization on self.data centered at z
	# not used for now
	def norm(self, z, std):
		rf = norm(loc=z, scale=std)
		result = self.data[:]
		for ind in range(len(self.data)):
			result[ind] *= rf.pdf(ind-z)
		return wave(result)

# core analyzing algorithms
class analyzer:
	def __init__(self, topimgs, botimgs):
		self.topimgs = topimgs
		self.botimgs = botimgs

	@staticmethod
	def pixavg (pic, x, y, size):
		val = 0
		num = 0
		yi = -size
		while yi <= size:
			xs = size - abs(yi)
			xi = -xs
			while xi <= xs:
				val += pic[y+yi,x+xi]
				num += 1
				xi += 1
			yi += 1
		return val/num
		
	@staticmethod
	def tempseq (imgs, x, y):
		seq = np.zeros(len(imgs))
		for ind in range(len(imgs)):
			seq[ind] = analyzer.pixavg(imgs[ind], x, y, 2)
		return wave(seq)
	
	def trnsl (self, z): 
		# given z in flt, select a few random points 
		# with variance higher than given threshold
		# assuming no translation between the two images
		num = 20 # make 10 random selections
		ots = np.zeros(num)
		for ind in range(num):
			ots[ind] = self.randtrnsl(z)
		# print ots
		return sp.stats.mode(ots)[0][0]
	
	# perform single random selection of point
	# and return optimal translation
	def randtrnsl(self, z, xi=100, xf=400, yi=100, yf=400):
		x = int(round(xi + np.random.rand(1)[0]*(xf-xi)))
		y = int(round(yi + np.random.rand(1)[0]*(yf-yi)))
		w1 = analyzer.tempseq(self.topimgs, x, y) 
		w2 = analyzer.tempseq(self.botimgs, x, y)
		# print "(%d,%d)" %(x,y)
		ot, dist = w1.trnsl(w2, z)
		return ot

# fetches all imeages from folderdir
def fetcher(folderdir):
	files = sorted(glob(os.path.join(folderdir, '*.jpg')))
	topimgs = {} 
	for ind in range(len(files)):
		topimgs[ind] = mpimg.imread(files[ind])
	return topimgs.values()

# start application with given arguments
# the first arg is flt, the second is ref
topimgs = fetcher(sys.argv[1])
botimgs = fetcher(sys.argv[2])
d = drawer(topimgs, botimgs)

"""
job sequences
0. solve the recursion max reach problem
1. tester that sequentially perform tests on many data sets (mainly GUI based)
2. automatically choose xi, xf, yi, yf in randtrnsl
3. improve metric system between waves (localize)
4. find way to match x,y on flt and ref, i.e. translation (***)
	ideally this can be param'ized and machine learning can solve it
5. matching when different scale (either x,y, or z)
6. systemize testing and result recording
7. implement machine learning

"""
