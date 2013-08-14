#!/usr/bin/python

# Z-axis Intensity-based Signal Matching Analysis (ZISMA)

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from glob import glob
from matplotlib.widgets import Button


class wave:
	def __init__(self, data):
		self.data = np.array(data)

	# optimal translation, calculated via metric function
	# the function is normalized by norm function around z
	def trnsl(self, another, z, std=4, trim=0.1):
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
		return (ot, dist)

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

class app:
	def __init__(self, folderdir, folderdir2):
		self.files = self.fetchFiles(folderdir)
		self.imgs = self.fetchImages(self.files)
		self.topylim, self.topxlim = self.imgs[0].shape

		self.files2 = self.fetchFiles(folderdir2)
		self.imgs2 = self.fetchImages(self.files2)
		self.botylim, self.botxlim = self.imgs2[0].shape

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
		plt.xlim([0,len(self.files)])

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
		plt.xlim([0,len(self.files2)])

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
		self.topleft.imshow(self.imgs[0])
		self.botleft.imshow(self.imgs2[0])
		plt.show()

	def fetchFiles (self, folderdir):
		files_list = glob(os.path.join(folderdir, '*.jpg'))
		return sorted(files_list)

	def fetchImages (self, files):
		imgs = {}
		for ind in range(len(files)):
			imgs[ind] = mpimg.imread(files[ind])
		return imgs.values()
	
	def pixAvg (self, pic, x, y, size):
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
		
	def tempSeq (self, imgs, x, y):
		seq = {}
		for ind in range(len(imgs)):
			seq[ind] = self.pixAvg(imgs[ind], x, y, 2)
		return seq.values()

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

	# function to be maximzed
	def findmax(self):
		maxval = 0
		maxx = 0
		maxy = 0
		maxz = 0
		for x in np.array(range(40))+self.topx-20:
			for y in np.array(range(40))+self.topy-20:
				seq = self.tempSeq(self.imgs2,x,y)
				currz, currval = self.sim(self.topseq,seq)
				if currval > maxval:
					maxval = currval 
					maxx = x
					maxy = y
					maxz = currz
		return (maxx, maxy, maxz)
				
	def onok(self, event):
		x,y,z = self.findmax()
		self.movebotxy(x,y)
		self.movebotz(z)	

	# invoked when "Compare" button is clicked
	def oncmp(self, event):
		# calculates optimal translation and similitude measure
		w1 = wave(self.topseq)
		w2 = wave(self.botseq)
		ot, dist = w1.trnsl(w2,self.topz)
		self.movebotz(self.topz-ot)
		print dist

	def movetopxy(self, x, y):
		if self.topline:
			self.finddel(self.topright.lines, self.topline)
		if self.topmarker:
			self.finddel(self.topleft.lines, self.topmarker)
		self.topx = round(x)
		self.topy = round(y)
		self.topseq = self.tempSeq(self.imgs, self.topx, self.topy)
		self.topline = self.topright.plot(self.topseq)[0]
		self.topmarker = self.topleft.plot(x, y, '-ro')[0]
		plt.show()
		
	def movebotxy(self, x, y):
		if self.botline:
			self.finddel(self.botright.lines, self.botline)
		if self.botmarker:
			self.finddel(self.botleft.lines, self.botmarker)
		self.botx = round(x)
		self.boty = round(y)
		self.botseq = self.tempSeq(self.imgs2, self.botx, self.boty)
		self.botline = self.botright.plot(self.botseq)[0]
		self.botmarker = self.botleft.plot(x, y, '-ro')[0]
		plt.show()

	def movetopz(self, z):
		if self.topvline:
			self.finddel(self.topright.lines, self.topvline)
		self.topz = int(round(z))
		self.topvline = self.topright.axvline(x=self.topz)
		self.topleft.imshow(self.imgs[self.topz])
		plt.show()

	def movebotz(self, z):
		if self.botvline:
			self.finddel(self.botright.lines, self.botvline)
		self.botz = int(round(z))
		self.botvline = self.botright.axvline(x=self.botz)
		self.botleft.imshow(self.imgs2[self.botz])
		plt.show()

# start application with given arguments
app(sys.argv[1], sys.argv[2])

