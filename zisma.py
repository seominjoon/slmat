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
		self.data = data[:]

	# Returns the probabilities that self.data and
	# data represent the same object with translation t
	# at slice z
	def sim(self, another, z, t, std=4):
		data = another.data
		init = z
		end = min(len(self.data),len(data)+t)
		errors = np.zeros(end-init)
		gauss = np.zeros(end-init)
		rf = norm(loc=z, scale=std)
		for ind in np.arange(init, end):
			errors[ind-init] = abs(self.data[ind]-data[ind-t])
			gauss[ind-init] = rf.pdf(ind)
		# normalized error
		ne = errors*gauss
		# euclidian distance (rss)
		ed = np.linalg.norm(ne)
		return ed

	# optimal translation, calculated via correlation
	def trnsl(self, another, z, std=4):
		sumlen = len(self.data)+len(another.data)
		vals = np.zeros(sumlen)
		for t in np.range(sumlen):
			init = max(t,len(self.data))
			end = min(t,len(another.data))+len(self.data)
			
			sw = wave(self.data[init-t,end-t]
			aw = wave(another.data[init-len(self.data),end-len(self.data)])
			swn = sw
				
		vals = np.correlate(self.data,data,'full')
		t =  corlen - int(np.argmax(vals))
		return t

	# precondition: self.data and another.data have same length
	def metric(self, another):
		if len(self.data) != len(another.data):
			raise NameError('need to be same length')
		num = len(self.data)
		sumval = 0
		for ind in len(self.data): 
			sumval += abs(self.data-another.data)
		return sumval / num

	# perform gaussian normalization on self.data centered at z
	def norm(self, z, std):
		rf = norm(loc=z, scale=std)
		result = self.data[:]
		for ind in len(self.data):
			result[ind] *= rf.pdf(ind-z)
		return wave(result)
	
	def get(self, z):
		return self.data[z]
		

class draw:
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
		self.bok.on_clicked(self.onbutton)

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
			seq[ind] = self.pixAvg(imgs[ind], x, y, 5)
		return seq.values()

	def finddel(self, arr, element):
		for ind in range(len(arr)):
			if arr[ind] == element:
				del arr[ind]
				return

	def trans(self, seq):
		return np.abs(np.fft.rfft(seq-np.mean(seq)))

	def sim (self, seq1, seq2):
		s1 = np.std(seq1)
		m1 = np.mean(seq1)
		s2 = np.std(seq2)
		m2 = np.mean(seq2)
		n1 = seq1-m1
		n2 = seq2-m2
		if s1 > 0:
			n1 /= s1
		if s2 > 0:
			n2 /= s2
		vals = np.correlate(n1,n2,'full')
		z =  min(len(seq1),len(seq2)) - int(np.argmax(vals))
		return (z,max(vals))

	def onclick(self, event):
		if event.inaxes == self.topleft:
			if self.topline:
				self.finddel(self.topright.lines, self.topline)
			if self.topmarker:
				self.finddel(self.topleft.lines, self.topmarker)
			x = round(event.xdata)
			y = round(event.ydata)
			seq = self.tempSeq(self.imgs, x, y)
			self.topseq = seq
			self.topx = x
			self.topy = y
			self.topline = self.topright.plot(seq)[0]
			self.topmarker = self.topleft.plot(x, y, '-ro')[0]
			# draw transformed graph
			# self.toptrans.plot(self.trans(seq))
			plt.show()
		if event.inaxes == self.topright:
			if self.topvline:
				self.finddel(self.topright.lines, self.topvline)
			self.topz = int(round(event.xdata))
			self.topvline = self.topright.axvline(x=self.topz)
			self.topleft.imshow(self.imgs[self.topz])
			plt.show()
		if event.inaxes == self.botleft:
			if self.botline:
				self.finddel(self.botright.lines, self.botline)
			if self.botmarker:
				self.finddel(self.botleft.lines, self.botmarker)
			x = round(event.xdata)
			y = round(event.ydata)
			seq = self.tempSeq(self.imgs2, x, y)
			self.botseq = seq
			self.botline = self.botright.plot(seq)[0]
			self.botmarker = self.botleft.plot(x, y, '-ro')[0]
			# draw transformed graph
			# self.bottrans.plot(self.trans(seq))
			plt.show()
		if event.inaxes == self.botright:
			if self.botvline:
				self.finddel(self.botright.lines, self.botvline)
			self.botz = int(round(event.xdata))
			self.botvline = self.botright.axvline(x=self.botz)
			self.botleft.imshow(self.imgs2[self.botz])
			plt.show()

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
				
	def onbutton(self, event):
		max_x = self.findmax()
			
		if self.botline:
			self.finddel(self.botright.lines, self.botline)
		if self.botmarker:
			self.finddel(self.botleft.lines, self.botmarker)
		if self.botvline:
			self.finddel(self.botright.lines, self.botvline)

		self.botseq = self.tempSeq(self.imgs2, max_x[0], max_x[1])
		self.botline = self.botright.plot(self.botseq)[0]
		self.botmarker = self.botleft.plot(max_x[0],max_x[1],'-ro')[0]
		self.botz = self.topz + max_x[2]

		self.botvline = self.botright.axvline(x=self.botz)
		self.botleft.imshow(self.imgs2[self.botz])
		plt.show()

	# invoked when "Compare" button is clicked
	def oncmp(self, event):
		# calculates optimal translation and similitude measure

		w1 = wave(self.topseq)
		w2 = wave(self.botseq)
		t = w1.trnsl(w2,self.topz)
		dist = w1.sim(w2,self.topz,t)
		print (t, dist)

	def movez(self, z):
		return None	


	def movexy(self, x, y):
		return None

draw(sys.argv[1], sys.argv[2])

