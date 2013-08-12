#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from glob import glob
from matplotlib.widgets import Button

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
		self.topleft = self.fig.add_subplot(231)
		self.topleft.set_autoscaley_on(False)
		self.topleft.set_autoscalex_on(False)
		plt.ylim([0,self.topylim])
		plt.xlim([0,self.topxlim])

		self.topright = self.fig.add_subplot(232)
		self.topright.set_autoscaley_on(False)
		self.topright.set_autoscalex_on(False)
		plt.ylim([0,255])
		plt.xlim([0,len(self.files)])

		self.toptrans = self.fig.add_subplot(233)

		self.botleft = self.fig.add_subplot(234)
		self.botleft.set_autoscaley_on(False)
		self.botleft.set_autoscalex_on(False)
		plt.ylim([0,self.botylim])
		plt.xlim([0,self.botxlim])

		self.botright = self.fig.add_subplot(235)
		self.botright.set_autoscaley_on(False)
		self.botright.set_autoscalex_on(False)
		plt.ylim([0,255])
		plt.xlim([0,len(self.files2)])

		self.bottrans = self.fig.add_subplot(236)

		plt.subplots_adjust(bottom=0.2)
		self.axok = plt.axes([0.475,0.05,0.1,0.075])
		self.bok = Button(self.axok, '')
		self.bok.on_clicked(self.onbutton)

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
			seq[ind] = self.pixAvg(imgs[ind], x, y, 7)
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
		vals = np.correlate((seq1-m1)/s1,(seq2-m2)/s2,'full')
		return max(vals)

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
			self.topline = self.topright.plot(seq)[0]
			self.topmarker = self.topleft.plot(x, y, '-ro')[0]
			# draw transformed graph
			self.toptrans.plot(self.trans(seq))
			plt.show()
		if event.inaxes == self.topright:
			if self.topvline:
				self.finddel(self.topright.lines, self.topvline)
			x = int(round(event.xdata))
			self.topvline = self.topright.axvline(x=x)
			self.topleft.imshow(self.imgs[x])
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
			self.bottrans.plot(self.trans(seq))
			plt.show()
		if event.inaxes == self.botright:
			if self.botvline:
				self.finddel(self.botright.lines, self.botvline)
			x = int(round(event.xdata))
			self.botvline = self.botright.axvline(x=x)
			self.botleft.imshow(self.imgs2[x])
			plt.show()

	def onbutton(self, event):
		print self.sim(self.topseq,self.botseq)

draw(sys.argv[1], sys.argv[2])

