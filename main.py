import numpy
import scipy.special
import pygame
from random import randint
from time import sleep

pygame.init()

class NeuralNet:
	
	def __init__(self, Inputsnodes, hidennodes, outputnodes, E, ID):
		self.inodes = Inputsnodes
		self.hnodes = hidennodes
		self.onodes = outputnodes
		self.E = E
		try:
			self.wih = numpy.load("wihNN" + str(ID) + ".npy")
			self.who = numpy.load("whoNN" + str(ID) + ".npy")
		except:
			self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
			self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		self.activation_function = lambda x: scipy.special.expit(x)
		
	def Train(self, Inputs_list, targets_list):
		Inputss = numpy.array(Inputs_list, ndmin = 2).T
		targetss = numpy.array(targets_list, ndmin = 2).T
		hiden_Inputss = numpy.dot(self.wih, Inputss)
		hiden_outputs = self.activation_function(hiden_Inputss)
		final_Inputss = numpy.dot(self.who, hiden_outputs)
		final_outputs = self.activation_function(final_Inputss)
		output_errors = targetss - final_outputs
		hiden_errors = numpy.dot(self.who.T, output_errors)
		self.who += self.E * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hiden_outputs))
		self.wih += self.E * numpy.dot((hiden_errors * hiden_outputs * (1 - hiden_outputs)), numpy.transpose(Inputss))
		
	def Query(self, Inputs_list):
		Inputss = numpy.array(Inputs_list, ndmin = 2).T
		hiden_Inputss = numpy.dot(self.wih, Inputss)
		hiden_outputs = self.activation_function(hiden_Inputss)
		final_Inputss = numpy.dot(self.who, hiden_outputs)
		final_outputs = self.activation_function(final_Inputss)
		return final_outputs
		
	def OutputWeight(self):
		return [self.who, self.wih]

def Sort(array):
	sort_array = []
	for i in range(len(array)):
		sort_array.append(max(array))
		array.remove(max(array))
	return sort_array

def FloatSplit(float, num):
	array_float = []
	for i in range(len(str(float))):
		array_float.append(str(float)[i])
	i = len(array_float) - 1
	try:
		float_parts = len(str(float).split(".")[1]) + len(str(float).split(".")[0])
	except:
		return float
	while i > float_parts - num:
		array_float[i] = '0'
		i -= 1
	str_float = ""
	for i in range(len(array_float)):
		str_float += array_float[i]
	return str_float

def FieldGenerator(x, y):
	x_array = []
	y_array = []
	for i in range(x):
		x_array.append(i)
		x_array.append(0)
		x_array.append(i)
		x_array.append(x)
	for i in range(y):
		y_array.append(0)
		y_array.append(i)
		y_array.append(x)
		y_array.append(i)
	x_array.append(x)
	y_array.append(y)
	return [x_array, y_array]

class Game:
	
	def __init__(self, size, sx, sy, wx, wy, hx, hy):
		self.size = size
		self.sx = sx * size
		self.sy = sy * size
		self.bx = [size, self.sx - size, self.sx - size, size]
		self.by = [size, size, self.sy - size, self.sy - size]
		self.wx = []
		self.wy = []
		self.hx = []
		self.hy = []
		for i in range(len(wx)):
			self.wx.append(wx[i] * size)
		for i in range(len(wy)):
			self.wy.append(wy[i] * size)
		for m in range(len(hx)):
			self.hx.append(hx[m] * size)
			self.hy.append(hy[m] * size)
			
	def Show(self, window):
		clock = pygame.time.Clock()
		max_effectivity = [0 for i in range(len(self.bx))]
		max_hp = [0 for i in range(len(self.bx))]
		font_size = self.size
		sys_font = pygame.font.SysFont("None", font_size)
		life = [True for i in range(len(self.bx))]
		iterations = 0
		epochs = 0
		max_epochs = 100
		scoreboard = [[] for i in range(len(self.bx))]
		Inputsnodes = 4
		hidennodes = 6
		outputnodes = 5
		E = 0.3
		NN = [NeuralNet(Inputsnodes, hidennodes, outputnodes, E, ID) for ID in range(len(self.bx))]
		run = True
		while run:
			hp = [100 for i in range(len(self.bx))]
			pygame.time.delay(100)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False
			clock.tick(60)
			while epochs < max_epochs:
				if True in life:
					iterations += 1
				if iterations >= 100:
					epochs += 1
					iterations = 0
				if (life[0] == False) and (life[1] == False) and (life[2] == False) and (life[3] == False):
					life = [True for i in range(len(self.bx))]
					hp = [100 for i in range(len(self.bx))]
				Inputs = [[0, 0, 0, 0] for i in range(len(self.bx))]
				targets = [[0, 0, 0, 0, 0] for i in range(len(self.bx))]
				window.fill((0, 0, 0))
				for i in range(len(self.bx)):
					color = hp
					if hp[i] <= 0:
						life[i] = False
					elif life[i]:
						hp[i] -= 1
						if color[i] >= 240:
							color[i] = 240 if color[i] >= 240 else color[i]
						else:
							color = hp
						try:
							Show_hp = [sys_font.render(str(hp[i]), 0, (0, color[i], 0)) for i in range(len(self.bx))]
						except:
							rendered_overflow = sys_font.render("<--- Overflow --->", 0, (255, 0, 0))
							window.blit(rendered_overflow, (self.sx + self.size, 10))
						pygame.draw.rect(window, (0, 0, color[i]), (self.bx[i], self.by[i], self.size, self.size))
						window.blit(Show_hp[i], (self.bx[i], self.by[i]))
						for k in range(len(self.wx)):
							pygame.draw.rect(window, (60, 60, 60), (self.wx[k], self.wy[k], self.size, self.size))
							if (self.bx[i] == self.wx[k]) and ((self.by[i] - self.size) == self.wy[k]):
								Inputs[i] = [0.3, 0, 0, 0]
							elif ((self.bx[i] + self.size) == self.wx[k]) and (self.by[i] == self.wy[k], self.size, self.size):
								Inputs[i] = [0, 0.3, 0, 0]
							elif (self.bx[i] == self.wx[k]) and ((self.by[i] + self.size) == self.wy[k]):
								Inputs[i] = [0, 0, 0.3, 0]
							elif ((self.bx[i] - self.size) == self.wx[k]) and (self.by[i] == self.wy[k]):
								Inputs[i] = [0, 0, 0, 0.3]
							
						for k in range(len(self.hx)):
							pygame.draw.rect(window, (0, 255, 0), (self.hx[k], self.hy[k], self.size, self.size))
							for c in range(self.sx):
								if (self.bx[i] == self.hx[k]) and (self.by[i] - self.size * c == self.hy[k]):
									Inputs[i] = [0.6, 0, 0, 0]
								elif (self.bx[i] + self.size * c == self.hx[k]) and (self.by[i] == self.hy[k]):
									Inputs[i] = [0, 0.6, 0, 0]
								elif (self.bx[i] == self.hx[k]) and (self.by[i] + self.size * c == self.hy[k]):
									Inputs[i] = [0, 0, 0.6, 0]
								elif (self.bx[i] - self.size * c == self.hx[k]) and (self.by[i] == self.hy[k]):
									Inputs[i] = [0, 0, 0, 0.6]
									
						#=== TEACHER ALGHORITHM ===
						if Inputs[i][0] == 0.6:
							targets[i][0] = 1
						elif Inputs[i][1] == 0.6:
							targets[i][1] = 1
						elif Inputs[i][2] == 0.6:
							targets[i][2] = 1
						elif Inputs[i][3] == 0.6:
							targets[i][3] = 1
						elif Inputs[i] == [0.3, 0.3, 0, 0]:
							targets[i][3] = 1
						elif Inputs[i] == [0.3, 0, 0, 0.3]:
							targets[i][2] = 1
						elif Inputs[i] == [0, 0, 0.3, 0.3]:
							targets[i][1] = 1
						elif Inputs[i] == [0, 0.3, 0.3, 0]:
							targets[i][0] = 1
						elif Inputs[i] == [0.3, 0, 0, 0] or [0, 0.3, 0, 0] or [0, 0, 0.3, 0] or [0, 0, 0, 0.3] or [0, 0, 0, 0]:
							targets[i][4] = 1
							
						output = [numpy.argmax(NN[o].Query(Inputs[o])) for o in range(len(NN))]
						for bot in range(len(NN)):
							NN[bot].Train(Inputs[bot], targets[bot])
						if output[i] == 0:
							self.by[i] -= self.size if self.by[i] > self.size else 0
						elif output[i] == 1:
							self.bx[i] += self.size if self.bx[i] < self.sx - self.size else 0
						elif output[i] == 2:
							self.by[i] += self.size if self.by[i] < self.sy - self.size else 0
						elif output[i] == 3:
							self.bx[i] -= self.size if self.bx[i] > self.size else 0
						elif output[i] == 4:
							rnd = randint(1, 2)
							if rnd == 1:
								rnd = randint(1, 2)
								if rnd == 1:
									self.bx[i] += self.size if self.bx[i] < self.sx - self.size else 0
								elif rnd == 2:
									self.bx[i] -= self.size if self.bx[i] > self.size else 0
							elif rnd == 2:
								rnd = randint(1, 2)
								if rnd == 1:
									self.by[i] -= self.size if self.by[i] > self.size else 0
								elif rnd == 2:
									self.by[i] += self.size if self.by[i] < self.sy - self.size else 0
									
						for k in range(len(self.hx)):
							if (self.bx[i] == self.hx[k]) and (self.by[i] == self.hy[k]):
								hp[i] += 15
								self.hx[k] = randint(1, 23) * self.size
								self.hy[k] = randint(1, 23) * self.size
						if targets[i][output[i]] == 1:
							scoreboard[i].append(1)
						else:
							scoreboard[i].append(0)
						scorecard_array = [numpy.asarray(scoreboard[i]) for i in range(len(self.bx))]
						scores = [0 for i in range(len(self.bx))]
						scores[i] = scorecard_array[i].sum()
						effectivity = Sort([float(FloatSplit(scorecard_array[i].sum() / scorecard_array[i].size * 100, len(str(scorecard_array[i].sum() / scorecard_array[i].size * 100)) - 4)) for i in range(len(scoreboard))])
						if effectivity[i] >= 90:
							numpy.save("whoNN" + str(i), NN[i].OutputWeight()[0])
							numpy.save("wihNN" + str(i), NN[i].OutputWeight()[1])
					else:
						continue
				try:
					rendered = [sys_font.render("Effectivity: {0}%. HP: {1}".format(effectivity[i], hp[i]), 0, (0, color[i], 0)) for i in range(len(self.bx))]
				except OverflowError:
					rendered_overflow = sys_font.render("<--- Overflow --->", 0, (255, 0, 0))
					window.blit(rendered_overflow, (self.sx + self.size, 10))
				rendered_time = sys_font.render("Iterations: {0}. Epochs: {1}".format(iterations, epochs), 0, (0, 255, 0))
				rendered_error = [sys_font.render("!!!", 0, (255, 0, 0)) for i in range(len(self.bx))]
				space = 10
				for i in range(len(self.bx)):
					max_effectivity[i] = effectivity[i] if max_effectivity[i] < effectivity[i] else max_effectivity[i]
					max_hp[i] = hp[i] if max_hp[i] < hp[i] else max_hp[i]
					max_hp[i] = 255 if max_hp[i] > 255 else max_hp[i]
					
				#=== INTERFACE ===
				for i in range(len(self.bx)):
					space += self.size
					if targets[i][output[i]] == 0:
							window.blit(rendered_error[i], (285 + 22.5 * self.size, self.sy + space))
					window.blit(rendered[i], (10, self.sy + space))
					window.blit(rendered_time, (self.sx * 1.05, self.size + 20))
					pygame.draw.rect(window, (60, 60, 60), (18 + 10.86 * self.size, self.sy + space - 2, 259, self.size + 4))
					pygame.draw.rect(window, (0, 0, effectivity[i] * 2.4), (20 + 10.86 * self.size, self.sy + space, effectivity[i] * 2.4, self.size))
					pygame.draw.rect(window, (255, 0, 0), (20 + 10.86 * self.size + max_effectivity[i] * 2.55, self.sy + space, 1, self.size))
					pygame.draw.rect(window, (60, 60, 60), (283 + 10.86 * self.size, self.sy + space - 2, 259, self.size + 4))
					try:
						pygame.draw.rect(window, (0, color[i], 0), (285 + 10.86 * self.size, self.sy + space, color[i], self.size))
					except OverflowError:
						rendered_overflow = sys_font.render("<--- Overflow --->", 0, (255, 0, 0))
						window.blit(rendered_overflow, (self.sx + self.size, 10))
					pygame.draw.rect(window, (255, 0, 0), (285 + 10.86 * self.size + max_hp[i], self.sy + space, 1, self.size))
				pygame.display.update()
		pygame.quit()
		
#=== GAME SETTINGS ===
window = pygame.display.set_mode((pygame.display.Info().current_w, pygame.display.Info().current_h), pygame.FULLSCREEN)
size = 20
sx = 25
sy = 25
wx = FieldGenerator(sx, sy)[0]
wy = FieldGenerator(sx, sy)[1]
hx = [randint(1, sx - 1) for i in range(15)]
hy = [randint(1, sy - 1) for i in range(15)]
Game = Game(size, sx, sy, wx, wy, hx, hy)
Game.Show(window)