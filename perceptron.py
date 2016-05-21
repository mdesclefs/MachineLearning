# -*- coding:utf-8 -*-
import random

import matplotlib.pyplot as plt
plt.style.use("bmh")



class Perceptron:

	def __init__(self, d):
		self.rangeX = [-1, 1]
		self.rangeY = [-1, 1]
		self.d = d

	def getRandomPoint(self):
		return [self.getRandom(), self.getRandom(self.rangeY)]

	def getRandom(self, specificRange=None):
		if specificRange == None:
			return random.uniform(self.rangeX[0], self.rangeX[1])
		else:
			return random.uniform(specificRange[0], specificRange[1])

	def constructTargetFunction(self):
		""" Construction target function from 2 randoms points """
		# Construct random target function f
		self.target = [self.getRandomPoint(), self.getRandomPoint()]

		"""
		# Function target : y=mx+b
		#y = (y2-y1/x2-x1) (x-x1) + y1
		m = (rPB[1] - rPA[1]) / (rPB[0] - rPA[0])
		b = rPA[1] - (m*rPA[0])

		self.targetFunction = "y = "+str(round(m, 5))+"x"
		if (b>0): self.targetFunction+="+"+str(round(b, 5))
		else: self.targetFunction+=str(round(b, 5))
		"""
	
	def constructTrainingSet(self, N):
		""" Construct training set with N points """
		self.dataset = []

		# Construct training set : [ [point(x,y), [+, -]], ...]
		for i in range(N):
			x = [self.getRandom() for j in range(self.d)]
			self.dataset.append([x, self.getY(x)])


	def initWeightVector(self):
		self.weightVector = [0 for i in range(self.d+1)]

	def run(self, N, turn):
		""" Run (turn) turns of PLA with N points"""
		probTot = 0
		convergeTot = 0

		for i in range(turn):
			self.constructTargetFunction()
			self.constructTrainingSet(N)

			convergeTot+=self.PLA(2000)
			probTot+=self.computeMissedProbability(turn)
			
		avgConverge = convergeTot/turn
		missedProb = probTot/turn
		print("Average iteration convergence: "+str((avgConverge)))
		print("Missed probability: "+str((missedProb)))

		return avgConverge, missedProb

	def PLA(self, maxIteration):

		self.initWeightVector()
		iterationNbr = 0
		while iterationNbr < maxIteration : # The classification is bounded to 5000 turns
			# Construct the misclassified point vector
			self.misclassifiedVector = self.constructMisclassifiedVector()

			# As long as misclassified point exists
			if (len(self.misclassifiedVector) == 0):
				break
			mcX = self.pickMisclassified()

			#Update the weight vector
			self.updateW(mcX)
			iterationNbr+=1

		return iterationNbr

	def computeMissedProbability(self, turn):
		probTmp = 0
		for j in range(turn):
			p = self.getRandomPoint()
			y = self.getY(p)
			estimatedY = self.getH(p)

			if(y != estimatedY):
				probTmp+=1

		return probTmp/turn

	def constructMisclassifiedVector(self):
		""" Construct the misclassified vector """
		vector = []
		for data in self.dataset:
			if self.getH(data[0]) != data[1]: # if the training function gives the wrong class value
				vector.append(data)

		return vector

	def pickMisclassified(self):
		""" Pick a random point from the misclassified vector """
		x = self.misclassifiedVector[random.randrange(len(self.misclassifiedVector))]
		return x

	def getY(self, point):
		""" Get the point class using the target function """
		if ( ((self.target[1][0]-self.target[0][0]) * (point[1]-self.target[0][1])) - ((self.target[1][1]-self.target[0][1]) *(point[0]-self.target[0][0])) > 0): 
			return 1
		return -1


	def getH(self, point):
		""" Get the point class using the training function """
		if ((self.weightVector[0]) + (self.weightVector[1] * point[0]) + (self.weightVector[2] * point[1]) >= 0):
			return 1
		return -1

	def updateW(self, point):
		""" Update weight vector by classifying a misclassified point """
		y = point[1]
		self.weightVector[0]+=y
		self.weightVector[1]+=y*point[0][0]
		self.weightVector[2]+=y*point[0][1]

if __name__ == '__main__':
	perceptron = Perceptron(d=2)
	#print("N = 10")
	#perceptron.run(10, 1000)

	#print("\nN = 100")
	#perceptron.run(100, 1000)

	"""
	c= ['b','g','r','c','m','y', "orange","saddlebrown"]
	"""
	avgConverge = []
	missedProb = []

	for n in range(100):
		tmpConverge, tmpProb = perceptron.run(n, 1000)
		avgConverge.append(tmpConverge)
		missedProb.append(tmpProb)
	

	nFile = open("nEvolution.txt", 'w')

	nFile.write(str(avgConverge))
	nFile.write("\n")

	nFile.write(str(missedProb))
	nFile.write("\n")

	nFile.close()

	"""
	avgConverge = []
	missedProb = []

	for turn in range(1000)
		tmpConverge, tmpProb = perceptron.run(100, n)
		avgConverge.append(tmpConverge)
		missedProb.append(tmpProb)

	nFile = open("runEvolution.txt", 'w')

	nFile.write(str(avgConverge))
	nFile.write("\n")

	nFile.write(str(missedProb))
	nFile.write("\n")
	"""

	nFile.close()