# -*- coding:utf-8 -*-
import random

class Perceptron:

	def __init__(self, d):
		self.rangeX = [-1, 1]
		self.rangeY = [-1, 1]
		self.d = d

		self.confusionMatrix = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
	def getRandomPoint(self):
        """ Builds a random point made up of two randoms values """
		return [self.getRandom(), self.getRandom(self.rangeY)]

	def getRandom(self, specificRange=None):
        """ Pick a random value from a specific or default range """
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

		self.normalizeConfusionMatrix(turn*turn)

		return avgConverge, missedProb, self.confusionMatrix

	def PLA(self, maxIteration):
        """" Executes a run of PLA """
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
        """ Computes the probability that a point is misclassified """
		probTmp = 0
		for j in range(turn):
			p = self.getRandomPoint()
			y = self.getY(p)
			estimatedY = self.getH(p)

			if(y != estimatedY):
				probTmp+=1

			self.setConfusionClass(y,estimatedY)

		return probTmp/turn

	def setConfusionClass(self, y, h):
        """ Adds a confusion in the matrix """
		if (y==h):
			if(y==1):
				self.confusionMatrix['tp']+=1
			else:
				self.confusionMatrix['tn']+=1
		else:
			if(y==1):
				self.confusionMatrix['fn']+=1
			else:
				self.confusionMatrix['fp']+=1

	def normalizeConfusionMatrix(self, turn):
		for key in self.confusionMatrix:
			self.confusionMatrix[key]/=turn

	def constructMisclassifiedVector(self):
		""" Constructs the misclassified vector """
		vector = []
		for data in self.dataset:
			if self.getH(data[0]) != data[1]: # if the training function gives the wrong class value
				vector.append(data)

		return vector

	def pickMisclassified(self):
		""" Picks a random point from the misclassified vector """
		x = self.misclassifiedVector[random.randrange(len(self.misclassifiedVector))]
		return x

	def getY(self, point):
		""" Gets the point class using the target function """
		if ( ((self.target[1][0]-self.target[0][0]) * (point[1]-self.target[0][1])) - ((self.target[1][1]-self.target[0][1]) *(point[0]-self.target[0][0])) > 0): 
			return 1
		return -1


	def getH(self, point):
		""" Gets the point class using the training function """
		if ((self.weightVector[0]) + (self.weightVector[1] * point[0]) + (self.weightVector[2] * point[1]) >= 0):
			return 1
		return -1

	def updateW(self, point):
		""" Updates weight vector by classifying a misclassified point """
		y = point[1]
		self.weightVector[0]+=y
		self.weightVector[1]+=y*point[0][0]
		self.weightVector[2]+=y*point[0][1]

if __name__ == '__main__':
	infile = False
	perceptron = Perceptron(d=2)
	#print("N = 10")
	#print(perceptron.run(10, 1000))

	print("\nN = 100")
	print(perceptron.run(100, 1000))

	if(infile):
		avgConverge = []
		missedProb = []

		for n in range(100):
			tmpConverge, tmpProb = perceptron.run(n, 1000)
			avgConverge.append(tmpConverge)
			missedProb.append(tmpProb)

			nFile = open("./data/nEvolution.txt", 'w')

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