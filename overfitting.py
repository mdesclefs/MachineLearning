# -*- coding:utf-8 -*-

import math
import numpy as np

class Overfitting:

	def __init__(self):
		dataIn= self.getData("in.dta")
		dataOut = self.getData("out.dta")

		self.nonLinearData = self.convertDataToNonLinear(dataIn)
		nonLinearDataOut = self.convertDataToNonLinear(dataOut)

		self.w = self.computeDecayWeightW(self.nonLinearData, 10**(-3))

		print("In-sample classification errors:", self.getClassificationError(self.nonLinearData))
		print("Out-of-sample classification errors:", self.getClassificationError(nonLinearDataOut))


	def computeW(self, data):
		"""Computes W by using the linear regression algorithm"""
		#We first construct the matrix X and the vector Y from data set
		self.constructX(data)
		self.constructY(data)

		# We compute the pseudo-inverse
		pseudoInverse =  (((self.X.T * self.X).I) * self.X.T)
	
		# We return w = (Pseudo-inverse) y
		return pseudoInverse * self.Y

	def computeDecayWeightW(self, data, lamba):
		"""Computes W by using the linear regression algorithm adding weight decay"""
		#We first construct the matrix X and the vector Y from data set
		self.constructX(data)
		self.constructY(data)

		# We compute the pseudo-inverse edited
		pseudoInverse =  ((((self.X.T * self.X) + (lamba*np.identity(self.d))).I ) * self.X.T)
	
		# We return w = (Pseudo-inverse) y
		return pseudoInverse * self.Y

	def getData(self, file):
		data = []
		for line in open(file):
			x1, x2, y = line.strip().split()
			data.append([float(x1), float(x2), float(y)])

		self.d = len(data) - 1

		return data

	def applyNonLinearTransformation(self, point):
		phi = [1, point[0], point[1], point[0]**2, point[1]**2, point[0]*point[1], math.fabs(point[0] - point[1]), math.fabs(point[0] + point[1]), point[2]]
		return phi

	def convertDataToNonLinear(self, data):
		nonLinearData = []
		for point in data:
			nonLinearData.append(self.applyNonLinearTransformation(point))

		self.d = len(nonLinearData[0]) - 1
		return nonLinearData

	def constructX(self, data):
		self.X = []
		for point in data:
			self.X.append(point[:(self.d)])

		self.X = np.matrix(self.X)


	def constructY(self, data):
		self.Y = []
		for point in data:
			self.Y.append(point[self.d])

		self.Y = np.matrix(self.Y).T


	def getH(self, point):

		res = point[:self.d] * self.w
		if (res >= 0):
			return 1
		return -1

	def getClassificationError(self, data):
		errorNbr = 0
		for point in data:
			if(self.getH(point) != point[self.d]):
				errorNbr+=1

		return errorNbr/(len(data))

if __name__ == '__main__':
	overfitting = Overfitting()