import matplotlib.pyplot as plt

plt.style.use("bmh")


def evalFile(path):
	f = open(path, 'r')
	res = []
	for line in f:
		res.append(eval(line))
	f.close()
	return res

def handlePlot(xLabel, yLabel, title, data, legend, fOut):
	c= ['b','g','r','c','m','y', "orange","saddlebrown"]

	for i in range(len(data)):
		plt.figure(figsize=(7,3))   
		plt.title(title)
	
		"""
		# Iteration
		newData = []
		tmp = 0
		j = 0
		for dta in data[i]:
			tmp+=dta
			if (j%10 == 0):
				tmp/=10
				print(tmp)
				newData.append(tmp)
				tmp=0
			j+=1
		newData.append(data[i][-1])
		"""

		#plt.plot(range(0, 110, 10), newData, label=legend[i])
		plt.plot(data[i], label=legend[i])   
		#plt.axis([0, 100, 0, 110])  

		plt.ylabel(yLabel[i])
		plt.xlabel(xLabel)
			
		#plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0),\
		#ncol=2, fancybox=True, shadow=True)

		plt.savefig(fOut+yLabel[i].replace(" ", "")+".png")
		plt.cla()
		plt.clf()

def readAndPlot(xLabel, yLabel, title, legend, fOut, file):
	
	data = evalFile(file)
	handlePlot(xLabel, yLabel, title, data, legend, fOut)



if __name__ == "__main__":

	readAndPlot("Length of training set", ["Number of iteration", "Misclassified probability"], "Evolution of training set lenght", ["Average iteration number", "Missed classification probability"], "nEvolution", 'nEvolution.txt')
