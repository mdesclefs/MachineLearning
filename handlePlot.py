import math
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
	
		if(i == 0):
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

			plt.plot(range(0, 110, 10), newData, label=legend[i])	
			plt.axis([0, 100, 0, 110])  
		else:
			plt.plot(data[i], label=legend[i])  

		plt.ylabel(yLabel[i])
		plt.xlabel(xLabel)
			
		#plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0),\
		#ncol=2, fancybox=True, shadow=True)

		plt.savefig("./images/"+fOut+yLabel[i].replace(" ", "")+".png")
		plt.cla()
		plt.clf()

def handleSlopePlot(xLabel, yLabel, title, data, legend, fOut):
	c= ['b','g','r','c','m','y', "orange","saddlebrown"]


	factor = 0

	plt.figure(figsize=(7,3))   
	plt.title(title)

	for i in range(len(data)-1, -1, -1):
		if(i == 0):
			factor/= max(data[i]) - min(data[i])
			newData = []
			# Iteration
			tmp = 0
			j = 0
			for dta in data[i]:
				tmp+=dta
				if (j%10 == 0):
					tmp/=10
					newData.append(tmp*factor)
					tmp=0
				j+=1
			#newData.append(data[i][-1])
			newData.insert(0,0)
			data[i] = newData
		else: 
			""" Keep factor between data """
			#factor /= (max(data[i]) - min(data[i]))
			factor = (max(data[i]) - min(data[i]))
			newData = []
			for dta in data[i]:
				newData.append(dta)#*factor)
			newData.insert(0,0.5)#*factor)
			data[i] = newData
		
		slope = []
		step = math.floor(len(data[i]) / 10)
		tmpSlope = 0
		for j in range(0, len(data[i])-step, step):
			tmpSlope = (data[i][j+step] - data[i][j]) / step 
			slope.append(tmpSlope)
		slope.append(tmpSlope)

		plt.plot(range(0, 110, 10), slope, label=legend[i])
			
	#plt.legend(bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True)
	lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), borderaxespad=0.)
  
	plt.ylabel("Slope")
	plt.xlabel(xLabel)

	plt.savefig("./images/"+fOut+"_slope.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.cla()
	plt.clf()

def readAndPlot(xLabel, yLabel, title, legend, fOut, file):
	
	data = evalFile(file)
	#handlePlot(xLabel, yLabel, title, data, legend, fOut)
	handleSlopePlot(xLabel, yLabel, title, data, legend, fOut)



if __name__ == "__main__":

	readAndPlot("Length of training set", ["Number of iteration", "Misclassified probability"], "Evolution of training set lenght", ["Average iteration number", "Missed classification probability"], "nEvolution", './data/nEvolution.txt')
