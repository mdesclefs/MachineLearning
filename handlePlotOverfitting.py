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

	plt.figure(figsize=(7,3))   
	plt.title(title)

	for i in range(len(data)):
		if fOut == "kEvolution7" :
			rangeY = range(-3, 4)
			data[i] = data[i][27:34]
		else:
			rangeY = range(-30, 20)
		plt.plot(rangeY, data[i], label=legend[i])  

	if fOut == "kEvolution7":
		plt.plot(rangeY, [0.02857142857142857]*len(rangeY), label="In-sample error without regularization")
		plt.plot(rangeY, [0.084]*len(rangeY), label="Out-of-sample error without regularization")

	plt.ylabel(yLabel)
	plt.xlabel(xLabel)

	lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.3), borderaxespad=0.)

	plt.gcf().subplots_adjust(bottom=0.20)
	plt.savefig("./images/"+fOut+".png", bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.cla()
	plt.clf()
	
def plotData(xLabel, yLabel, title, file, fOut):  
	
	plt.figure(figsize=(7,3))
	plt.title(title)
	
	for line in open(file):
		x1, x2, y = line.strip().split()
		if(float(y) > 0):
			point = "bs"
		else:
			point = "g^"
			
		plt.plot(x1, x2, point)
		

	plt.ylabel(yLabel)
	plt.xlabel(xLabel)
	plt.gcf().subplots_adjust(bottom=0.20)
	plt.savefig("./images/"+fOut+".png")#, bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.cla()
	plt.clf()

def readAndPlot(xLabel, yLabel, title, legend, fOut, file):
	data = evalFile(file)
	handlePlot(xLabel, yLabel, title, data, legend, fOut)

if __name__ == "__main__":
	readAndPlot("k", "Error", "Sample error variation", ["In-sample error", "Out-of-sample error"], "kEvolution", './data/kEvolution.txt')
	readAndPlot("k", "Error", "Sample error variation", ["In-sample error", "Out-of-sample error"], "kEvolution7", './data/kEvolution.txt')
	#plotData("x(1)", "x(2)", "Training set representation", "./data/in.dta", "representation")
