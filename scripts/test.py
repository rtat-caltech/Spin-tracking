import fileCompression as fc
import os


directory = '/netscratch/dmathews33/integratorTesting/'
files = os.listdir(directory)
outputExt = '.h5'

inputNames = []
outputNames = []
for f in files:
	if os.path.splitext(f)[1]=='.out':
		inputNames.append(directory+f)
		outputNames.append(directory+os.path.splitext(f)[0]+outputExt)
for i,o in zip(inputNames, outputNames):
	print(i, o)
	try:
		fc.compressFile(i, o)
	except:
		print('\t well that didnt work')
