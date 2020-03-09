
# You can calculate 1. AUC mean and 2. AUC standard deviation by using this code.

import os

from statistics import stdev 


arr = os.listdir()

for x in arr:

	file = open(x, 'r') 
	Lines = file.readlines()

	Final = []
	for y in Lines:
		if y[:4] == "acc:":
			Final += [float(y[5:])]
	
	print(len(Final))
	print(Lines[0])
	print(mean(Final))
	print(stdev(Final))

print("End")


