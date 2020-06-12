import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset, split_with_shuffle, get_data_labels, split_like_paper, get_cb513
import model

np.set_printoptions(threshold=np.inf)

sample=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y']
print("Enter Amino Acid sequence")
aa=input()
#MNSFSTSAFGPVAFSLGLLLVLPAAFPAPVPPGEDSKDVAAPHRQPLTSSERIDKQIRYILDGISALRKETCNKSNMCESSKEALAENNLNLPKMAEKDGCFQSGFNEETCLVKIITGLLEFEVYLEYLQNRFESSEEQARAVQMSTKVLIQFLQKKAKNLDAITTPDPTTNASLLTKLQAQNQWLQDMTTHLILRSFKEFLQSSLRALRQM'

n=len(aa)
res = [ [ 0 for i in range(21) ] for j in range(700) ] 

#print(str(res))

for i in range(0,n):
	j=0
	for j in range(0,21):
		if aa[i]==sample[j]:
			res[i][j]=1

res=np.array([res])

net = model.CNN_model()
net.load_weights("previous.hdf5")

#dataset = get_dataset()

#D_train, D_test, D_val = split_with_shuffle(dataset, 100)

#X_train, Y_train = get_data_labels(D_train)
#X_test, Y_test = get_data_labels(D_test)
#X_val, Y_val = get_data_labels(D_val)

#predictions = net.predict(np.array([X_test[0]]))

#n =0
#x=X_test[0].tolist()
#for i in range(700):
#	temp=0
#	for j in range(21):
#		if x[i][j]==0:
#			temp=temp+1
#	if temp==21:
#		break
#	n=n+1

#y=predictions[0].tolist()

#for i in range(700):
#	max = y[i][0]
#	if i<n:
#		for j in range(8):
#			if max<y[i][j]:
#				max=y[i][j]
#		for j in range(8):
#			if max>y[i][j]:
#					y[i][j]=0
#			else:
#					y[i][j]=1
#				
#	else:
#		for j in range(8):
#			y[i][j]=0

predictions = net.predict(res)
x=res[0].tolist()
y=predictions[0].tolist()
for i in range(700):
	max = y[i][0]
	if i<n:
		for j in range(8):
			if max<y[i][j]:
				max=y[i][j]
		for j in range(8):
			if max>y[i][j]:
					y[i][j]=0
			else:
					y[i][j]=1
				
	else:
		for j in range(8):
			y[i][j]=0


y=np.array(y)
print(y, file=open("output.txt", "w"))