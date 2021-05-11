

# plot 1 thousand Pg arrivals
import glob
import numpy as np
import matplotlib.pyplot as plt

from obspy import read
from sklearn.linear_model import LogisticRegression


pgfilenames = glob.glob('/Users/albert/Pn/train/*Pg*')[:10000]
print(len(pgfilenames))

pgdatamat = np.zeros((6000,1000))
row = 0
while row < 6000:
	for filename in pgfilenames:
		#while row < 100:
		st = read(filename)
		#st.merge()
		#print(st)
		try:
			st.merge()
			tempst = st.select(sampling_rate=100,channel='*Z').detrend()
			tempst.filter('bandpass',freqmin=1,freqmax=20).normalize()
			if len(tempst) == 0:
				continue
			# check SNR
			snr = np.sqrt(np.sum(tempst[0].data[3000:3500]**2))/np.sqrt(np.sum(tempst[0].data[2500:3000]**2))
			if snr > 1:
				try:
					pgdatamat[row,:] = tempst[0].data[2800:3800]
					row +=1
					#print(row)
				except Exception as e:
					print(e)
		except Exception as e: 
			print(e) 


pnfilenames = glob.glob('/Users/albert/Pn/train/*Pn*')[:10000]
print(len(pnfilenames))

pndatamat = np.zeros((6000,1000))
row = 0
while row < 6000:
	for filename in pnfilenames:
		#while row < 100:
		st = read(filename)
		#st.merge()
		#print(st)
		try:
			st.merge()
			tempst = st.select(sampling_rate=100,channel='*Z').detrend()
			tempst.filter('bandpass',freqmin=1,freqmax=20).normalize()
			if len(tempst) == 0:
				continue
			# check SNR
			snr = np.sqrt(np.sum(tempst[0].data[3000:3500]**2))/np.sqrt(np.sum(tempst[0].data[2500:3000]**2))
			if snr > 1:
				try:
					pndatamat[row,:] = tempst[0].data[2800:3800]
					row +=1
					#print(row)
				except Exception as e:
					print(e)
		except Exception as e: 
			print(e) 

plt.figure(figsize=(5,10))
plt.subplot(2,1,1)
plt.imshow(pgdatamat,cmap='seismic',aspect='auto')
plt.xticks(np.arange(0,1200,200),np.arange(0,11,2))
plt.axvline(150)
plt.axvline(450)
plt.xlabel('seconds')
plt.title('Pg')

plt.subplot(2,1,2)
plt.imshow(pndatamat,cmap='seismic',aspect='auto')
plt.xticks(np.arange(0,1200,200),np.arange(0,11,2))
plt.axvline(150)
plt.axvline(450)
plt.xlabel('seconds')
plt.title('Pn')

plt.savefig('class_examples.png')
"""
#plt.imshow(pgdatamat,cmap='seismic',aspect='auto')
#plt.savefig('pgexamplemat.png')
#plt.imshow(pndatamat,cmap='seismic',aspect='auto')
#plt.savefig('pnexamplemat.png')

trainmatrix = np.concatenate((pndatamat[:5000,150:450],pgdatamat[:5000,150:450]),axis=0)
train_labels = np.concatenate((np.ones(5000),np.zeros(5000)),axis=0).astype(int)
testmatrix = np.concatenate((pndatamat[5000:,150:450],pgdatamat[5000:,150:450]),axis=0)
test_labels = np.concatenate((np.ones(1000),np.zeros(1000)),axis=0).astype(int)

print(trainmatrix.shape)
print(train_labels.shape)
print(testmatrix.shape)
print(test_labels.shape)
np.savez('traindata_nosnr',trainmatrix=trainmatrix,train_labels=train_labels,testmatrix=testmatrix,test_labels=test_labels)


#from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=100000)
log_reg.fit(trainmatrix,train_labels)

train_predictions = log_reg.predict(trainmatrix)
test_predictions = log_reg.predict(testmatrix)
print(test_predictions)
print(test_labels)
#print(predictions.shape)
#print(log_reg.n_iter)

#calculate acccuracy 
train_accuracy = np.sum(train_predictions == train_labels)/len(train_labels)
test_accuracy = np.sum(test_predictions == test_labels)/len(test_labels)
print(train_accuracy)
print(test_accuracy)



# create a matrix with the waveforms from network GI
filenames = glob.glob('test/GI*')
print(len(filenames))

datamat = np.zeros((len(filenames),7000))
datalabels = np.zeros((len(filenames),1))

row=0
for filename in filenames:
	#while row < 100:
	if 'Pn' in filename:
		datalabels[row] = 1
	st = read(filename)
	#st.merge()
	#print(st)
	try:
		st.merge()
		tempst = st.select(sampling_rate=100,channel='*Z').detrend()
		tempst.filter('bandpass',freqmin=1,freqmax=20).normalize()
		if len(tempst) == 0:
			continue
			print('no data')
		# check SNR
		snr = np.sqrt(np.sum(tempst[0].data[3000:3500]**2))/np.sqrt(np.sum(tempst[0].data[2500:3000]**2))
		#print(snr)
		if snr > 0:
			try:
				datamat[row,:] = tempst[0].data[:7000]
				row +=1
				#print(row)
			except Exception as e:
				print(e)
	except Exception as e: 
		print(e) 

print(row)
print(np.sum(datalabels))

np.savez('igdata',testmatrix=datamat,test_labels=datalabels)
"""
