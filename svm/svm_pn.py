
# run on obspy_dev
import numpy as np
import pickle
import sklearn
from sklearn import svm
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# this is for a smaller dataset, ignore
"""
data = np.load('traindata.npz')
print(len(data['trainmatrix']))
trainmatrix = data['trainmatrix']
trainlabels = data['train_labels']
testmatrix  = data['testmatrix']
testlabels  = data['test_labels']

classifier = svm.SVC(kernel='rbf',gamma=2)
classifier.fit(trainmatrix,trainlabels)
y = classifier.predict(testmatrix)
print(y)
yy = classifier.predict(trainmatrix)
print(yy)
"""

data = np.load('../traindata_nosnr.npz')

print(len(data['trainmatrix']))
trainmatrix = data['trainmatrix']
trainlabels = data['train_labels']
testmatrix  = data['testmatrix']
testlabels  = data['test_labels']

chilematrix = np.load('../sections/chilemat.npz')['arr_0'][:,50:350]


#gammas = [0.01,0.1,1,2]
classifier = svm.SVC(kernel='rbf',gamma=2)
classifier.fit(trainmatrix,trainlabels)

# option number two is classifier.predict_proba() 

test_preds = classifier.predict(testmatrix)

train_preds = classifier.predict(trainmatrix)

chile_preds = classifier.predict(chilematrix)
print(chile_preds)

# probabilistic classification
#classifier2 = svm.SVC(kernel='rbf',gamma=2,probability=True)
#classifier2.fit(trainmatrix,trainlabels)
#test_probs = classifier2.predict_proba(testmatrix)
#test_probs = test_probs[:,1]


#save the classifiers
outname = 'svm_classifier.sav'
pickle.dump(classifier,open(outname,'wb'))
outname = 'svm_classifier_probabilistic.sav'
#pickle.dump(classifier2,open(outname,'wb'))

# for future reference
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)



train_acc = np.sum(train_preds == trainlabels)/len(train_preds)
test_acc = np.sum(test_preds == testlabels)/len(test_preds)
print('train accuracy', train_acc)
print('test accuracy',test_acc)
F1        = sklearn.metrics.f1_score(testlabels,test_preds)
precision = sklearn.metrics.precision_score(testlabels,test_preds)  
recall    = sklearn.metrics.recall_score(testlabels,test_preds)
roc_auc   = sklearn.metrics.roc_auc_score(testlabels,test_probs)
ConfusionMatrix=sklearn.metrics.confusion_matrix(testlabels, test_preds)
Tpn, Fpg, Fpn, Tpg = ConfusionMatrix.ravel()
print(Tpn, Fpg, Fpn, Tpg,'Tpn, Fpg, Fpn, Tpg')

print('test accuracy ',test_acc)
print('F1 score ', F1)
print('precision ',precision)
print('recall ',recall)
"""
# plotting the roc curve
fpr,tpr, thresholds = roc_curve(testlabels,test_probs)

plt.figure(figsize=(5,5))
plt.plot(fpr,tpr,c='r',label='Area Under ROC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],  linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('auroc_svm.png')

# predictions for the IG data, the full 700 samples are there, so there is a subsettting


gidata = np.load('igdata.npz')
gitestmatrix     = gidata['testmatrix'][:,2800:3800]
gitestmatrix     = gitestmatrix[:,150:450]
gitestlabels     = gidata['test_labels']
gitest_predictions = classifier.predict(gitestmatrix)
print(gitest_predictions.shape)
print(gitestmatrix.shape)


test_acc = np.sum(gitest_predictions.flatten() == gitestlabels.flatten())/len(gitestlabels.flatten())
print('IG network results')
#print(test_predictions)
print(np.sum(gitest_predictions == gitestlabels))
#print(len(testlabels))
print('test accuracy',test_acc)
"""



