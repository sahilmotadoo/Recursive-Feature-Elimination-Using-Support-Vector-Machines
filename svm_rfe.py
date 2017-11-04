import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

#Function to test model accuracy based on testing data
def accuracy(model, x_test, y_test):
	prediction = model.predict(x_test)
	print "Accuracy of model:", accuracy_score(y_test, prediction) * 100, "%"

#Supress Warnings
warnings.filterwarnings("ignore")

#Set parameter
C = 1.0

#Create DataFrame
df = pd.read_csv('mbscores.csv')
nFeatures = len(df.columns) - 1

#Split data and scores from modified .CSV file
samples = df.filter(['HMM', 'SSD', 'OGS'])
scores = df.filter(['Scores'])

rfeIndex = nFeatures

#Recursively eliminate features based on the lowest weight
while True:
	#Split into training and testing
	x_train, x_test, y_train, y_test = train_test_split(samples, scores, test_size = 0.50, train_size = 0.50)
	
	#Create SVM model using a linear kernel
	model = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
	coef = model.coef_

	#Print co-efficients of features
	for i in range(0, nFeatures):
		print samples.columns[i],":", coef[0][i]
	
	#Find the minimum weight among features and eliminate the feature with the smallest weight
	min = coef[0][0]
	index = 0
	for i in range(0, rfeIndex):
		if min > coef[0][i]:
			index = index + 1
			min = coef[0][i]
	if len(samples.columns) == 1:
		print "After recursive elimination we have the", samples.columns[index], "feature with a score of:", min
		accuracy(model, x_test, y_test)
		break
	else:
		print "Lowest feature weight is for", samples.columns[index], "with a value of:", min
		print "Dropping feature", samples.columns[index]  

		#Drop the feature in the 'samples' dataframe based on the lowest feature index
		samples.drop(samples.columns[index], axis = 1, inplace = True)
		accuracy(model, x_test, y_test)
		print "\n"
		rfeIndex = rfeIndex - 1
		nFeatures = nFeatures - 1
