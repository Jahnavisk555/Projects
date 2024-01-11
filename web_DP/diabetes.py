#IMPORTS
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


#DATA HANDLING
data = pd.read_csv('diabetes.csv')
print(data.head())
print(data.describe())
X = data[["Glucose"]]
Y = data["DiabetesPedigreeFunction"]
Y_ = data["Outcome"]


#DATA ANALYSIS
plt.scatter(X['Glucose'], Y, color='b')
plt.xlabel('Glucose')  
plt.ylabel('DiabetesPedigreeFunction') 
plt.show()


#OBSERVATIONS
print("The highest number of patients has 78 to 108 glucose with the DiabetesPedigreeFunction in between 0.1 and 0.5.")


#LINEAR REGRESSION
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[2]])
print("Predicted value (LR): ",pred[0])
print("Accuracy (LR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Glucose'], Y, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('DiabetesPedigreeFunction') 
plt.show()


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (LGR): ",pred[0])
print("Accuracy (LGR): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('DiabetesPedigreeFunction') 
plt.show()


#SVR
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf')
mdl.fit(X, Y)
pred = mdl.predict([[148]])
print("Predicted value (SVR): ",pred[0])
print("Accuracy (SVR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Glucose'], Y, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')
plt.ylabel('DiabetesPedigreeFunction')
plt.show()



#SVC
from sklearn.svm import SVC
mdl = SVC(kernel='poly')
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (SVC): ",pred[0])
print("Accuracy (SVC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#NB
from sklearn.naive_bayes import GaussianNB
mdl = GaussianNB()
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (NB): ",pred[0])
print("Accuracy (NB): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#KNN
from sklearn.neighbors import KNeighborsClassifier
mdl = KNeighborsClassifier()
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (KNN): ",pred[0])
print("Accuracy (KNN): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
mdl = RandomForestClassifier(criterion='entropy')
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (RFC): ",pred[0])
print("Accuracy (RFC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict([[148]])
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Glucose'], Y, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(max_leaf_nodes=3, random_state=1)
mdl.fit(X, Y_)
pred = mdl.predict([[148]])
print("Predicted value (DTC): ",pred[0])
print("Accuracy (DTC): ",mdl.score(X[:100], Y_[:100])*100)

plt.scatter(X['Glucose'], Y_, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
mdl =  DecisionTreeRegressor(max_depth=3)
mdl.fit(X, Y)
pred = mdl.predict([[148]])
print("Predicted value (DTR): ",pred[0])
print("Accuracy (DTR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Glucose'], Y, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('High/Low') 
plt.show()


#KMEANS
from sklearn.cluster import KMeans
k = 2
mdl = KMeans(n_clusters=k)
mdl.fit(data.iloc[1:])
centroids = mdl.cluster_centers_
print("Centroids: ",centroids)
pred = mdl.predict([[6,148,72,35,0,33.6,50,0.627,1]])
print("Predicted value (KM): ",pred[0])

labels = mdl.labels_
colors = ['blue','red','green','black','purple']
y = 0
for x in labels:
    # plot the points acc to their clusters
    # and assign different colors
    plt.scatter(data.iloc[y,0], data.iloc[y,1],color=colors[x])
    y+=1
        
for x in range(k):
    #plot the centroids
    lines = plt.plot(centroids[x,0],centroids[x,1],'kx')    
    #make the centroid larger    
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
    
title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('Glucose(mins)')
plt.ylabel('DiabetesPedigreeFunction(mins)')
plt.show()




#MULTIPLE LINEAR REGRESSION
X = data[["Glucose","DiabetesPedigreeFunction"]]
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[148,0.627]])
print("Predicted value (MLR): ",pred[0])
print("Accuracy (MLR): ",mdl.score(X[:100], Y[:100])*100)

plt.scatter(X['Glucose'], Y, color='b')
plt.plot(X['Glucose'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('Glucose')  
plt.ylabel('DiabetesPedigreeFunction') 
plt.show()
