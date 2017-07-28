# Importing libraries

import pandas as pd
import numpy as np


#Reading the CSV file
dataset=pd.read_csv('Churn_Modelling.csv')

# selecting independent features from 3 to 12
x=dataset.iloc[:,3:13].values

# selecting dependent variable i.e 13
y=dataset.iloc[:,13].values

#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encoding categorical features countries
labelencoder_X_1=LabelEncoder()
x[:,1]=labelencoder_X_1.fit_transform(x[:,1])

#Encoding categorical features Gender
labelencoder_X_2=LabelEncoder()
x[:,2]=labelencoder_X_2.fit_transform(x[:,2])

#since countries are not ordinal variables converting them to dummy variables>2
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

#to avoid falling in dummy variable trap remove one column of dummy varaible
x=x[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)



# Part 2: Making ANN importing keras package and libraries


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN
classifier=Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))
#Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))

# Adding output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.add(Dropout(p=0.1))

# Compiling the neural network means applying gradient to whole ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

#Predicting single value
""" 
Geography: France
Credt Score:60
Gender:male
Age:40
Tenure:3
Balance:60000
Number of Products:2
Has credit card:y
Is active member:y
Estimated Salary:50000
"""

new_predict=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_predict=(new_predict>0.5)

# part-4 Evaluating, Improving and tuning ANN

#part 4- 1: Applying K-fold Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies= cross_val_predict(estimator=classifier,X=x_train,y=y_train, cv=10,n_jobs=3)

mean=accuracies.mean()
variance=accuracies.std

#Improving Ann
#Part 4-2 Avoid overfitting using Dropout reguarization

# Tuning Ann using parameter tunning (Grid Search)
#Part 4-3 Performance Tunning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV # for grid search
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):# optimizer is passed as argumnet  for tunning optimizer in line 219
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)


parameters = {'batch_size':[25,32],
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

grid_search=grid_search.fit(x_train,y_train)  # fitting to training set
best_parameters=grid_search.best_params_
best_accracy=grid_search.best_score_