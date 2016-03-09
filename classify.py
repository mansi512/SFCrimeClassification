import pandas as pd
import numpy as np
import math
#import zipfile
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from datetime import datetime

def parse_time(x):
    DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    time=DD.hour#*60+DD.minute
    day=DD.day
    month=DD.month
    year=DD.year
    return time,day,month,year

def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

def readData():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print("Number of cases in the training set: %s" % len(train))
    print("Number of cases in the testing set: %s" % len(test))
    return (train,test)

def sliceByCategory(categories, train):
    trainWithCategories = train.loc[train['Category'].isin(categories)]
    return trainWithCategories

def applyFunction(train, inputCol, check, outputCol):
    train[outputCol] = train[inputCol].apply(lambda x:1 if x == check else 0 )
    return train

def convertToFeatures(train):
    train = applyFunction(train, 'DayOfWeek',"Sunday", "sun")
    train = applyFunction(train, 'DayOfWeek',"Monday", "mon")
    train = applyFunction(train, 'DayOfWeek', "Tuesday", "tues")
    train = applyFunction(train, 'DayOfWeek',"Wednesday", "wed")
    train = applyFunction(train, 'DayOfWeek',"Thursday", "thur")
    train = applyFunction(train, 'DayOfWeek',"Friday", "fri")
    train = applyFunction(train, 'DayOfWeek',"Saturday", "sat")
    train = applyFunction(train, 'PdDistrict',"BAYVIEW", "BAYVIEW")
    train = applyFunction(train, 'PdDistrict',"CENTRAL", "CENTRAL")
    train = applyFunction(train, 'PdDistrict',"INGLESIDE", "INGLESIDE")
    train = applyFunction(train, 'PdDistrict',"MISSION", "MISSION")
    train = applyFunction(train, 'PdDistrict',"NORTHERN", "NORTHERN")
    train = applyFunction(train, 'PdDistrict',"PARK", "PARK")
    train = applyFunction(train, 'PdDistrict',"RICHMOND", "RICHMOND")
    train = applyFunction(train, 'PdDistrict',"SOUTHERN", "SOUTHERN")
    train = applyFunction(train, 'PdDistrict',"TARAVAL", "TARAVAL")
    train = applyFunction(train, 'PdDistrict',"TENDERLOIN", "TENDERLOIN")
    for i in range(0,len(train),1):
        [time,day,month,year] = parse_time(train.loc[i]['Dates'])
        train.loc[i]['time'] = time
        train.loc[i]['year'] = year
        train.loc[i]['month'] = month
        train.loc[i]['day'] = day
    return train

def divideIntoTrainAndEvaluationSet(fraction, train):
    msk = np.random.rand(len(train)) < fraction
    trainOnly = train[msk]
    evaluateOnly = train[~msk]
    print("Number of cases in the training only set: %s" % len(trainOnly))
    print("Number of cases in the evaluation  set: %s" % len(evaluateOnly))
    return(trainOnly,evaluateOnly)

def classify(name, train, evaluate, test,all_categories):
    if(name == "knn"):
        return knnClassifier(train, evaluate, test)
    elif(name =="svm"):
        return svmClassifier(train, evaluate, test)
    elif(name == "logit"):
        return logisticRegressionClassifier(train, evaluate, test, all_categories)
    elif(name == "dtrees"):
        return dtreesClassifier(train, evaluate, test)
    else:
        print(" Specify the right name of the classifier : knn/svm/logit/dtrees")


def knnClassifier(train, evaluate, test):
    print('In K nerarest neighbour')
    x = train[['X', 'Y']]
    y = train['Category'].astype('category')
    actual = evaluate['Category'].astype('category')

    # Fit
    logloss = []
    for i in range(1, 50, 1):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x, y)

        # Predict on test set
        outcome = knn.predict(evaluate[['X', 'Y']])

        # Logloss
        logloss.append(llfun(actual, outcome))

    plt.plot(logloss)
    plt.savefig('n_neighbors_vs_logloss.png')

    # Fit test data
    x_test = test[['X', 'Y']]
    knn = KNeighborsClassifier(n_neighbors=40)
    knn.fit(x, y)
    outcomes = knn.predict(x_test)
    return outcomes
# Move this to a separate function later
'''
    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in y.cat.categories:
        submit[category] = np.where(outcomes == category, 1, 0)

    submit.to_csv('k_nearest_neigbour.csv', index = False)
'''

def svmClassifier(train, test):
    print('In SVM')

def logisticRegressionClassifier(train,evaluate,test,all_categories):
    print('In Logistic Regression')
    x_train = train[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters']]
    x_eval =  evaluate[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters']]
    x_test = test[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters']]
    y_train = train['Category'].astype('category')
    y_eval = evaluate['Category'].astype('category')

    #logloss = []
    #minC = 0
    #minL = 100
    c = [0.05]
    #,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    #for cc in c:
    #c = [0.05,0.1]
    for cc in c:
        print(cc)
        logreg = linear_model.LogisticRegression(C=cc,multi_class='multinomial',solver='lbfgs')
        logreg.fit(x_train, y_train)
        outcome = logreg.predict(x_eval)
        l = llfun(y_eval, outcome)
        #logloss.append(l)
        #if l<minL:
        #    minL = l
        #    minC = cc
        print('c: ',cc,'loss: ',l)

    outcomes = logreg.predict(x_test)
    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in all_categories:
        submit[category] = np.where(outcomes == category, 1, 0)
    submit.to_csv('logit.csv', index = False)
    return outcomes

    #print('Min C: %s',minC)

    #plt.plot(logloss)
    #plt.savefig('logit_logloss_vs_C.png')

    #test data
    #logreg = linear_model.LogisticRegression(C=0.05,multi_class='multinomial',solver='lbfgs')

def dtreesClassifier(train, test):
    print('In decision trees')

#def createSubmissionFile(lables, fileName):

def kMeansClustering(train,evaluate,test):
    km = KMeans(n_clusters=40)
    f_train = km.fit_predict(train[['X','Y']])
    f_eval = km.predict(evaluate[['X','Y']])
    f_test = km.predict(test[['X','Y']])
    print(km.cluster_centers_)
    cf_train=[]
    for i in range(0,len(f_train)-1,1):
        cf_train.append(km.cluster_centers_[f_train[i]])
    print(f_train)
    print(cf_train)
    print(f_eval)
    print(f_test)
    return (f_train,f_eval,f_test)


def main():
   (train, test) = readData()

   train = convertToFeatures(train)
   test = convertToFeatures(test)

   print(train.columns.values)
   print(train[1111:1136])
   all_categories = pd.Series(train.Category.values).unique()
   print(all_categories)

   categories = ["LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL","ASSAULT", "DRUG/NARCOTIC"]
   trainWithTopCategories = sliceByCategory(categories, train)

   (trainOnly,evaluateOnly) = divideIntoTrainAndEvaluationSet(0.8, trainWithTopCategories)
   (f_train,f_eval,f_test)=kMeansClustering(trainOnly,evaluateOnly,test)

   trainOnly["n_clusters"]=f_train
   evaluateOnly["n_clusters"]=f_eval
   test["n_clusters"]=f_test

   # Call the classifiers - replace with your classifier
   #predictedLabels = classify("knn",trainOnly, evaluateOnly, test)
   #print(predictedLabels)

   #predictedLabels = classify("logit",trainOnly,evaluateOnly,test,all_categories)
   #print(predictedLabels)

main()