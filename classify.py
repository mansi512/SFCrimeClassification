import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib


def parse_time(x,type):
    DD=datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    if(type == "time"):
        return DD.hour#*60+DD.minute
    if(type == "day"):
        return DD.day
    if(type == "month"):
        return DD.month
    if(type == "year"):
        return DD.year

def llfun(act, pred):
    """ Logloss function for 1/0 probability
    """
    return (-(~(act == pred)).astype(int) * math.log(1e-15)).sum() / len(act)

def readData():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    print "Number of cases in the training set: %s" % len(train)
    print "Number of cases in the testing set: %s" % len(test)
    return (train,test)

def sliceByCategory(categories, train):
    trainWithCategories = train.loc[train['Category'].isin(categories)]
    return trainWithCategories

def applyFunction(train, inputCol, check, outputCol):
    train[outputCol] = train[inputCol].apply(lambda x:1 if x == check else 0 )
    return train

def applyDateFunction(train):
    train['time'] = train['Dates'].apply(lambda x:parse_time(x,"time"))
    train['day'] = train['Dates'].apply(lambda x:parse_time(x,"day"))
    train['month'] = train['Dates'].apply(lambda x:parse_time(x,"month"))
    train['year'] = train['Dates'].apply(lambda x:parse_time(x,"year"))
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
    train = applyDateFunction(train)
    # for i in range(0,len(train),1):
    #     [time,day,month,year] = parse_time(train.loc[i]['Dates'])
    #     train.loc[i]['time'] = time
    #     train.loc[i]['year'] = year
    #     train.loc[i]['month'] = month
    #     train.loc[i]['day'] = day
    return train

def divideIntoTrainAndEvaluationSet(fraction, train):
    msk = np.random.rand(len(train)) < fraction
    trainOnly = train[msk]
    evaluateOnly = train[~msk]
    print "Number of cases in the training only set: %s" % len(trainOnly)
    print "Number of cases in the evaluation  set: %s" % len(evaluateOnly)
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
        print " Specify the right name of the classifier : knn/svm/logit/dtrees"

def logisticRegressionClassifier(train,evaluate,test,all_categories):
    print('In Logistic Regression')
    x_train = train[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters','time','day','month','year']]
    x_eval =  evaluate[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters','time','day','month','year']]
    x_test = test[['sun','mon','tues','wed','thur','fri','sat','BAYVIEW',
 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN','n_clusters','time','day','month','year']]
    y_train = train['Category'].astype('category')
    y_eval = evaluate['Category'].astype('category')

    c = [0.05]
    for cc in c:
        print cc
        logreg = linear_model.LogisticRegression(C=cc,multi_class='ovr')
        logreg.fit(x_train, y_train)
        outcome = logreg.predict(x_eval)
        l = llfun(y_eval, outcome)
        print 'c: ',cc,'loss: ',l

    outcomes = logreg.predict(x_test)
    submit = pd.DataFrame({'Id': test.Id.tolist()})
    for category in all_categories:
        submit[category] = np.where(outcomes == category, 1, 0)
    submit.to_csv('logit.csv', index = False)
    return outcomes

def kMeansClustering(train,evaluate,test):
    km = KMeans(n_clusters=40)
    f_train = km.fit_predict(train[['X','Y']])
    f_eval = km.predict(evaluate[['X','Y']])
    f_test = km.predict(test[['X','Y']])
    print km.cluster_centers_
    print f_train
    print f_eval
    print f_test
    return (f_train,f_eval,f_test)

def plots(train,categories):
    matplotlib.style.use('ggplot')
    temp = pd.crosstab([train.Category],train.PdDistrict)
    temp.plot(kind='barh')
    temp = pd.crosstab([train.Category],train.DayOfWeek)
    temp.plot(kind='barh')
    temp = pd.crosstab([train.Category],train.time)
    temp.plot(kind='barh')
    temp = pd.crosstab([train.loc[train['Category'].isin(categories),'Category']],train.time)
    temp.plot(kind='barh')
    train.time.value_counts().plot(kind='barh')
    train.DayOfWeek.value_counts().plot(kind='barh')
    train.PdDistrict.value_counts().plot(kind='barh')
    train.Category.value_counts().plot(kind='barh')
    matplotlib.pyplot.show()
    #[train.loc[train['Category'].isin(categories),'Category']]

def main():
   (train, test) = readData()
   train = convertToFeatures(train)
   test = convertToFeatures(test)
   
   all_categories = pd.Series(train.Category.values).unique()
   # print all_categories

   categories = ["LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL","ASSAULT", "DRUG/NARCOTIC"]
   plots(train,categories)
   
   trainWithTopCategories = sliceByCategory(categories, train)
   
   (trainOnly,evaluateOnly) = divideIntoTrainAndEvaluationSet(0.8, trainWithTopCategories)
   (f_train,f_eval,f_test)=kMeansClustering(trainOnly,evaluateOnly,test)
   
   trainOnly["n_clusters"]=f_train
   evaluateOnly["n_clusters"]=f_eval
   test["n_clusters"]=f_test

   # Call the classifiers - replace with your classifier
   
   predictedLabels = classify("logit",trainOnly,evaluateOnly,test,all_categories)
   print(predictedLabels)

   
main()