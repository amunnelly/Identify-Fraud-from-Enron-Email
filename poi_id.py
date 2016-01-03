#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

all_features = ['deferral_payments',
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'restricted_stock_deferred',
                      'deferred_income',
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      'director_fees',
                      'to_messages',
                      'from_poi_to_this_person',
                      'from_messages',
                      'from_this_person_to_poi',
                      'shared_receipt_with_poi']



#[features_list.append(a) for a in all_features]

for a in all_features:
    counter = 0
    for key, value in data_dict.iteritems():
        temp = value.get(a)
        if value.get(a) != 'NaN':
            counter += 1
        else:
            continue
    if counter > 100: # To set the number of features added
        features_list.append(a)
    else:
        continue


### Task 2: Remove outliers

# A helper data frame
initial_df = pd.DataFrame(data_dict)
enron = initial_df.T # making the features the keys to the data frame

def color_checker(color):
    '''
    (bool) -> (str)
    A helper function to identify poi candidates in plots.
    '''
    if color == True:
        return 'red'
    else:
        return 'green'
        
enron['poi_color'] = enron['poi'].map(lambda x: color_checker(x))


# Look at some plots to get a feel for the data

def create_basic_plots():
    plt.plot(enron.total_stock_value,marker = '*', linewidth = 0)
    plt.plot(enron.total_payments,marker = '*', linewidth = 0)
    plt.plot(enron.restricted_stock, marker = '*', linewidth = 0)
    plt.plot(enron.exercised_stock_options, marker = '*', linewidth = 0)
    plt.plot(enron.salary, marker = '*', linewidth = 0)
    plt.plot(enron.expenses, marker = '*', linewidth = 0)
    plt.plot(enron.other, marker = '*', linewidth = 0)
    plt.plot(enron.bonus, marker = '*', linewidth = 0)
    plt.plot(enron.long_term_incentive, marker = '*', linewidth = 0)
    plt.plot(enron.deferred_income, marker = '*', linewidth = 0)
    plt.plot(enron.director_fees, marker = '*', linewidth = 0)
    enron[enron.loan_advances != 'NaN']

    return
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.



def normalise(data):
    '''
    A helper function from the first project in the course.
    '''
    mean = data.mean()
    sd = data.std()
    return (data-mean)/sd


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV
my_dtc_parameters = {'criterion':['gini', 'entropy'],
                 'splitter':['best', 'random'],
                'max_features':['auto', 'sqrt', 'log2']}
                
my_svc_parameters = {'kernel':['rbf', 'linear'],
                     'C':[1,2]}
                
my_rfc_parameters = {'criterion':['gini', 'entropy'],
                'max_features':['auto', 'sqrt', 'log2']}



from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA
#clf = make_pipeline(PCA(), RandomForestClassifier(max_depth = None, min_samples_split=1))

#dtc = GridSearchCV(RandomForestClassifier(), my_dtc_parameters)
#svm = GridSearchCV(SVC(), my_svc_parameters)
#clf = make_pipeline(PCA(), dtc)
rfc = GridSearchCV(RandomForestClassifier(), my_rfc_parameters)
clf = make_pipeline(PCA(), rfc)
#clf = make_pipeline(PCA(), RandomForestClassifier())


# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.cross_validation import KFold
kf = KFold(len(data), n_folds = 2, shuffle=True)
for train, test in kf:
    features_train = [features[t] for t in train]
    features_test = [features[t] for t in test]
    labels_train = [labels[t] for t in train]
    labels_test = [labels[t] for t in train]
    

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print "We're using {} features.".format(len(features_list))