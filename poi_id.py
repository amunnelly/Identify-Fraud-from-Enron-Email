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


def add_to_feature_list(all_features, features_list, feature_count):
    '''
    (list, list, int) -> list
    The function iterates through each item in all_features, and counts
    the number of legitimate (that is, not 'NaN') values that exist in
    the data_dict corresponding to the list item. If the legimate feature
    count is higher than feature_count, that feature is added to the
    features_list returned by the function.
    '''
    for a in all_features:
        counter = 0
        for key, value in data_dict.iteritems():
            if value.get(a) != 'NaN':
                counter += 1
            else:
                continue
        if counter > feature_count: # To set the number of features added
            features_list.append(a)
        else:
            continue
        
    return features_list
    
features_list = add_to_feature_list(all_features, features_list, 100)

### Task 2: Remove outliers

# A helper data frame
initial_df = pd.DataFrame(data_dict)
enron = initial_df.T # making the features the keys to the data frame

'''
A quick scan through the names and the figures reveals
an obvious outlier.
'''

enron.drop('TOTAL', inplace=True) # OUTLIER REMOVED

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

def create_basic_plots(df):
    myColumns = list(df.columns)
    df = normalise(df)
    for m in myColumns:
        temp = df.get(m)
        plt.plot(temp.values, marker = '*', linewidth = 0, label = m)
#        plt.legend()

    return
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

'''
3A. Create a second df, enron2, where the column dtypes
    are converted to floats where possible.
'''

myColumns = list(enron.columns)
holding_dict = {}
for col in myColumns:
    temp = enron.get(col)
    try:
        holding_dict[col] = temp.astype(float)
    except:
        holding_dict[col] = temp
    
enron2 = pd.DataFrame(holding_dict)

'''
3B. Create a third df, enronN, where the float values
    are normalised relative to each other
    
'''

def normalise(data):
    '''
    A helper function from the first project in the course.
    '''
    mean = data.mean()
    sd = data.std()
    return (data-mean)/sd
    

holding_dict = {}
myColumns_norm = myColumns
myColumns_norm.remove('poi')
for col in myColumns:
    temp = enron.get(col)
    
    try:
        newCol = temp.astype(float)
        holding_dict[col] = normalise(newCol)
    except:
        holding_dict[col] = temp


enronN = pd.DataFrame(holding_dict)
enronN['poi'] = enron['poi'] # Restore the poi column post-normalisation
enronN_corr = enronN.corr()
#enronN.to_csv('enronN_corr.csv')
#
#enron2.corr().to_csv("enron_corr.csv")

'''
3C. As the columns related to stocks seem to correlate, we need to create
    a new column, stocks, to represent these values. We'll create two
    functions to create this new column - one by regression, and one by
    Principal Component Analysis.
'''

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold


reg = LinearRegression()
pca = PCA()

def create_new_feature_regression(df):
    stocks = df[['total_stock_value', 
                     'exercised_stock_options']]
    stocks.dropna(inplace=True)
    '''
    3C-Regression. Create test and training data from the stocks df,
            and use that to create a new variable, stock_data
    '''
    
    
    kf = KFold(100, n_folds = 2, shuffle=True)
    for train, test in kf:
        total_train = [stocks.ix[t]['total_stock_value'] for t in train]
        total_test = [stocks.ix[t]['total_stock_value'] for t in test]
        options_train = [stocks.ix[t]['exercised_stock_options'] for t in train]
        options_test = [stocks.ix[t]['exercised_stock_options'] for t in test]
    
    reg.fit(np.array(total_train).reshape(50,1),
            np.array(options_train).reshape(50,1))
            
    stock_data = [reg.predict(x)[0][0] for x in stocks.total_stock_value]
    stocks['stock_data'] = stock_data
    stock_data = pd.DataFrame(stocks['stock_data'], columns = ['stock_data'])
    
    return stock_data
    
def create_new_feature_pca(df):
    stocks = df[['total_stock_value', 
                     'exercised_stock_options']]
    stocks.dropna(inplace=True)

    stock_data = pca.fit_transform(
                np.array(stocks.total_stock_value).reshape(100,1),
                np.array(stocks.exercised_stock_options).reshape(100,1))
    stocks['stock_data'] = stock_data
    stock_data = pd.DataFrame(stocks['stock_data'], columns = ['stock_data'])
    
    return stock_data

'''
3D. Create the stock_data column and then add it to either the
    enronN normalised data frame or the enron2 data frame as
    appropriate
'''
stock_data = create_new_feature_regression(enronN)
enron_final = enronN.join(stock_data)


'''
3E. Transform enron_final to a dictionary, marking sure to change NaN/nan
    values to 'NaN' strings - the feature_format() function will throw an error
    otherwise.
'''

enron_final.fillna(value = 'NaN', inplace = True)

my_dataset = enron_final.T.to_dict()

features_list = ['poi',
                 'salary',
                 'total_payments',
                 'stock_data']

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



#clf = GaussianNB()
#clf = make_pipeline(PCA(), GaussianNB())
#clf = SVC()
#clf = AdaBoostClassifier()
clf = DecisionTreeClassifier()
#clf = make_pipeline(PCA(), DecisionTreeClassifier())
#clf = GridSearchCV(DecisionTreeClassifier(), my_dtc_parameters)

#clf = make_pipeline(PCA(), RandomForestClassifier(max_depth = None, min_samples_split=1))

#clf = RandomForestClassifier()
#dtc = GridSearchCV(RandomForestClassifier(), my_dtc_parameters)
#svm = GridSearchCV(SVC(), my_svc_parameters)
#clf = make_pipeline(PCA(), dtc)
#rfc = GridSearchCV(RandomForestClassifier(), my_rfc_parameters)
#clf = make_pipeline(PCA(), rfc)
#clf = make_pipeline(PCA(), RandomForestClassifier())


# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

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