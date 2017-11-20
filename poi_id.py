#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
target_label = 'poi'

features = [#"deferral_payments",
            #"deferred_income",
            #"director_fees",
            "exercised_stock_options",
            #"expenses",
            #"from_messages",
            #"from_poi_to_this_person",
            #"from_this_person_to_poi",
            #"long_term_incentive",
            #"loan_advances",
            #"other",
            "bonus",
            #"restricted_stock",
            #"restricted_stock_deferred",
            #"salary",
            #"shared_receipt_with_poi",
            #"to_messages",
            #"total_payments",
            "total_stock_value"
            ]

features_list = [target_label] + features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# How many people in the Enron dataset?
print len(data_dict)
# 146

# How many features for each person in the Enron dataset?
print len(data_dict.values()[0]) - 1 #POI 
# 20

# Person of Interest Count
poi_count = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi_count+=1
print poi_count
# 18

# View keys from Enron data
data_dict.keys()

### Task 2: Remove outliers
# Remove outlier 'TOTAL' from previous lesson as well as 'THE TRAVEL AGENCY
# IN THE PARK' found from scanning the dictionary keys
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

# Remove email_address from features_list for being irrelevant

# View counts of empty fields for features
for feature in features_list:
    count = 0
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            count += 1
    print feature + ' = ' + str(count) + ' NaN values'


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Create fraction_to_POI and fraction_from_POI features
for key in my_dataset:
    person = my_dataset[key]
    from_msgs = person['from_messages']
    to_msgs = person['to_messages']
    msgs_from_poi = person['from_poi_to_this_person']
    msgs_to_poi = person['from_this_person_to_poi']
    if from_msgs != 'NaN' and msgs_to_poi != 'NaN':
        person['fraction_to_poi'] = msgs_to_poi / float(from_msgs)
    else:
        person['fraction_to_poi'] = 0
    if to_msgs != 'NaN' and msgs_from_poi != 'NaN':
        person['fraction_from_poi'] = msgs_from_poi / float(to_msgs)
    else:
        person['fraction_from_poi'] = 0
    


# Create new features list that includes two new features
# NOTE: IF FOLLOWING 2 LINES COMMENTED OUT, IT'S TO EXCLUDE THEM FROM
# FEATURES LIST FOR FINAL ALGORITHM TUNE

# features_list += ['fraction_to_poi']
# features_list += ['fraction_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# SelectKBest to determine feature selection
from sklearn.feature_selection import SelectKBest
k_best = SelectKBest(k='all')
k_best.fit_transform(features, labels)
indices = k_best.get_support

# Next 3 lines of code utilized from Udacity forums
# The following code gets the kBest score (2 decimals), then creates a tuple of 
# feature names and scores so that the score can be associated with the 
# feature representing it. The tuple is then sorted by highest scoring features.
feature_scores = ['%.2f' % elem for elem in k_best.scores_ ]
features_selected_tuple=[(features_list[i+1], feature_scores[i]) for i 
                         in k_best.get_support(indices=True)]
features_selected_tuple = sorted(features_selected_tuple, key=lambda 
                                 feature: float(feature[1]) , reverse=True)

# Print top scoring features with kBest scores
print features_selected_tuple

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Import Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Choose and test different Classifiers
# clf = GaussianNB()
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=9,
                             max_features=1, max_leaf_nodes=None,
                             min_impurity_split=1e-07, min_samples_leaf=1,
                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                             presort=False, random_state=None, splitter='best')



# Utilize tester.py test_classifier for accurate validation
from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

### Used the following code initially, however the results from utilizing
### test/train split were too unstable due to a small number of training/testing
### data, therefor I opted to utilize the tester.py function which uses
### StratifiedShuffleSplit instead of train/test for better results.

# Train
# clf.fit(features_train, labels_train)

# Predict
# pred = clf.predict(features_test)

# Precision and Recall
# print precision_score(labels_test, pred)
# print recall_score(labels_test, pred)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
