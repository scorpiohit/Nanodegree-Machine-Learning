#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit

### Task 1: Select what features you'll use.
###         features_list is a list of strings, each of which is a feature name.
###         The first feature must be "poi".
###         load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
print ' Total number of records:    ', len(data_dict.keys())

i=0
list_1 = []
for value in data_dict.values():
    list_1.append(data_dict.values()[i].keys())
    i+=1    

result = set(x for l in list_1 for x in l)
print '\n Features for each record:    ', result
print '\n Count of features for every record:    ', len(result)

features_list_base = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'exercised_stock_options',
                     'expenses',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages']

def Best_Features(data_dict, features_list, k_value):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k_value)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = sorted(unsorted_pairs, key=lambda x: x[1],reverse = True)[:k_value]
    k_best_features = dict(sorted_pairs[:k_value])
    print "\n {0} best features:    {1}".format(k_value, sorted_pairs)
    return k_best_features

k_best = Best_Features(data_dict, features_list_base, 8)

features_list = ['poi']
features_list += k_best.keys()
new_features = ["total_to_salary", "expenses_to_salary"]
features_list += new_features



### Task 2: Remove outliers
###         To remove ouliers from dataset, first we will reomve NANs from dataset and then top outlier
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:3])
print '\n The possible outliers in dataset are:    ', outliers_final

#### No need to remove other outliers found in above step as they could be of crucial importance in our machine learning algorithms later.
#data_dict.pop(outliers_final[0][0], 0)
#data_dict.pop(outliers_final[1][0], 0)
#data_dict.pop(outliers_final[2][0], 0)

### plot features
data1 = featureFormat(data_dict, features)
for point in data1:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


### Task 3: Create new feature(s)

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

#fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
#fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
total_to_salary=dict_to_list("total_payments","salary")
expenses_to_salary=dict_to_list("expenses","salary")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["total_to_salary"]=total_to_salary[count]
    data_dict[i]["expenses_to_salary"]=expenses_to_salary[count]
    count +=1

### Store to my_dataset for easy export below.
my_dataset = data_dict


def count_valid_values(data_dict):
    """ counts the number of non-NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] != 'NaN':
                counts[field] += 1
    return counts

valid_features = count_valid_values(my_dataset)
print '\n Count of valid(non-NAN) records for each feature:    ', valid_features

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list,sort_keys = True)
labels, features = targetFeatureSplit(data)

# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Try a variety of classifiers.
### use KFold for split and validate algorithm


from sklearn import cross_validation
kf=cross_validation.KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]



t0 = time()

### K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm='SAMME.R')

### Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

### Stochastic Gradient Descent - Logistic Regression
from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
d_clf = DecisionTreeClassifier()

### Naive Bayes GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()

### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
l_clf = LogisticRegression(C=10**20, solver='liblinear', tol=10**-20)


l_clf.fit(features_train,labels_train)
pred = l_clf.predict(features_test)
score = l_clf.score(features_test,labels_test)

print '\n Accuracy score before tuning =    ', score
#print 'Precision Score = ', precision_score(labels_test,pred,average = 'binary')
print " Decision tree algorithm time =    ", round(time()-t0, 5), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Uncomment validation 1 and validation 2 one at a time to get the individual results


#### Validation 1: train_test_split ####
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=50)


#### Validation 2: StratifiedShuffleSplit ####
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )


### Two classifier have been tuned for better performance. Try to uncomment one classifier at a time to get individual result

#### Decision Tree Classifier: ####
#clf = DecisionTreeClassifier(class_weight='balanced', criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=18, min_samples_leaf=1,
#            min_samples_split=11, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')
#clf = clf.fit(features_train,labels_train)
#pred= clf.predict(features_test)


####Logistic Regression Classifier: #####
clf = LogisticRegression(C=10**27, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=150,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=10**-30, verbose=0, warm_start=False)
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)


accuracy=accuracy_score(labels_test, pred)

print "\n Validating algorithm:"
print " Accuracy score after tuning =    ", accuracy

# Precision score is the ratio of true positives to both true positives and false positives.
# Using precision_score function to calculate precision score. 
print ' Precision Score =    ', precision_score(labels_test,pred,pos_label = 0,average ='binary')

# Recall score is the ratio of true positives to true positives and false negatives
# Using recall_score function to calculate recall score. 
print ' Recall Score =    ', recall_score(labels_test,pred,pos_label = 0,average ='binary')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )