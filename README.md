#Introduction

The ENRON Scandal is considered to be one of the most notorious within American history; an ENRON scandal summary of events is considered by many historians and economists alike to have been an unofficial blueprint for a case study on White Collar Crime. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for to executives.
With the help of scikit-learn and machine learning methodologies, I built a "person of interest" (POI) identifier to find out the people who were responsible for the fraud.

#Short Questions

>1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

The goal of this project was to use the provided Enron’s financial and email data and build a predictive model with the help of machine learning algorithms to identify the culpable persons of interest(poi). 
The provided dataset contains labelled data, which shows the responsible person for the fraud as POIs. Apart from one labelled ‘poi’ feature, this dataset comprises 14 Financial features and 6 Email features:
  *Financial features: salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees.
  *Email Features: to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi, shared_receipt_with_poi
A total of 146 records were present in the dataset. Out of these records, I found 3 records to be the outliers for our analysis. This was done by plotting the graph between salary and bonus and finding the outliers in the scatter plot. Apart from that, by using simple excel tools, I found out few outliers as well. All those 3 outliers with their reason for removal are given below: 
TOTAL:  This is the sum of all values in the features.
THE TRAVEL AGENCY IN THE PARK: This record represents an agency itself and not an individual.
LOCKHART EUGENE E: This record contained no useful data. 

Apart from I found out three other records as well whose financial info found out to be way beyond than other records. Those records were of:
  * SKILLING JEFFREY K
  * LAY KENNETH L
  * FREVERT MARK A

But there is no need to remove these 3 outliers found in above step as they could be of crucial importance in our machine learning algorithms later.


>2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that doesn't come ready-made in the dataset--explain what feature you tried to make, and the rationale behind it. If you used an algorithm like a decision tree, please also give the feature importances of the features that you use.
To obtain the best features from our dataset for the classifier, I used scikit-learn's SelectKBest module to select the 8 most influential features from the dataset apart from ‘poi’ feature. Their associated scores and total respective records are given in below table:

|Feature	|Score	|Valid	|
|---------------|-------|-------|
|shared_receipt_with_poi	|8.9038215571655712	|86	|
|from_poi_to_this_person	|5.4466874833253529	|86	|
|from_this_person_to_poi	|2.470521222656084	|86	|
|to_messages	|1.7516942790340737	|86	|
|director_fees	|0.54908420147980874	|16	|
|total_payments |0.34962715304280179	|122	|
|deferral_payments	|0.23899588985313305	|38	|
|exercised_stock_options	|0.22826733729104948	|100	|

Out of these 8 best features, top 4 are from email features list and bottom 4 are from financial features list. Two financial features, namely ‘director_fees’ and ‘deferral_payments’ have only 16 and 38 valid records respectively. Still their rankings in top 8 is somewhat surprising.
As I saw that the financial features in my top 8 list had less scores against them, I decided to create two new features, namely ‘total_to_salary’ and ‘expenses_to_salary’. The feature ‘total_to_salary’ is equal to the ratio of total payments to the salary of a particular record. And the feature ‘expenses_to_salary’ is equal to the ratio of expenses made by a particular record to their respective salary. After inclusion of these two features along with our other 8 best features and ‘poi’feature, I have a total of 11 features for the machine learning algorithms used later in this project.
I scaled all features using a min-max scaler before putting them to train for machine learning algorithms. This was vitally important, as the features had different units (e.g. # of email messages and USD) and varied significantly by several orders of magnitude. Feature-scaling ensured that for the applicable classifiers, the features would be weighted evenly.

>3. What algorithm did you end up using? What other one(s) did you try?
Before putting features in the classifiers for machine learning, I used KFold cross-validation iterator to split the data into 3 folds of test/train sets. Now while trying machine learning algorithms, two algorithms performed better than the rest of them. These two algorithms are:
  * Logistic Regression Classifier
  * Decision Tree Classifier

For a better tuning of the parameters of the logistic regression and Decision Tree classifiers using exhaustive searches with the following parameters:
Logistic regression: C (inverse regularization parameter), tol (tolerance), and class_weight (Weights associated with classes), max_iter (Maximum number of iterations taken for the solvers to converge).
Decision Tree Classifier: max_leaf_nodes, min_samples_split, class_weight, 
The other algorithms were tuned experimentally, with unremarkable improvement. I have commented Logistic regression algorithm in the submitted poid.py script. In order to compare both Decision Tree and Logistic regression, uncomment the Logistic regression part in the script and comment the Decision Tree Classifier algorithm during tuning the parameter section.


>4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? 
Tuning an algorithm or machine learning technique, can be simply thought of as process which one goes through in which they optimize the parameters that impact the model in order to enable the algorithm to perform the best. If the tuning of an algorithm is done using inappropriate parameters or if the value of those parameters are not set as per the dataset we are using, it might lead into wrong results or might throw some error in the methodolgy used.
To tune the parameters in the two algorithms I used for this project, I used follwing values:
  *Before Tuning:
1. Logistic Regression: (C=100000000000000000000L, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=1e-20, verbose=0, warm_start=False)
2. Decision Tree Classifier: (class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
  
  *After Tuning:
1. Logistic Regression: (C=10**27, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, max_iter=150, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=10**-30, verbose=0, warm_start=False)
2. Decision Tree Classifier: (class_weight='balanced', criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=18, min_samples_leaf=1, min_samples_split=11, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')

>5. What is validation, and what's a classic mistake you can make if you do it wrong? How did you validate your analysis?
Validation is performed to ensure that a machine learning algorithm generalizes well. A classic mistake is over-fitting, where the model is trained and performs very well on the training dataset, but markedly worse on the cross-validation and test datasets. I utilized ‘train_test_split’ method for validating the analysis. Prior to my final validation, I used KFold validation technique as well with number of folds equal to 3. Now in final validation before tuned classifier, I used test size as 0.1. So apart from test_train_split, I used another validation methodology called StratifiedShuffleSplit. While using it, I was confused whether to use StratifiedShuffleSplit or StratifiedKFold methodology. But I chose the earlier one because StratifiedShuffleSplit is a merge of StratifiedKFold and ShuffleSplit (where samples are first shuffled and then split into a pair of train and test sets), which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class. Now I used a total of 1000 folds in my StratifiedShuffleSplit method. The main reason for using StratifiedShuffleSplit and not StratifiedKFold was the size of our dataset. As our dataset was not big enough and only few POIs were flagged in them. Therefore instead of normal K-Folds, a Shuffle split was more favourable.

>6. Give at least 2 evaluation metrics, and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm's performance.
The 2 evaluation metrics I used in the project are Precision and Recall. Precision (also called positive predictive value) is the fraction of retrieved instances that are relevant, while Recall (also known as sensitivity) is the fraction of relevant instances that are retrieved. Mathematically, precision is the ratio of true positives to the sum of both true and false positives. And recall is mathematically equal to the ratio of true positives to the sum of true positives and false negatives in the dataset. For this project, precision will give us the info on how exact are we in flagging our POIs for fraud. Similarly, recall will tell us more about how many of suspect individuals have we covered in our dataset. The evaluation metrics gave following results for logistic regression and Decision Tree classifier algorithms:
  *Validation 1:	cross_validation.train_test_split(test_size=0.1, random_state=50)

|Classifier	|Precision(pre tester.py)	|Precision(post tester.py)	|Recall(pre tester.py)	|Recall(post tester.py)	|Features	|
|-----------|-----------|-------|-----------|
|Logistic Regression	|0.923	|0.301	|0.857	|0.599	|11	|
|Decision Tree Classifier	|0.917	|0.303	|0.786	|0.443	|11	|

  *Validation 2:	cross_validation. StratifiedShuffleSplit(n_iter = 1000, random_state = 42)

|Classifier	|Precision(pre tester.py)	|Precision(post tester.py)	|Recall(pre tester.py)	|Recall(post tester.py)	|Features	|
|-----------|-----------|-------|-----------|
|Logistic Regression	|0.917	|0.325	|0.846	|0.640	|11	|
|Decision Tree Classifier	|0.923	|0.377	|0.923	|0.596	|11|

As we can see that there is a drastic difference between the values of precision and recall in the pre-tester.py and post-tester.py columns. The most probable reason for this could be the dataset, which tester.py is using as input, has already been run on a machine learning algorithm previously in poid.py script. In our dataset, POIs refer to those individuals who are suspect of doing fraud. As the count of flagged POIs is less in our dataset, it is important that the suspect individuals should be included rather than innocent individuals for the machine learning algorithm. A high recall value would ensure that truly culpable individuals were flagged as POIs and would be investigated more thoroughly. To improve the recall value, I have used the class_weight parameter in both classifiers with value as 'balanced'.
Personally, I believe that a higher recall value is better than higher precision value as our dataset has not much records in it. And it is necessary to include as much as possible culprit individuals in the flagged list. We can see that before running tester.py our precision came out to be higher but after running tester.py script our recall value has gone up considerably. Although there is a reduction in the precision value, but as mentioned earlier this could be attributed to the prepared dataset from our poid.py script. 

###References:
1. http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
2. http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
3. http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
4. https://github.com/allanbreyes/udacity-data-science.git
5. https://en.wikipedia.org/wiki/Precision_and_recall