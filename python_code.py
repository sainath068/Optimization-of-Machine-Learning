#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import mode
from sqlalchemy import create_engine
from urllib.parse import quote
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,accuracy_score,f1_score, r2_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import pickle
import sklearn.metrics as skmet


# In[6]:


# import the data

machine_downtime = pd.read_csv(r"C:\Users\Windows\project1\Machine_Downtime.csv")
machine_downtime


# In[7]:


# Credentials to connect to Database

user = 'root'
pw = 'sainath2'
db = 'machine_downtime'

engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote (f'{pw}'))


# In[8]:


# to_sql() - function to push the dataframe onto a SQL table.

machine_downtime.to_sql('machine_downtime', con = engine,if_exists = 'replace',chunksize = 1000 ,index = False )


# In[9]:


sql = 'select * from machine_downtime;'
df = pd.read_sql_query(sql, engine)


# In[7]:


# Data types
df.info()


# In[8]:


df.head()


# In[10]:


df.columns


# In[11]:


columns_to_drop = ['Date', 'Machine_ID', 'Assembly_Line_No']
df.drop(columns=columns_to_drop, inplace=True)



# In[12]:


df


# In[13]:


# Checking Null values if any 

null_values =  df.isnull().sum()
null_values


# In[14]:


df.describe()


# In[15]:


# Checking the results through the Sweetiz
s = sv.analyze(df)
s.show_html()


# In[16]:


import warnings

# Ignore warnings
warnings.filterwarnings("ignore")


# In[17]:


# Input and Output Split
predictors = df.loc[:, df.columns != "Downtime"]
type(predictors)

predictors




# In[18]:


target = df["Downtime"]
type(target)

target


# In[19]:


# Segregating Non-Numeric features

categorical_features = predictors.select_dtypes(include = ['object']).columns

categorical_features


# In[20]:


# Segregating Numeric features
numeric_features = predictors.select_dtypes(exclude = ['object']).columns

numeric_features


# In[ ]:





# # Data Preprocessing
# 

# In[21]:


## Missing values Analysis
# Checking for Null values
df.isnull().sum()

# Define pipeline for missing data if any
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

imputation = preprocessor.fit(predictors)

joblib.dump(imputation, 'meanimpute')




# In[22]:


cleandata = pd.DataFrame(imputation.transform(predictors), columns = numeric_features)
cleandata

cleandata.isnull().sum()


# # Measure Of Central Tendency

# In[23]:


# Select the columms for dispersion calculation

columns = ['Hydraulic_Pressure','Coolant_Pressure','Air_System_Pressure','Coolant_Temperature','Hydraulic_Oil_Temperature','Spindle_Bearing_Temperature','Spindle_Vibration','Tool_Vibration','Spindle_Speed','Voltage','Torque','Cutting']

for column in columns:
    values = cleandata[column]
    
    # Mean
    mean_value = np.mean(values)
    
    # Median
    median_value = np.median(values)
    
    # Mode
    mode_result = mode(values)
    mode_value = mode_result.mode[0] if len(mode_result.mode) > 0 else None
    
    print(f'{column}: Mean = {mean_value}, Median = {median_value}, Mode = {mode_value}')


# # Measure of Dispersion

# In[24]:


for column in columns:
    values = cleandata[column]
    mean = np.mean(values)
    variance = np.var(values, ddof=1)  # ddof=1 for sample variance
    std_dev = np.sqrt(variance)
    
    print(f'{column}: Variance = {variance}, Standard Deviation = {std_dev}')


# # Skewness

# In[25]:


Skewness = cleandata.skew(numeric_only = True)
Skewness


# # Kurtosis

# In[26]:


kurtosis_values = cleandata.kurtosis(numeric_only = True)
kurtosis_values


# In[ ]:





# # Outlier Detection

# In[23]:


cleandata.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# In[24]:


from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.columns))

clean = winsor.fit(cleandata)


# In[25]:


# Save winsorizer model
joblib.dump(clean, 'winsor')




# In[26]:


cleandata1 = clean.transform(cleandata)
cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# In[27]:


## Scaling with MinMaxScaler

from sklearn.preprocessing import MinMaxScaler


scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata1)



scaled_data = pd.DataFrame(scale.transform(cleandata1))
scaled_data



# In[28]:


joblib.dump(scale, 'minmax')


# In[29]:


# Checking The dataset is balanced or imbalanced

target.value_counts()


# In[30]:


# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, target, test_size = 0.2, 
                                                    stratify = target, random_state = 0) 


# # RandomForestClassifier

# In[77]:


# ## Random Forest Model
# from sklearn.ensemble import RandomForestClassifier

rf_Model = RandomForestClassifier()


# #### Hyperparameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [2, 4]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree
bootstrap = [True, False]


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
n_estimators

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(param_grid)


# ### Hyperparameter optimization with GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 10, verbose = 1, n_jobs = -1)

rf_Grid.fit(X_train, Y_train)

rf_Grid.best_params_

cv_rf_grid = rf_Grid.best_estimator_


# ## Check Accuracy
# Evaluation on Test Data
test_pred = cv_rf_grid.predict(X_test)

accuracy_test = np.mean(test_pred == Y_test)
accuracy_test

cm = skmet.confusion_matrix(Y_test, test_pred)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Machine_failure', 'No_Machine_Failure'])
cmplot.plot()
cmplot.ax_.set(
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

print (f'Train Accuracy - : {rf_Grid.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test, Y_test):.3f}')



# In[79]:


# ### Hyperparameter optimization with RandomizedSearchCV
rf_Random = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid, cv = 10, verbose = 2, n_jobs = -1)

rf_Random.fit(X_train, Y_train)

rf_Random.best_params_

cv_rf_random = rf_Random.best_estimator_

# Evaluation on Test Data
test_pred_random = cv_rf_random.predict(X_test)

accuracy_test_random = np.mean(test_pred_random == Y_test)
accuracy_test_random

cm_random = skmet.confusion_matrix(Y_test, test_pred_random)

cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm_random, display_labels = ['machine_failure', 'no_machine_failure'])
cmplot.plot()
cmplot.ax_.set( 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


print (f'Train Accuracy - : {rf_Random.score(X_train, Y_train):.3f}')
print (f'Test Accuracy - : {rf_Random.score(X_test, Y_test):.3f}')





# In[80]:


# ## Save the best model from Randomsearch CV approach
pickle.dump(cv_rf_random, open('rfc.pkl', 'wb'))


# # GradientBoostingClassifier

# In[50]:


from sklearn.ensemble import GradientBoostingClassifier



# Create a Gradient Boosting Classifier
gbc = GradientBoostingClassifier()

# Fit the model to the training data
gbc.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = gbc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


# In[70]:


# Define the hyperparameters and their possible values
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Create the randomized search object
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42
)
# Splitting data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, target, test_size = 0.2, 
                                                    stratify = target, random_state = 0) 

# Fit the randomized search to the data
random_search.fit(X_train, Y_train)

# Get the best parameters from the randomized search
best_params = random_search.best_params_

# Use the best parameters to train the final model
best_gbc = GradientBoostingClassifier(**best_params)
best_gbc.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = best_gbc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)


# In[71]:


from sklearn.metrics import confusion_matrix

# Save the ML model using Pickle
pickle.dump(best_gbc, open('gradient_boosting_model.pkl', 'wb'))

# Load the saved model
loaded_model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))

# Evaluation on Testing Data
print(confusion_matrix(Y_test, Y_pred))
print('\n')
print(accuracy_score(Y_test, Y_pred))

# Evaluation on Training Data
print(confusion_matrix(Y_train, best_gbc.predict(X_train)))
print(accuracy_score(Y_train, best_gbc.predict(X_train)))


# # Support Vector Machines (SVM):
# 
# 

# In[32]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Create an SVM classifier
svm = SVC()

# Define the hyperparameters and their possible values
param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Create the randomized search object
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

# Fit the randomized search to the data
random_search.fit(X_train, Y_train)

# Get the best parameters from the randomized search
best_params = random_search.best_params_

# Use the best parameters to train the final model
best_svm = SVC(**best_params)
best_svm.fit(X_train, Y_train)

# Train accuracy
train_accuracy = accuracy_score(Y_train, best_svm.predict(X_train))

# Test accuracy
test_accuracy = accuracy_score(Y_test, best_svm.predict(X_test))

print("Best Parameters:", best_params)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[ ]:





# In[ ]:





# # KNeighborsClassifier

# In[33]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a KNeighborsClassifier
knn = KNeighborsClassifier()

# Define the hyperparameters and their possible values
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Adjust the range based on your dataset
    'metric': ['euclidean', 'manhattan']  # Explore different distance metrics
}

# Create the grid search object
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Use the best parameters to train the final model
best_knn = KNeighborsClassifier(**best_params)
best_knn.fit(X_train, Y_train)

# Make predictions on the training set
Y_train_pred_knn = best_knn.predict(X_train)

# Make predictions on the test set
Y_test_pred_knn = best_knn.predict(X_test)

# Calculate training accuracy
train_accuracy_knn = accuracy_score(Y_train, Y_train_pred_knn)

# Calculate testing accuracy
test_accuracy_knn = accuracy_score(Y_test, Y_test_pred_knn)

print("Best Parameters:", best_params)
print("Training Accuracy:", train_accuracy_knn)
print("Testing Accuracy:", test_accuracy_knn)


# # LogisticRegression

# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score


# Create a Logistic Regression model
logreg = LogisticRegression()

# Define the hyperparameters and their possible values
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga']  # Optimization algorithm
}

# Create the grid search object
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='f1', cv=5)

# Fit the grid search to the data
grid_search.fit(X_train, Y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Use the best parameters to train the final model
best_logreg = LogisticRegression(**best_params)
best_logreg.fit(X_train, Y_train)

# Make predictions on the training set
Y_train_pred_logreg = best_logreg.predict(X_train)

# Make predictions on the test set
Y_test_pred_logreg = best_logreg.predict(X_test)

# Calculate training F1 score
train_f1_logreg = f1_score(Y_train, Y_train_pred_logreg, pos_label='Machine_Failure')

# Calculate testing F1 score
test_f1_logreg = f1_score(Y_test, Y_test_pred_logreg, pos_label='Machine_Failure')

print("Best Parameters:", best_params)
print("Logistic Regression Training F1 Score:", train_f1_logreg)
print("Logistic Regression Testing F1 Score:", test_f1_logreg)


# # DecisionTreeClassifier

# In[38]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Assuming 'scaled_data' is your feature matrix and 'target' is your target variable
X_train, X_test, Y_train, Y_test = train_test_split(scaled_data, target, test_size=0.2, stratify=target, random_state=0)

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=0)

# Define the hyperparameters and their possible values
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None] + list(np.arange(10, 101, 10)),
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
}

# Create the randomized search object
random_search = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist, n_iter=10, scoring='f1', cv=5, random_state=0)

# Fit the randomized search to the data
random_search.fit(X_train, Y_train)

# Get the best parameters from the randomized search
best_params = random_search.best_params_

# Use the best parameters to train the final model
best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=0)
best_dt_classifier.fit(X_train, Y_train)

# Make predictions on the training set
Y_train_pred_dt = best_dt_classifier.predict(X_train)

# Make predictions on the test set
Y_test_pred_dt = best_dt_classifier.predict(X_test)

# Calculate training F1 score
train_f1_dt = f1_score(Y_train, Y_train_pred_dt, pos_label='Machine_Failure')

# Calculate testing F1 score
test_f1_dt = f1_score(Y_test, Y_test_pred_dt, pos_label='Machine_Failure')

print("Best Parameters:", best_params)
print("Decision Tree Training F1 Score:", train_f1_dt)
print("Decision Tree Testing F1 Score:", test_f1_dt)


# # Naive Bayes
# 
# 

# In[39]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np



# Create a Gaussian Naive Bayes model
nb = GaussianNB()

# No hyperparameter tuning is typically performed for Gaussian Naive Bayes

# Train the model
nb.fit(X_train, Y_train)

# Make predictions on the training set
Y_train_pred_nb = nb.predict(X_train)

# Make predictions on the test set
Y_test_pred_nb = nb.predict(X_test)

# Calculate training accuracy
train_accuracy_nb = accuracy_score(Y_train, Y_train_pred_nb)

# Calculate testing accuracy
test_accuracy_nb = accuracy_score(Y_test, Y_test_pred_nb)

print("Naive Bayes Training Accuracy:", train_accuracy_nb)
print("Naive Bayes Testing Accuracy:", test_accuracy_nb)


# In[ ]:




