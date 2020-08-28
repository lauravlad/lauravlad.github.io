---
layout: post
title:      "UCI Dataset - Heart disease modeling!"
date:       2020-08-27 20:56:40 -0400
permalink:  heart_disease_modeling
---

![](https://imgur.com/zdu8h3z.png)


This is my third project with Flatiron Scool of Data Science.

Mayoclinic defines heart disease as:

"a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others.

The term "heart disease" is often used interchangeably with the term "cardiovascular disease." Cardiovascular disease generally refers to conditions that involve narrowed or blocked blood vessels that can lead to a heart attack, chest pain (angina) or stroke. Other heart conditions, such as those that affect your heart's muscle, valves or rhythm, also are considered forms of heart disease."

My goal in this project is to find the best working model for this specific dataset.

I started working with the Heart.csv dataset found on Kaggle but soon I found out that some of the features were plain weird. The target feature was supposed to have two values: 0 for healthy (no heart disease) and 1 for not healthy (heart disease) but they were somehow switched. This was only the begining so I decided to get the dataset directly from <a href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">here </a>. The dataset contains 14 atribute columns and a little over 303 instances. 

Attribute Information:
1. age: age in years
2. sex: sex (1 = male; 0 = female)
3. cp: chest pain type:
Value 1: typical angina 
Value 2: atypical angina 
Value 3: non-anginal pain
Value 4: asymptomatic
4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
5. chol: serum cholestoral in mg/dl
6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. restecg: resting electrocardiographic results( 0 = normal, 1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
8. thalach: maximum heart rate achieved
9. exang: exercise induced angina (1 = yes; 0 = no)
10. oldpeak = ST depression induced by exercise relative to rest
11. slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
12. ca: number of major vessels (0-3) colored by flourosopy
13. Thalium stress test result (normal, fixed defect, or reversible defect)
14. Target: diagnosis of heart disease (angiographic disease status)(0 = absence of heart disease, 1,2,3,4 = presence)


First we imported the necessary packages for data analysis and vizualisations:
```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn')

import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, accuracy_score, recall_score, 

```

We turned the csv document into a dataframe for analysis:

```
data=pd.read_csv('processed.cleveland.csv')
data.head()
```

Then we separated features into categorical data and continuous data using this function:

```
# Function that will find the number of unique values and tell us if it's high or low.
def find_number_unique_values(df):
    #Creates two empty lists for categorical and continuous data
    cont_val=[]
    cat_val=[]
    bool_val =[]
    #Adds column to categorical data list if it has less than 5 unique values or adds it to the continuous data list it it has more than 5 unique values.
    for i in df.columns:
        if df[i].nunique()>6:
            cont_val.append(i)
        elif 2 < df[i].nunique()<6:
            cat_val.append(i)
        else:
            bool_val.append(i)
    print(f" categorical values - {cat_val}.")
    print(f" continuous values - {cont_val}.")
    print(f" boolean values - {bool_val}.")  
```

While analysing each one of the predictors I discovered that 'ca' and 'thal' features had a few values missing so we dropped those rows:

```
#Drop rows with ca = ?.
data = data[data['ca']!='?']
```
Aftyer analysing the dataset we can conclude the obvious: age, lower maximum blood pressure, high blood glucose levels, abnormal elecrocardiographic results, blockages on major blood vessels increase the probability of having heart disease.
 
 We can also deduct less obvious 

According to this dataset, earlier in life man are more likely to develop heart disease than women.

![](https://imgur.com/siWUz1V.png)

Later in life the difference is not that striking.

![](https://imgur.com/QsnCnej.png)

The problem is that there is this stereotype that women are less likely to develop cardiovascular diseases because they are protected by higher leves of estrogen therefore they are less likely to be tested when complaining of heart issues. Instead they are given antidepressants because of another stereotype: women are more likely to be depressed.

"Cardiovascular disease develops 7 to 10 years later in women than in men and is still the major cause of death in women. The risk of heart disease in women is often underestimated due to the misperception that females are ‘protected’ against cardiovascular disease. The under-recognition of heart disease and differences in clinical presentation in women lead to less aggressive treatment strategies and a lower representation of women in clinical trials." (Neth Heart J 2010)

The claim is supported by this dataset where women account for 30% of the total number of pacients/instances.

Preprocessing data:  

Create dummy categories for categorical data:
 
```
 categ_dummy_data= pd.get_dummies(data[categ_cols], drop_first=True)
```
 
Normalize continuous data:

```
def normalize(feature):
    return (feature - feature.mean()) / feature.std()

continuous_data_norm = continuous_data.apply(normalize)

```

Separate dataset into predictors and target and train and test.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
```
 
I tested 5 different models on the dataset, but the Logistic Regression Model performed the best time wise and Recall score wise.

A high Recall Score is needed to reflect a low false negative result.

A false positive will cost more but it's likely that the doctor will order more tests, and down the line they will find out the good news, that the patient has no heart disease. 
A false hegative will cost more down the line because it's less costly to treat a disease in incipient phases, but also delaying the correct diagnosis will ireverebly damage the health of the patient reducing the life quality and expectancy.

Instatiate the model:
```
# Instantiate
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
```

Fit and predict:

```
# Fit the model to training data
model_log = logreg_1.fit(X_train, y_train)

# Predict on test set
y_hat_test = logreg_1.predict(X_test)
```

```
# Calculate the probability scores of each point in the training set
y_train_score = model_log.decision_function(X_train)

# Calculate the fpr, tpr, and thresholds for the training set
train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)

# Calculate the probability scores of each point in the test set
y_test_score = model_log.decision_function(X_test)

# Calculate the fpr, tpr, and thresholds for the test set
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_score)
```

ROC curve:

# Seaborn
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

# ROC curve for training set
plt.figure(figsize=(10, 8))
lw = 2
plt.plot(train_fpr, train_tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks([i/20.0 for i in range(21)])
plt.xticks([i/20.0 for i in range(21)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve for Training Set')
plt.legend(loc='lower right')
print('Training AUC: {}'.format(metrics.auc(train_fpr, train_tpr)))
plt.show()

![](https://imgur.com/90H9Oq0.png)

Calculate Recall Score:

```
y_hat_train = logreg_1.predict(X_train)
y_hat_test = logreg_1.predict(X_test)
```

```
print('Training Recall: ', recall_score(y_train, y_hat_train))
print('Testing Recall: ', recall_score(y_test, y_hat_test))
print('\n\n')
```

Et voila!

![](https://imgur.com/BRgT6ZZ.png)

