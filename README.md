# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```python

Program to implement the SVM For Spam Mail Detection..
Developed by: Sai Hrishi M
RegisterNumber:  212224240140


import chardet
with open("/content/spam.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv("/content/spam.csv", encoding='windows-1252')
print(data.head())

print(data.info())

print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

## Output:

![Screenshot 2025-05-22 142906](https://github.com/user-attachments/assets/7daf901a-e23d-4b05-963a-6e8209965a9f)

![Screenshot 2025-05-22 142935](https://github.com/user-attachments/assets/d4a61356-3f2d-4238-8cee-1e5481afc0d6)

![Screenshot 2025-05-22 143001](https://github.com/user-attachments/assets/48257c21-bc80-42aa-b13a-ad17b58e02bd)

![Screenshot 2025-05-22 143255](https://github.com/user-attachments/assets/d6922ea9-1a57-4c74-8c9f-f799c795359d)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
