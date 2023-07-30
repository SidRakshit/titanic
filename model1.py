import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC  
from sklearn.ensemble import GradientBoostingClassifier

# Usage:
# fillna_with_group_means(df, 'Age', 'Sex')
def fillna_with_group_means(df, fill_col, group_col):
    # Calculate the group means
    group_means = df.groupby(group_col)[fill_col].mean()

    # Function to apply
    def replace_nan(row):
        if pd.isnull(row[fill_col]):
            return group_means[row[group_col]]
        else:
            return row[fill_col]

    # Apply the function
    df[fill_col] = df.apply(replace_nan, axis=1)

# Cleans the data and sets it up to be processed
def clean_data(data, pred_inp):
    fillna_with_group_means(data, "Age", "Sex")
    fillna_with_group_means(pred_inp, "Age", "Sex")
    data = pd.get_dummies(data)
    pred_inp = pd.get_dummies(pred_inp)
    return data, pred_inp

# Splits the data into train, validation and test subsets
# Returns 6 data frames
def split_data(features, target):
    train_data = data[:int(0.5*len(data))]
    val_data = data[int(0.5*len(data)):int(0.75*len(data))]
    test_data = data[int(0.75*len(data)):]
    return train_data[features], train_data[target], \
        val_data[features], val_data[target], \
            test_data[features], test_data[target]

# Random Forest Classifier
def RFC():
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(x_train, y_train)
    return model

def LR():
    logreg = LogisticRegression(random_state=16)
    logreg.fit(x_train, y_train)
    return logreg

def SVM():
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    return clf

def GBC():
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
    gb_clf.fit(x_train, y_train)
    return gb_clf

# Evaluates all the models
def evaluate():
    # Random Forest Classification
    rfc_report = classification_report(y_val, RFC().predict(x_val))
    print("Random Forest Classification report: ")
    print(rfc_report)
    # Logistic Regression
    lr_report = classification_report(y_val, LR().predict(x_val))
    print("Logistic Regression report: ")
    print(lr_report)
    # Support Vector Machine
    svm_report = classification_report(y_val, SVM().predict(x_val))
    print("Support Vector Machine report: ")
    print(svm_report)
    # Gradient Boosting Classic=fication
    gbc_report = classification_report(y_val, GBC().predict(x_val))
    print("Gradient Boosting Classification report: ")
    print(gbc_report)
    

# # Creates a csv for final submission
# def create_csv():
#     output = pd.DataFrame({'PassengerId': pred_inp.PassengerId, 'Survived': predictions})
#     output.to_csv('submission2.csv', index=False)
#     print("Your submission was successfully saved!")

data = pd.read_csv("train.csv")
pred_inp = pd.read_csv("test.csv")
data, pred_inp = clean_data(data, pred_inp)

features = ["Pclass", "Age", "Sex_female", "Sex_male", "SibSp", "Parch"]
target = ["Survived"]
x_train, y_train, x_val, y_val, x_test, y_test = split_data(features, target)

evaluate()

