import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC  
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

class DataPrep:

    def __init__(self):
        pass

    def fillna_with_group_means(self, df, fill_col, group_col):
        # Calculate the group means
        group_means = df.groupby(group_col)[fill_col].mean()

        # Function to apply
        def replace_nan(row):
            if pd.isnull(row[fill_col]):
                return group_means[row[group_col]]
            else:
                return row[fill_col]

        # Apply the function and create a new DataFrame
        new_df = df.copy()
        new_df[fill_col] = new_df.apply(replace_nan, axis=1)
        return new_df

    def clean_data(self, df, cat_cols):
        df = self.fillna_with_group_means(df, "Age", "Sex")
        df = self.fillna_with_group_means(df, "Age", "Sex")          
        dummies = pd.get_dummies(df[cat_cols])
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, dummies], axis=1)
        
        return df

    def split_data(self, df, features, target):
        train_data = df[0:int(0.5*len(df))]
        val_data = df[int(0.5*len(df)):int(0.75*len(df))]
        test_data = df[int(0.75*len(df)):len(df)]
        return (
            train_data[features], train_data[target],
            val_data[features], val_data[target],
            test_data[features], test_data[target]
        )

class DataExp:
    def __init__(self):
        pass
    def corr(self, df1, df2, target):
        corr_mat = pd.concat([df1, df2], axis=1).corr()
        print(corr_mat)
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm')
        plt.show()

class Model:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
                        
    # Random Forest Classifier
    def RFC(self):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(self.x_train, self.y_train)
        return model

    # Logistic Regression
    def LR(self):
        logreg = LogisticRegression(random_state=16)
        logreg.fit(self.x_train, self.y_train)
        return logreg

    # Support Vector Machine
    def SVM(self):
        clf = SVC(kernel='linear')
        clf.fit(self.x_train, self.y_train)
        return clf

    # Gradient Boosting Classifier
    def GBC(self):
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
        gb_clf.fit(self.x_train, self.y_train)
        return gb_clf

# Evaluates all the models
def evaluate():
    # Random Forest Classification
    rfc_report = classification_report(y_val, model.RFC().predict(x_val))
    print("Random Forest Classification report: ")
    print(rfc_report)
    # Logistic Regression
    lr_report = classification_report(y_val, model.LR().predict(x_val))
    print("Logistic Regression report: ")
    print(lr_report)
    # Support Vector Machine
    svm_report = classification_report(y_val, model.SVM().predict(x_val))
    print("Support Vector Machine report: ")
    print(svm_report)
    # Gradient Boosting Classic=fication
    gbc_report = classification_report(y_val, model.GBC().predict(x_val))
    print("Gradient Boosting Classification report: ")
    print(gbc_report)

# # Creates a csv for final submission
# def create_csv():
#     output = pd.DataFrame({'PassengerId': pred_inp.PassengerId, 'Survived': predictions})
#     output.to_csv('submission2.csv', index=False)
#     print("Your submission was successfully saved!")

data = pd.read_csv("train.csv")
pred_inp = pd.read_csv("test.csv")
cat_cols = ["Pclass", "Sex", "Embarked"]

# Data Prep Stage
dp = DataPrep()
data, pred_inp = dp.clean_data(data, cat_cols), dp.clean_data(pred_inp, cat_cols)


features = data.columns
features = features.drop(['Survived','PassengerId','Name', 'Ticket', 'Fare', 'Cabin'])
target = ["Survived"]
x_train, y_train, x_val, y_val, x_test, y_test = dp.split_data(data, features, target)


# Data Exploration
de = DataExp()
de.corr(x_train, y_train, target)

# Modelling Stage
model = Model(x_train, y_train)

# Evaluation Stage
evaluate()