# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

if __name__ == "__main__":
    # print("***** Train_Set *****")
    # print(train.head())
    # print("\n")
    # print("***** Test_Set *****")
    # print(test.head())
    # print(train.columns.values)

    # For the train set
    #train.isna().head()
    # For the test set
    #test.isna().head()

    # print("*****In the train set*****")
    # print(train.isna().sum())
    # print("\n")
    # print("*****In the test set*****")
    # print(test.isna().sum())

    # Fill missing values with mean column values in the train set
    train.fillna(train.mean(), inplace=True)
    # Fill missing values with mean column values in the test set
    test.fillna(test.mean(), inplace=True)

    # print("*****In the train set*****")
    # print(train.isna().sum())
    # print("\n")
    # print("*****In the test set*****")
    # print(test.isna().sum())

    # print(train[['Pclass', 'Survived']].groupby(['Pclass'],
    #     as_index=False).mean().sort_values(by='Survived', ascending=False))

    # print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
    #     ascending=False))

    # print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
    #     ascending=False))
    
    # g = sns.FacetGrid(train, col='Survived')
    # fig = g.map(plt.hist, 'Age', bins=20)
    # fig.savefig("output.png")

    train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
    test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
    print(train.head())

    labelEncoder = LabelEncoder()
    labelEncoder.fit(train['Sex'])
    labelEncoder.fit(test['Sex'])
    train['Sex'] = labelEncoder.transform(train['Sex'])
    test['Sex'] = labelEncoder.transform(test['Sex'])
    # Let's investigate if you have non-numeric data left
    #train.info()
    X = np.array(train.drop(['Survived'], axis=1).astype(float))
    y = np.array(train['Survived'])
    print(y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)
    kmeans.fit(X_scaled)

    print(kmeans.labels_)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X_scaled[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me)) 
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

    print(correct/len(X))