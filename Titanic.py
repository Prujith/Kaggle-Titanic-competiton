# For loading DF
import pandas as pd
# For visualization
import seaborn as sns
# Encoding
from sklearn.preprocessing import LabelEncoder
# For Prediction
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

New_train_DF = train_df.copy()
New_test_DF = test_df.copy()

# Correcting
# Dropping
New_train_DF = New_train_DF.drop('Ticket', axis=1)
New_train_DF = New_train_DF.drop('PassengerId', axis=1)

# Completing
a = New_train_DF.Embarked.dropna().mode()[0]
New_train_DF['Embarked'] = New_train_DF['Embarked'].fillna(a)

New_train_DF[ 'Age' ] = New_train_DF.Age.fillna( New_train_DF.Age.mean() )
New_test_DF[ 'Age' ] = New_test_DF.Age.fillna( New_test_DF.Age.mean() )

New_train_DF['Cabin'] = New_train_DF['Cabin'].fillna('U')
New_test_DF['Cabin'] = New_test_DF['Cabin'].fillna('U') #Try removing nums by regex or something

# New_train_DF[ 'Fare' ] = New_train_DF.Fare.fillna( New_train_DF.Fare.mean() )
New_test_DF[ 'Fare' ] = New_test_DF.Fare.fillna( New_test_DF.Fare.mean() )


# Converting
LE = LabelEncoder()
New_train_DF['Sex'] = LE.fit_transform(New_train_DF['Sex'])
New_test_DF['Sex'] = LE.fit_transform(New_test_DF['Sex'])

New_train_DF['Cabin'] = New_train_DF.Cabin.apply(lambda y: y[0])
New_test_DF['Cabin'] = New_test_DF.Cabin.apply(lambda y: y[0]) #Try removing nums by regex or something

New_train_DF['Name'] = New_train_DF.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
New_test_DF['Name'] = New_test_DF.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Ms":         "Mrs",
                    "Mrs" :       "Mrs",
                    "Mlle":       "Miss",
                    "Miss" :      "Miss",
                    "Mr" :        "Mr",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
New_train_DF['Name'] = New_train_DF.Name.map(Title_Dictionary)
New_test_DF['Name'] = New_test_DF.Name.map(Title_Dictionary)


# Creating
# OneHotEncoding using pandas
q = ['Cabin','Name','Embarked','Pclass','SibSp','Parch']

for i in q:
     OH = pd.get_dummies(New_train_DF[i], prefix = i)
     New_train_DF = pd.concat([New_train_DF,OH], axis=1)
     New_train_DF = New_train_DF.drop(i,axis=1)

for i in q:
     OH = pd.get_dummies(New_test_DF[i], prefix = i)
     New_test_DF = pd.concat([New_test_DF,OH], axis=1)
     New_test_DF = New_test_DF.drop(i,axis=1)

X_train = New_train_DF.drop('Survived', axis=1)
y_train = New_train_DF['Survived']
X_test = New_test_DF.drop(['PassengerId','Ticket'], axis=1)


RFC = RandomForestClassifier(n_estimators = 100)
RFC.fit(X_train, y_train)
result = RFC.score(X_train,y_train)
print(result)

y_test = RFC.predict(X_test)
Ids = New_test_DF['PassengerId']
Completion = pd.DataFrame({'PassengerId':Ids,
               'Survived':y_test})
Completion.to_csv('Titanic_Predictions.csv', index = False)