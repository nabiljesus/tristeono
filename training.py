import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import AdaBoostClassifier

data = pd.read_csv('emotions_vector.csv')
data.drop(data.columns[[0,2]],axis=1,inplace=True)


# train, test = train_test_split(data, test_size = 0.2)


clf = AdaBoostClassifier(n_estimators=10)

target = data[[0]].replace('sadness','1.0').replace('happiness','0.0').astype('float64')
data.drop(data.columns[[0]],axis=1,inplace=True)

predicted = cross_val_predict(clf, data, target.values, cv=10)
# import pdb; pdb.set_trace()

print(predicted)
