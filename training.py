import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import cPickle

#Limpiando la data
data = pd.read_csv('emotions_vector.csv')
data.drop(data.columns[[0,2]],axis=1,inplace=True)
# train, test = train_test_split(data, test_size = 0.2)
target = data[[0]].replace('sadness','1.0').replace('happiness','0.0').astype('float64')
target = pd.DataFrame.as_matrix(target).ravel()
data.drop(data.columns[[0]],axis=1,inplace=True)

#AdaBoostClassifier
# predicted = cross_val_predict(clf, data, target, cv=10)
for i in [10,100,200,350]:
    clf = AdaBoostClassifier(n_estimators=350)
    predicted = cross_val_score(clf, data, target, cv=10)
    print(predicted.mean())

    # Guardando el modelo
    with open('AdaBoost'+str(i)+".pkl", 'wb') as fid:
        cPickle.dump(clf, fid)   

# import pdb; pdb.set_trace()

# print(predicted)
 


# Bagged Decision Trees for Classification
X = data
Y = target
seed = 85
kfold = KFold(n_splits=100, random_state=seed)
cart = DecisionTreeClassifier()
for num_trees in [1,10,30,50,100]:
    #num_trees=50
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())

    # Guardando el modelo
    with open('BaggingTrees'+str(num_trees)+".pkl", 'wb') as fid:
        cPickle.dump(cart, fid)   

##Para cargar un modelo:
#with open('my_dumped_classifier.pkl', 'rb') as fid:
#    gnb_loaded = cPickle.load(fid)
