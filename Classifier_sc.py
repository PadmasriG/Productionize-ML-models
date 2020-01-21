import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

basic_params_dict = {
    
    'X':['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'y':['species_id'],
    'seed': 123,
    'test_size': 0.3
}

classifier_config_dict = {

    # Classifiers
   

    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'LogisticRegression':LogisticRegression(),
    'KMeans': KMeans(),
    'KNeighborsClassifier' : KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC':SVC()
    
   # 'KNeighborsRegressor':KNeighborsRegressor()
   # 'LinearRegression' : LinearRegression()
    #RandomForestRegressor
    #SVR

}

gridsearchParameters_dict={
#Hyper parameters for the respective models
    
	'DecisionTreeClassifier':{'criterion':['gini','entropy'],
                              'max_depth':[2,4,8,16,20,24,30,32],'min_samples_split':[4],'max_features':['auto','sqrt']},
    
    'LogisticRegression':{'penalty': ['l1', 'l2'],'C':[0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,5.0,8.0,10.0,15.0]},
    
    'KMeans': {'n_clusters': [5,10,15,20,25,30]},
    
    'KNeighborsClassifier':{'n_neighbors': [3,6,9,12],'weights': ['uniform', 'distance'],'p':[1,2]},
    
    'RandomForestClassifier':{'n_estimators': range(5,20,2),'criterion':['gini','entropy'],'max_features' : ['auto', 'sqrt'],
              'max_depth' : [30,40,50],'min_samples_split':[2,4,6,8],'min_samples_leaf':[1,2,3,4,5]},
    
    'SVC':{'C':[0.001, 0.01, 0.1, 1, 10],'gamma' :[0.001, 0.01, 0.1, 1]}
    
    
   # 'KNeighborsRegressor':{'n_neighbors': [3,6,9,12],'leaf_size':[10,20,30],'weights': ['uniform', 'distance'],'p':[1,2]}   
    
                         # 'class_weight':['dict']}
    #'class_weight':{0:0.4, 1:0.6}}
     
	
	#'KNearestNeighbour' :{},
	#'LinearRegression' : {}


}