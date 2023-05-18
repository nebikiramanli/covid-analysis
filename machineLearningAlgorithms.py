import pandas as pd
import numpy as np
import seaborn as sns
from dataPreprocesing import selected_features
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
#! Hataların cıkmaması için
from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score


df=selected_features

#?  x and y
x=df.drop("COVID-19",axis=1)
#%%
print(x)
#%%
y=df["COVID-19"]

#? empty list for models score

scores=[]
model_names=[]



#? split training and test
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import StandardScaler
def standart_scaler(X_train,X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train,X_test

def model_authentication(name,model,y_test,y_pred,X_train,X_test):
    print("***********************",name," Authentication *********************")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    meanSquared_error = mean_squared_error(y_test,y_pred)
    print("Mean Squared Error",np.sqrt(meanSquared_error))

    # ? classification_report
    print("classification_report \n", classification_report(y_test, y_pred))
    print("cross_val_score mean ", cross_val_score(model, X_test, y_test, cv=10).mean())

    # ? Sınıflandırma olasılarını tahmin etme
    print(model.predict_proba(X_train)[0:10])
    print(model.predict(X_train)[0:10])
    #? Model Score /Accuracy and Accurate classification rate
    print("accuracy_score \n", accuracy_score(y_test, y_pred))


#! Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
def logistic_regression(X_train,X_test,y_train,y_test):
    #? standart_scaler
    X_train,X_test=standart_scaler(X_train,X_test)
    #? logistic model
    logictic=LogisticRegression(solver="liblinear",random_state=42)
    logistic_model=logictic.fit(X_train,y_train)
    
    #? model prediction 
    y_pred=logistic_model.predict(X_test)
    #? model authentication
    model_authentication("Logistic Regression",logistic_model,y_test,y_pred,X_train,X_test)
    #! model tuning
    logic_params = {"penalty":["l1","l2","elasticnet","none"],"C":[0.5,1.0,2,5,10,100,500,1000],
                    "solver":["newton-cg","lbfgs","liblinear","sag","saga"]}
    #? default model
    logictic_default=LogisticRegression()
    #! Logistic Model GridSearchCV
    #logictic_cv_model=GridSearchCV(logictic_default,logic_params,verbose=2,cv=10,n_jobs=-1)
    #logictic_cv_model.fit(X_train,y_train)
    #print(logictic_cv_model.best_score_)
    #print(logictic_cv_model.best_params_)
    #? Model Tuned
    logictic_tuned=LogisticRegression(penalty="none",C=0.5,solver="newton-cg")
    logictic_tuned_model=logictic_tuned.fit(X_train,y_train)
    y_pred_tuned=logictic_tuned_model.predict(X_test)
    #? model tuned authentication
    model_authentication("Logistic Model Tuned ",logictic_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Model score \n", logictic_tuned_model.score(X_test, y_test) * 100)
    scores.append(logictic_tuned_model.score(X_test,y_test)*100)
    model_names.append("Logistic Model")

#! Neural Network
from sklearn.neural_network import MLPClassifier
def neural_network(X_train,X_test,y_train,y_test):
    # ? standart_scaler
    X_train, X_test = standart_scaler(X_train, X_test)
    #? Artificial Neural Network Model
    artificialNeural=MLPClassifier(random_state=42)
    artificialNeural_model=artificialNeural.fit(X_train,y_train)
    #? model prediction
    y_pred=artificialNeural_model.predict(X_test)
    # ? model authentication
    model_authentication("Artificial Neural Network ", artificialNeural_model, y_test, y_pred, X_train, X_test)

    #! Model tuning
    artificialNeural_params={"alpha":[1,0.1,0.01,0.02,0.005,0.0001,0.00001],
                             "hidden_layer_sizes":[(10,10,10),(100,100,100),(100,100),(3,5),(5,3),(100)],
                             "solver":["lbfgs","adam","sgd"],
                             "activation":["relu","logistic","identity","tanh"]}
    #? default model
    #artificialNeural_default=MLPClassifier()
    #artificialNeural_cv_model=GridSearchCV(artificialNeural_default,artificialNeural_params,n_jobs=-1,cv=10,verbose=2)
    #artificialNeural_cv_model.fit(X_train,y_train)
    #print(artificialNeural_cv_model.best_score_)
    #print(artificialNeural_cv_model.best_params_)

    #? Model Tuned
    artificialNeural_tuned_model=MLPClassifier(hidden_layer_sizes=(10,10,10),activation='tanh',alpha=0.01,solver="adam",random_state=42)
    artificialNeural_tuned_model.fit(X_train,y_train)
    y_pred_tuned=artificialNeural_tuned_model.predict(X_test)

    model_authentication("Artificial Neural Network Tuned Model",artificialNeural_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Model score \n", artificialNeural_tuned_model.score(X_test, y_test) * 100)
    scores.append(artificialNeural_tuned_model.score(X_test, y_test) * 100)
    model_names.append("Artificial Neural Network")

#! Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
def decision_tree(X_train,X_test,y_train,y_test):
    #? standart scaler
    X_train,X_test=standart_scaler(X_train,X_test)
    #? DecisionTree Model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree_model=decision_tree.fit(X_train,y_train)
    #? model prediction
    y_pred = decision_tree_model.predict(X_test)
    # ? model authentication
    model_authentication("Decision Tree Model",decision_tree_model,y_test,y_pred,X_train,X_test)
    #! model tuning
    decision_tree_params={"criterion":["gini","entropy"],"splitter":["best","random"],
                          "max_depth":[3,4,5,6,9,10,12,15],"min_samples_split":[2,3,4,5,10,15,20],
                          "min_samples_leaf":[0.5,1,2,3,5],"max_features":["auto","sqrt","log2"]}
    #? default model
    #decision_tree_default=DecisionTreeClassifier()
    #decision_tree_cv_model=GridSearchCV(decision_tree_default,decision_tree_params,cv=10,verbose=2,n_jobs=-1)
    #decision_tree_cv_model.fit(X_train,y_train)
    #print(decision_tree_cv_model.best_score_)
    #print(decision_tree_cv_model.best_params_)
    #? Model tuned
    decision_tree_=DecisionTreeClassifier(criterion="entropy",max_depth=10,max_features="log2",
                                          min_samples_leaf=2,min_samples_split=2,splitter="best")
    decision_tree_tuned_model=decision_tree_.fit(X_train,y_train)
    y_pred_tuned=decision_tree_tuned_model.predict(X_test)
    model_authentication("Decision Tree Tuned Model",decision_tree_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Model Score :\n ",decision_tree_tuned_model.score(X_test, y_test) * 100)
    scores.append(decision_tree_tuned_model.score(X_test, y_test)*100)
    model_names.append("Decision Tree")
#! KNN
from sklearn.neighbors import KNeighborsClassifier
def k_neighbors(X_train,X_test,y_train,y_test):
    # ? standart scaler
    X_train, X_test = standart_scaler(X_train, X_test)
    #? KNN Algorithm
    k_neighbors=KNeighborsClassifier()#default n_neighbors =5
    k_neighbors_model=k_neighbors.fit(X_train,y_train)
    #? model prediction
    y_pred=k_neighbors_model.predict(X_test)
    model_authentication("K Neighbors ",k_neighbors_model,y_test,y_pred,X_train,X_test)

    #! Model Tuning
    k_neighbors_params={"n_neighbors":np.arange(1,50),"weights":["uniform","distance"],
                        "algorithm":["auto","ball_tree","kd_tree","brute"],"leaf_size":[15,30,45,60]}
    #k_neighbors_default=KNeighborsClassifier()
    #k_neighbors_cv_model=GridSearchCV(k_neighbors_default,k_neighbors_params,n_jobs=-1,cv=10,verbose=2)
    #k_neighbors_cv_model.fit(X_train,y_train)
    #print(k_neighbors_cv_model.best_score_)
    #print(k_neighbors_cv_model.best_params_)
    #? Model Tuned
    k_neighbors_tuned=KNeighborsClassifier(n_neighbors=5,algorithm="auto",leaf_size=60,weights="distance")
    k_neighbors_tuned_model=k_neighbors_tuned.fit(X_train,y_train)
    y_pred_tuned=k_neighbors_tuned_model.predict(X_test)
    model_authentication("K Neighbors Tuned Model ",k_neighbors_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Model Score",k_neighbors_tuned_model.score(X_test, y_test) * 100)
    scores.append( k_neighbors_tuned_model.score(X_test, y_test) * 100)
    model_names.append("K Neighbors")

#! Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
def gradient_boosting(X_train,X_test,y_train,y_test):
    # ? standart scaler
    X_train, X_test = standart_scaler(X_train, X_test)
    #?  GradientBoosting
    gradient_boost=GradientBoostingClassifier()
    gradient_boost_model=gradient_boost.fit(X_train,y_train)
    #? model prediction
    y_pred=gradient_boost_model.predict(X_test)
    model_authentication("Gradient Boosting",gradient_boost_model,y_test,y_pred,X_train,X_test)

    #! Model Tuning
    gradient_boost_params={"loss":["deviance","exponential"],"learning_rate":[0.001,0.01,0.1,1,0.5,0,6,0.03],
                           "n_estimators":[10,100,500,1000],"criterion":["friedman_mse","mse"],
                           "max_depth":[3,5,10],"min_samples_split":[2,5,10]}

    gradient_boost_default=GradientBoostingClassifier()
    #gradient_boost_cv_model=GridSearchCV(gradient_boost_default,gradient_boost_params,n_jobs=-1,cv=10,verbose=2)
    #gradient_boost_cv_model.fit(X_train,y_train)
    #print(gradient_boost_cv_model.best_score_)
    #print(gradient_boost_cv_model.best_params_)
    #? Model Tuned
    gradient_boost_tuned=GradientBoostingClassifier(criterion="friedman_mse",learning_rate=1,
                                                    loss="deviance",max_depth=5,min_samples_split=2,
                                                    n_estimators=100)
    gradient_boost_tuned_model=gradient_boost_tuned.fit(X_train,y_train)
    y_pred_tuned=gradient_boost_tuned_model.predict(X_test)
    model_authentication("Gradient Boosting Tuned Model",gradient_boost_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Model Score",gradient_boost_tuned_model.score(X_test, y_test) * 100)
    scores.append( gradient_boost_tuned_model.score(X_test, y_test) * 100)
    model_names.append("Gradient Boosting")



#! Extreme Gradient Boosting
import xgboost
from xgboost import XGBClassifier
def extreme_gradient_boost(X_train,X_test,y_train,y_test):
    # ? standart scaler
    X_train, X_test = standart_scaler(X_train, X_test)
    # ? XGboost
    x_gradient=XGBClassifier()
    x_gradient_model=x_gradient.fit(X_train,y_train)
    #? prediction
    y_pred= x_gradient_model.predict(X_test)
    model_authentication("Extreme Gradient Boost Model ",x_gradient_model,y_test,y_pred,X_train,X_test)
    #! Model Tuning
    x_gradient_params= {"max_depth":[3,4,5,6],"learning_rate":[0.1,0.01,0.02,0.05],
                        "n_estimators":[100,500,1000,2000],"subsample":[0.6,0.8,1.0],
                        "min_child_weight":[1,2,5,10,20]}
    #x_gradient_default=XGBClassifier()
    #x_gradient_cv=GridSearchCV(x_gradient_default,x_gradient_params,cv=10,verbose=2,n_jobs=-1)
    #x_gradient_cv_model=x_gradient_cv.fit(X_train,y_train)
    #print(x_gradient_cv_model.best_score_)
    #print(x_gradient_cv_model.best_params_)
    #? Model Tuned
    x_gradient_tuned=XGBClassifier(learning_rate=0.1,max_depth=5,min_child_weight=2,n_estimators=1000,subsample=0.8)
    x_gradient_tuned_model=x_gradient_tuned.fit(X_train,y_train)
    y_pred_tuned=x_gradient_tuned_model.predict(X_test)
    model_authentication("Extreme Gradient Boosting Tuned Model",x_gradient_tuned_model,y_test,y_pred_tuned,X_train,X_test)
    print("Extreme Gradient Boosting Tuned Model Score",x_gradient_tuned_model.score(X_test,y_test)*100)
    scores.append( x_gradient_tuned_model.score(X_test, y_test) * 100)
    model_names.append("Extreme Gradient Boosting")


from dataPreprocesing import fake_data

#! Support Vector Machines
from sklearn.svm import SVC
def support_vector_machine(X_train,X_test,y_train,y_test):
    # ? standart scaler
    X_train, X_test = standart_scaler(X_train, X_test)
    #? SVM
    support_vector=SVC(probability=True)
    support_vector_model=support_vector.fit(X_train,y_train)
    #? prediction
    y_pred=support_vector_model.predict(X_test)
    model_authentication("Support Vector Machines Model",support_vector_model,y_test,y_pred,X_train,X_test)
    #! Model Tuning
    svm_params={'C':[0.0001,0.001,0.1,1,5,10,50,100],"kernel":["linear","poly","rbf","sigmoid"],
                "gamma":[0.0001,0.001,0.1,1,5,10,50,100]}
    support_vector_default=SVC()
    #support_vector_cv=GridSearchCV(support_vector_default,svm_params,verbose=2,cv=10,n_jobs=-1)
    #support_vector_cv_model=support_vector_cv.fit(X_train,y_train)
    #print(support_vector_cv_model.best_score_)
    #print(support_vector_cv_model.best_params_)
    #? Model Tuned
    support_vector_tuned=SVC(C=1,gamma=0.1,kernel="rbf",probability=True)
    support_vector_tuned_model=support_vector_tuned.fit(X_train,y_train)
    y_pred_tuned=support_vector_tuned_model.predict(X_test)
    model_authentication("Support vector Machines  Tuned Model",support_vector_tuned_model,y_test,y_pred,X_train,X_test)
    #! Fake data Test
    fake_df=fake_data
    sc=StandardScaler()
    x_fake=fake_df.drop("COVID-19",axis=1)
    y_fake=fake_df["COVID-19"]
    x_fake=sc.fit_transform(x_fake)
    y_fake_pred=support_vector_tuned_model.predict(x_fake)

    print("*************************Fake Pred*******\n")
    cm = confusion_matrix(y_fake, y_fake_pred)
    print(cm)
    print("accuracy_score \n", accuracy_score(y_fake, y_fake_pred))

    print(support_vector_tuned_model.score(X_test,y_test)*100)
    scores.append(support_vector_tuned_model.score(X_test, y_test) * 100)
    model_names.append("Support Vector Machines")


def model_selection():
    pass
logistic_regression(X_train,X_test,y_train,y_test)
neural_network(X_train,X_test,y_train,y_test)
decision_tree(X_train,X_test,y_train,y_test)
k_neighbors(X_train,X_test,y_train,y_test)
gradient_boosting(X_train,X_test,y_train,y_test)
extreme_gradient_boost(X_train,X_test,y_train,y_test)
support_vector_machine(X_train,X_test,y_train,y_test)
#%%
import plotly.express as px

model_scores =pd.DataFrame(scores,columns=["Score"])
model_names =pd.DataFrame(model_names,columns=["Model"])
model_scores=pd.concat((model_names,model_scores),axis=1)
#print(model_scores)
model_scores=model_scores.sort_values(by='Score')
#print(model_scores)
#model_scores.plot(x="Model",y="Score",figsize=(18,8))
#plt.show()

#! Score Table
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=model_scores.values, colLabels=model_scores.columns, loc='center')
fig.tight_layout()
plt.show()
#%