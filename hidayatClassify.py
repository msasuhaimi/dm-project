import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

os_data_X=pd.read_csv("os_data_X.csv")
os_data_y=pd.read_csv("os_data_y.csv")
boruta_score = pd.read_csv("borutaScore.csv")

def app():
	cols = boruta_score[boruta_score['Score']>0]['Features'].ravel()
	X=os_data_X[cols]
	y=os_data_y.Wash_Item_clothes
	dropcol = ['Dryer_No','Kids_Category','Basket_colour','Body_Size','Spectacles','shirt_type']
	X = X.drop(dropcol,1)

	X_train,X_test,y_train,y_test=train_test_split(X,y.values.ravel(),test_size=0.2,random_state=0)

	#NB
	st.markdown("# Naive Bayes Classifier")
	nb = GaussianNB()
	nb.fit(X_train, y_train)

	y_pred = nb.predict(X_test)
	y_pred

	nb.score(X_test, y_test)
	st.write("Accuracy on test set: {:.3f}".format(nb.score(X_test, y_test)))



	prob_NB = nb.predict_proba(X_test)
	prob_NB = prob_NB[:,1]

	auc_NB= roc_auc_score(y_test, prob_NB)
	st.write('AUC: %.2f' % auc_NB)

	confusion_majority=confusion_matrix(y_test, y_pred)


	st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

	st.write('**')
	st.write('Mjority TN= ', confusion_majority[0][0])
	st.write('Mjority FP=', confusion_majority[0][1])
	st.write('Mjority FN= ', confusion_majority[1][0])
	st.write('Mjority TP= ', confusion_majority[1][1])
	st.write('**')

	st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
	st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
	st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
	st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

	#KNN
	st.markdown("# KNN Classifier")
	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors=5)
	 
	knn.fit(X_train, y_train)



	st.write("Accuracy on test set: {:.3f}".format(knn.score(X_test, y_test)))

	prob_KNN = nb.predict_proba(X_test)
	prob_KNN = prob_KNN[:,1]

	auc_KNN= roc_auc_score(y_test, prob_KNN)
	st.write('AUC: %.2f' % auc_KNN)

	confusion_majority=confusion_matrix(y_test, y_pred)


	st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

	st.write('**')
	st.write('Mjority TN= ', confusion_majority[0][0])
	st.write('Mjority FP=', confusion_majority[0][1])
	st.write('Mjority FN= ', confusion_majority[1][0])
	st.write('Mjority TP= ', confusion_majority[1][1])
	st.write('**')

	st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
	st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
	st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
	st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))