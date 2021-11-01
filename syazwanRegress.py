#syazwanRegression.py

#import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

df_GD = pd.read_csv("dfEncoded.csv")
boruta_score = pd.read_csv("borutaScore.csv")
os_data_X=pd.read_csv("os_data_X.csv")
os_data_y=pd.read_csv("os_data_y.csv")

def app():
	y = df_GD.loc[:, df_GD.columns == 'Wash_Item_clothes']
	X = df_GD.drop(["Wash_Item_blankets","Wash_Item_clothes"], 1)
	colnames = X.columns

	st.markdown("## Oversampling")
	with st.expander(label = "Data before ovesampling "):
		st.write('length of data is ',len(X))
		st.write('number of clothes in original data :',len(y[y['Wash_Item_clothes']==1]))
		st.write('number of bankets in original data :',len(y[y['Wash_Item_clothes']==0]))
		st.write('proportion of clothes data in original data :',len(y[y['Wash_Item_clothes']==1])/len(X))
		st.write('proportion of blankets in original data :',len(y[y['Wash_Item_clothes']==0])/len(X))

	with st.expander("Data after ovesampling "):
		st.write('length of oversampled data is ',len(os_data_X))
		st.write('number of clothes in ovesampled data :',len(os_data_y[os_data_y['Wash_Item_clothes']==1]))
		st.write('number of bankets in ovesampled data :',len(os_data_y[os_data_y['Wash_Item_clothes']==0]))
		st.write('proportion of clothes data in oversampled data :',len(os_data_y[os_data_y['Wash_Item_clothes']==1])/len(os_data_X))
		st.write('proportion of blankets in oversampled data :',len(os_data_y[os_data_y['Wash_Item_clothes']==0])/len(os_data_X))

	st.markdown("## Feature Selection")
	st.markdown("### Top 5 features")
	st.dataframe(boruta_score.head(5))
	st.markdown("### Bottom 5 features")
	st.dataframe(boruta_score.tail(5))
	st.markdown("Boruta all features")
	st.write(alt.Chart(boruta_score).mark_bar().encode(
		y=alt.X('Features', sort = boruta_score['Features'].ravel()),
		x='Score').properties(height = 700, width = 700)
	)

	cols = boruta_score[boruta_score['Score']>0]['Features'].ravel()
	X=os_data_X[cols]
	y=os_data_y.Wash_Item_clothes
	logit_model=sm.Logit(y,X)
	result=logit_model.fit(maxiter=200)
	with st.expander(label = "Logit model"):
		st.write(result.summary2())
		
	dropcol = ['Dryer_No','Kids_Category','Basket_colour','Body_Size','Spectacles','shirt_type']
	X1 = X.drop(dropcol,1)
	logit_model=sm.Logit(y,X1)
	result=logit_model.fit(maxiter=200)
	with st.expander(label = "Logit model after removing some columns"):
		st.code("dropcol = ['Dryer_No','Kids_Category','Basket_colour','Body_Size','Spectacles','shirt_type']", language = "python")
		st.write(result.summary2())

	st.markdown("## Logistic Regression with removed features")
	X_train,X_test,y_train,y_test=train_test_split(X1,y.values.ravel(),test_size=0.2,random_state=0)
	pickle_in = open('logreg.pkl', 'rb')
	logreg = pickle.load(pickle_in)
	y_pred = logreg.predict(X_test)
	confusion_matrix1 = confusion_matrix(y_test,y_pred)
	st.markdown("### confusion matrix")
	st.write(confusion_matrix1)
	st.markdown("### classification report")
	st.write(classification_report(y_test, y_pred))

	st.markdown("### ROC Curve") 
	plot_roc_curve(logreg, X_test, y_test)
	st.pyplot()

	st.markdown("## Logistic Regression without removed features")
	X2 = X
	x_train,x_test,Y_train,Y_test=train_test_split(X2,y.values.ravel(),test_size=0.2,random_state=0)
	pickle_in = open('logreg2.pkl', 'rb')
	logreg = pickle.load(pickle_in)
	Y_pred = logreg.predict(x_test)
	confusion_matrix2 = confusion_matrix(Y_test,Y_pred)
	st.markdown("### confusion matrix")
	st.write(confusion_matrix2)
	st.markdown("### classification report")
	st.write(classification_report(Y_test, Y_pred))

	st.markdown("### ROC Curve") 
	plot_roc_curve(logreg, x_test, Y_test)
	st.pyplot()