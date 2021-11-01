import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from streamlit_folium import folium_static
import folium
from matplotlib import pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes

def app():
	sns.set_theme(style="whitegrid")
	###Data Cleaning####
	df = pd.read_csv("LaundryData.csv")
	df = df.dropna()

	######Association Rule Mining#########
	df_arm = df.copy()
	df_arm = df_arm[(df_arm['Washer_No'] == 4) | (df_arm['Dryer_No'] == 8)]
	df_arm.drop(columns=['No', 'Date', 'Time', 'Age_Range'], axis = 1, inplace = True)
	one_hot = pd.get_dummies(data = df_arm, columns=['Race', 'Gender', 'Body_Size', 'With_Kids', 'Kids_Category', 'Basket_Size', 'Basket_colour', 'Attire', 'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Wash_Item', 'Spectacles', 'Washer_No', 'Dryer_No'])
	colname = one_hot.columns.tolist()
	colname.insert(0, '-')
	arm_result = pd.read_csv('arm_results.csv', index_col=0)
	st.title("Association Rule Mining")
	item_lhs = st.selectbox("Base Feature",options=colname)
	item_rhs = st.selectbox("Added Feature",options=colname)
	if st.button('Apply'):
		if (item_rhs == '-') & (item_lhs == '-'):
			st.write(arm_result.head(100))
		elif (item_rhs == '-') & (item_lhs != '-'):
			st.write(arm_result[(arm_result['Left Hand Side'] == item_lhs)][:100])
		elif (item_lhs == '-') & (item_rhs != '-'):
			 st.write(arm_result[(arm_result['Right Hand Side'] == item_rhs)][:100])
		else:
			st.write(arm_result[(arm_result['Left Hand Side'] == item_lhs) & (arm_result['Right Hand Side'] == item_rhs)][:100])
		st.text("Showing first 100 results")

	############END######################
	########Clustering###################
	st.title("Clustering using K-Modes")
	no_of_cluster = st.slider("Clusters", 1, 10,1)
	if st.button('Generate Cluster'):
		df_clus = df.copy()
		#binning age
		df_clus['Age_Bins'] = pd.cut(x=df_clus['Age_Range'], bins=[20, 29, 39, 49, 59])
		df_clus.drop(columns=['No', 'Date', 'Time', 'Age_Range'], inplace=True)
		all_features = df_clus.columns
		# Building the model with n clusters
		kmode = KModes(n_clusters=no_of_cluster, init = "random", n_init = 5, verbose=1)
		clusters = kmode.fit_predict(df_clus)
		df_clus.insert(0, "Cluster", clusters, True)
		for col in all_features:
			fig,ax = plt.subplots(figsize = (15,5))
			sns.countplot(x='Cluster',hue=col, data = df_clus, axes=ax)
			st.write(fig)
	############END######################