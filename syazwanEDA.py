#syazwanEDA.py

#import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import io

def app ():
	#read CSV
	df = pd.read_csv("LaundryData.csv")

	st.title("Exploratory Data Analysis and Data Pre Processing")

	# What are the data types for each column?
	st.markdown("## What are the data types for each column?")
	buffer = io.StringIO()
	with st.expander(label='info of dataframe'):
		df.info(buf=buffer)
		st.text(buffer.getvalue())

	# Are there any null values in each column?
	st.markdown("## Are there any null values in each column?")

	tempdf = df
	null_columns=tempdf.columns[tempdf.isnull().any()]
	with st.expander(label = "Number of null values for each column."):
		st.write(tempdf[null_columns].isnull().sum())
		buffer2 = io.StringIO()
	tempdf[tempdf.isnull().any(axis=1)][null_columns].info(buf=buffer2)
	with st.expander(label="Info of rows with null values."):
		st.text(buffer2.getvalue())
	with st.expander(label="Subset of dataframe with null values in respective rows and columns."):
		st.dataframe(tempdf[tempdf.isnull().any(axis=1)][null_columns])

	# Removing null values.
	st.markdown("## Removing null values.")
	with st.expander(label='info of dataframe after dropping null rows'):
		df = df.dropna()
		buffer3 = io.StringIO()
		df.info(buf=buffer3)
		st.text(buffer3.getvalue())
	with st.expander(label='dataframe after dropping null rows'):
		st.dataframe(df)

	# Which age groups frequent the laundry the most?
	st.markdown("## Which age groups frequent the laundry the most?")

	bins = [20,30,40,50,60]
	fig, ax = plt.subplots()
	sns.histplot(data=df, x="Age_Range", bins = bins)
	with st.expander(label = 'Histogram of Age Range'):
		st.pyplot(fig)

	# Which race frequently goes for the laundry the most?
	st.markdown("## Which race frequently goes for the laundry the most?")

	fig, ax = plt.subplots()
	df["Race"].value_counts().plot(kind="barh",xlabel = "Race")
	with st.expander(label = 'Total number of Race'):
		st.pyplot(fig)

	# are there any particularly popular washer-dryer combos?
	st.markdown("## Are there any particularly popular washer-dryer combinations?")

	df1=df.groupby(['Washer_No','Dryer_No']).size().reset_index().rename(columns={0:'count'})
	df1['Washer_Dryer_combo'] = '['+df1['Washer_No'].astype(str) + ', ' + df1['Dryer_No'].astype(str)+']'
	df1 = df1.sort_values('count', ascending = False).reset_index(drop=True)

	with st.expander(label = 'dataframe of Washer-Dryer combinations'):
		st.dataframe(df1)
	fig, ax = plt.subplots()
	sns.barplot(x = 'Washer_Dryer_combo', y = 'count', data = df1.head())
	with st.expander(label = "Top 5 Washer-Dryer combinations"):
		st.pyplot(fig)

	# what kind of customers are using washer 3 and dryer 7?
	st.markdown("## What kind of customers are using washer 3 and dryer 7?")

	popwash = df['Washer_No']==3
	popdry = df['Dryer_No']==7
	df2col = ['Race','Gender','Body_Size','Wash_Item','Basket_Size']
	df2 = df[popwash]
	df2 = df2[popdry]
	df2 = df2[df2col]
	fig, ax = plt.subplots()

	with st.expander(label = 'Bar chart of Customer demographic'):
		demo_sel = st.selectbox("Demographic", options = df2col)
		if st.button('Apply Demographic'):
			df2 = df2.groupby([demo_sel]).size().reset_index().rename(columns={0:'count'})
			sns.barplot(x = demo_sel, y = 'count', data = df2, ci = None)
		st.write(fig)

	# is there a relation between race and basket size?
	st.markdown("## Is there a relation between race and basket size?")
	df3=df.groupby(['Basket_Size','Race']).size().reset_index().rename(columns={0:'Basket_count'}).sort_values('Basket_count', ascending=False)
	fig, ax = plt.subplots()
	ax.set(ylim=(0,200))
	with st.expander(label = 'Bar chart of Race and Basket Size'):
		size_sel = st.selectbox("Basket_Size", options = ['big','small'])
		if st.button('Apply Size'):
			sns.barplot(x = 'Race', y = 'Basket_count', data=df3[df3['Basket_Size']==size_sel], ci=None)
		st.write(fig)
