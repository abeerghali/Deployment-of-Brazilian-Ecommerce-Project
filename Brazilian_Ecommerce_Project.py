
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

model_classification = joblib.load('Brazilian Ecommerce Classification.bkl')
model_clustering = joblib.load('Brazilian Ecommerce Clustering.bkl')
sidebar = st.sidebar
mode = sidebar.radio('Mode', ['Classification', 'Clustering'])
st.markdown("<h1 style='text-align: center; color: #ff0000;'></h1>", unsafe_allow_html=True)

if mode == "Classification":

    st.image('satisfaction.jpg')

    def predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, 
    estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate):

        prediction_classification = model_classification.predict(pd.DataFrame({'freight_value' :[freight_value], 'product_description_lenght' :[product_description_lenght], 'product_photos_qty' :[product_photos_qty], 'payment_type' :[payment_type], 'payment_installments' :[payment_installments], 'payment_value' :[payment_value], 'estimated_days' :[estimated_days], 'arrival_days' :[arrival_days], 'arrival_status' :[arrival_status], 'seller_to_carrier_status' :[seller_to_carrier_status], 'estimated_delivery_rate' :[estimated_delivery_rate], 'arrival_delivery_rate' :[arrival_delivery_rate], 'shipping_delivery_rate' :[shipping_delivery_rate]}))
        return prediction_classification

    def main():
        
        html_temp="""
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1>
                    </div>
                """
        st.markdown(html_temp,unsafe_allow_html=True)
        
        sidebar.title('Numerical Features')
        product_description_lenght = sidebar.slider('product_description_lenght', 4,3990,100)
        product_photos_qty = sidebar.slider('product_photos_qty', 1,20,1)
        payment_installments = sidebar.slider('payment_installments', 1,24,1)
        estimated_days = sidebar.slider('estimated_days', 3,60,1)
        arrival_days = sidebar.slider('arrival_days', 0,60,1)
        st.title('Categorical Features')
        payment_type = st.selectbox('payment_type', ['credit_card', 'boleto', 'voucher', 'debit_card'])
        arrival_status = st.selectbox('arrival_status', ['OnTime/Early', 'Late'])
        seller_to_carrier_status = st.selectbox('seller_to_carrier_status', ['OnTime/Early', 'Late'])
        estimated_delivery_rate = st.selectbox('estimated_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        arrival_delivery_rate = st.selectbox('arrival_delivery_rate', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        shipping_delivery_rate = st.selectbox('shipping_delivery_rate Date', ['Very Slow', 'Slow', 'Neutral', 'Fast', 'Very Fast'])
        payment_value = st.text_input('payment_value', '')
        freight_value = st.text_input('freight_value', '')
        result = ''

        if st.button('Predict_Satisfaction'):
            result = predict_satisfaction(freight_value, product_description_lenght, product_photos_qty, payment_type, payment_installments, payment_value, 
                                        estimated_days, arrival_days, arrival_status, seller_to_carrier_status, estimated_delivery_rate, arrival_delivery_rate, shipping_delivery_rate)
                                        
        st.success(f'The Customer is {result}')

    if __name__ == '__main__':
        main()

if mode == "Clustering":
    
    st.image('segmentation.jpg')

    def predict_clustering(freight_value, price, payment_value, payment_installments):

        prediction_clustering = model_clustering.predict(pd.DataFrame({'freight_value' :[freight_value], 'price' :[price], 'payment_installments' :[payment_installments], 'payment_value' :[payment_value]}))
        return prediction_clustering

    def main():

        html_temp="""
                <div style="background-color:#F5F5F5">
                <h1 style="color:#31333F;text-align:center;"> Customer Segmentation </h1>
                </div>
            """
        st.markdown(html_temp,unsafe_allow_html=True)

        payment_installments = sidebar.slider('payment_installments', 1,24,1)
        freight_value = st.text_input('freight_value', '')
        price = st.text_input('price', '')
        payment_value = st.text_input('payment_value', '')
        result_cluster = ''

        # Load dataset
        df_cluster = pd.read_csv('df_cluster.csv')

        # Define sidebar options
        num_clusters = sidebar.slider('Select number of clusters', 2, 10, 1)

        # Get cluster labels for selected number of clusters
        labels = model_clustering.predict(df_cluster)

        # Add cluster labels to dataset
        df_cluster['Cluster'] = labels
        # Define plotting function
        def plot_clusters(data, clusters, palette):
            # Define figure and axes
            fig, ax = plt.subplots(figsize=(10, 6))
            pca = PCA(n_components=2, random_state=42)
            pca_cluster = pca.fit_transform(df_cluster)

            sns.scatterplot(data=df_cluster, x=pca_cluster[:, 0], y=pca_cluster[:, 1], 
            hue= df_cluster.Cluster, palette= 'crest_r', ax=ax)
                # Set plot title and axis labels
            ax.set_title('Clusters after PCA', fontsize= 20)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')

            # Add PCA components to dataset
            df_cluster['PC1'] = pca_cluster[:, 0]
            df_cluster['PC2'] = pca_cluster[:, 1]

            # Show plot
            plt.axis('off')
            st.pyplot(fig)

        # Plot clusters using PCA components
        plot_clusters(df_cluster, df_cluster.Cluster, 'crest_r')

        if st.button('Predict_Cluster'):
            result_cluster = predict_clustering(freight_value, price, payment_value, payment_installments)
                                        
        st.success(f'Customer Cluster is {result_cluster}')

    if __name__ == '__main__':
        main()
