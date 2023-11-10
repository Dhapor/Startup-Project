import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression


data = pd.read_csv('startUp(1).csv')
# print(data.head())


dx = data.copy()
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# for i in dx.columns:
#     if dx[i].dtypes != 'O':
#         dx[i] = scaler.fit_transform(dx[[i]])

# dx.head()


# #ENCODING THE NORMINAL DATA( transforming from categorical to numerical)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in dx.columns:
    if dx[i].dtypes == 'O':
        dx[i] = encoder.fit_transform(dx[i])
dx.head()


# # drop state since it doesnt satisfy the assumption of linearity
dx.drop(['Unnamed: 0', 'State'], axis = 1, inplace=True)
# dx.columns



# # assumption of multicolinearity
# plt.figure(figsize = (19,3))
# sns.heatmap(dx.corr(), annot=True, cmap='BuPu')


# #Test and train split
x = dx.drop('Profit', axis = 1)
y = dx.Profit

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.80, random_state = 210806077)
# print(f'xtrain: {xtrain.shape}')
# print(f'ytrain: {ytrain.shape}')
# print(f'xtest: {xtest.shape}')
# print(f'ytest: {ytest.shape}')


# train_set = pd.concat([xtrain, ytrain], axis = 1)
# test_set = pd.concat([xtest, ytest], axis = 1)

# print(f"\t\tTrain DataSet")
# print(train_set.head())
# print(f"\n\tTest DataSet")
# print(test_set.head())


# # ------------MODELLING---------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

Lin_reg = LinearRegression()

# # Create a linear regression model
Lin_reg.fit(xtrain, ytrain)

# #---------- cross validation (to see how well the model will predict after being trained) -----------
# cross_validate = Lin_reg.predict(xtrain) # ----- this code predicts the y
# score = r2_score(cross_validate, ytrain) # ----- this give the accuracy percentage, if it perfomrs well, it's underfitting and if it doesn't it's overfitting.
# print(f"The Cross Validation Score is: {score.round(2)}")


# test_prediction = Lin_reg.predict(xtest)
# score = r2_score(test_prediction, ytest)
# print(f"The Cross Validation Score is: {score.round(2)}")


# # COMPARING THE ACTUAL RESULTS TO THE PREDICTED RESULTS
# pd.DataFrame({'Actual': [i for i in ytest], 'Prediction': [i for i in test_prediction]})


# print(f"Intercept of the model: {Lin_reg.intercept_}\n")
# print(f"Coefficient of the model: {Lin_reg.coef_}\n")

import pickle


# # save model
pickle.dump(Lin_reg, open('StartUp_Model.pkl', "wb"))

st.markdown("<h1 style = 'color: #BEADFA; text-align: center; font-family:montserrat'>START-UP PROJECT</h1>",unsafe_allow_html=True)
st.markdown("<h3 style = 'margin: -15px; color: #BEADFA; text-align: center; font-family:montserrat'>Start-up Project Built By Orpheaus</h3>",unsafe_allow_html=True)

st.markdown("<br></br>", unsafe_allow_html=True)
st.image('3545758.png',  width = 650)


st.markdown("<br></br>", unsafe_allow_html=True)

st.markdown("<h3 style = 'margin: -15px; color: #BEADFA; text-align: center; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)

st.markdown("<p>By analyzing a diverse set of parameters, including market trends, competitive landscape, financial indicators, and operational strategies, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success.</p>", unsafe_allow_html = True)

data = pd.read_csv('startUp(1).csv')
st.write(data.head())
st.sidebar.image('profile image.jpg')
st.sidebar.markdown('<br>', unsafe_allow_html= True)

input_type = st.sidebar.radio("Select Your Preferred Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    research = st.sidebar.slider("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.slider("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.slider("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())
else:
    st.sidebar.header('Input Your Information')
    research = st.sidebar.number_input("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.number_input("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.number_input("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())


st.header('Input Values')
input_variables = pd.DataFrame([{'R&D Spend':research, 'Administration': admin, 'Marketing Spend': mkt_spend}])
st.write(input_variables)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# for i in input_variables.columns:
#     input_variables[i] = scaler.transform(input_variables[[i]])

# print(f'\t\t\n\nPredicted Output: {Lin_reg.predict([[-1.134305,	1.206419,	-1.509074]])}')
st.write(input_variables)

import pickle
model = pickle.load(open('StartUp_Model.pkl', "rb"))

tab1, tab2 = st.tabs(['Modelling', 'Interpretation'])
with tab1:
    if st.button('Press to predict'):
        # profit = Lin_reg.predict(input_variables)
        st.toast('Profitability Predicted')
        st.image('pngwing.com (1).png', width = 200)
        st.success('Predicted. pls check the Interpretation Tab for Interpretation')


with tab2:
    st.subheader('Model Interpretation')
    profit = Lin_reg.predict(input_variables)
    st.success(f'Predicted Profit is: {profit}')
    
    st.write(f"Profit = {Lin_reg.intercept_.round(2)} + {Lin_reg.coef_[0].round(2)} R&D Spend + {Lin_reg.coef_[1].round(2)} Administration + {Lin_reg.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {Lin_reg.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${Lin_reg.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${Lin_reg.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")