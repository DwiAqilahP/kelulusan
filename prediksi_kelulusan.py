import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import io 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

with st.sidebar:
    choose = option_menu("MENU", ["Home","Dataset", "Data Training", "Data Testing", "Accuracy", "My Profile"],
                         icons=['mortarboard','folder', 'card-list', 'card-checklist', 'graph-up','person lines fill'],
                         menu_icon="menu-button-wide", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#fffff"},
    }
    )

dataset = pd.read_csv("https://raw.githubusercontent.com/DwiAqilahP/kelulusan/main/data_lulus_tepat_waktu.csv")
# dataset.shape
# dataset.info()
dataset.isnull().values.any()
ket = {"Ya" : 0, "Tidak" : 1}
dataset["tepat"] = dataset["tepat"].map(ket)
dataset['tepat'].value_counts()

# print('Ya', round(dataset['tepat'].value_counts()[0]/len(dataset) * 100,2), '% of the dataset')
# print('Tidak', round(dataset['tepat'].value_counts()[1]/len(dataset) * 100,2), '% of the dataset')

# colors = ["#0101DF", "#DF0101"]

# sns.countplot('tepat', data=dataset, palette=colors)
# plt.title('Class Distributions \n (1: YA || 0: Tidak)', fontsize=14)

# # Dataset tidak balance, jomplang bet perbedaannya.
# kita Resampling datanya. Resampling ada 2 :

# Random Oversampling: Randomly duplicate examples in the minority class.
# Random Undersampling: Randomly delete examples in the majority class.
# disini kita menggunakan random over sampling

# Class count
count_class_0, count_class_1 = dataset.tepat.value_counts()

# Divide by class
df_class_0 = dataset[dataset['tepat'] == 0]
df_class_1 = dataset[dataset['tepat'] == 1]

df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print('Random over-sampling:')
print(df_test_over.tepat.value_counts())

df_test_over.tepat.value_counts().plot(kind='bar', title='Count (tepat)');
df_test_over.tepat.value_counts()
dataset.head()
x = df_test_over.iloc[:, :-1].values
y = df_test_over.iloc[:, -1].values

from sklearn.model_selection import train_test_split

validation_size = 0.20
num_trees = 100
seed = 5
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=validation_size, random_state=seed)

rf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)
acc_rf = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
predictions = gnb.predict(x_test)
acc_gnb = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
acc_knn = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
predictions = dt.predict(x_test)
acc_dt = accuracy_score(y_test, predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

logo = Image.open('logo2.png')
# st.image(logo, use_column_width=False)
if choose == "Home":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Prediksi Kelulusan Mahasiswa</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    
    st.write("""
                #### Pilih Metode Terbaik:"""
                )
    algoritma = st.selectbox(
    '*Pilih Algoritma dengan cek Accuracy terbaik',
    ('Random Forest', 'Decision Tree', 'KNN', 'Gaussian Naive Bayes')
    )      
    nama = st.text_input("Nama Mahasiswa =")
    ip1 = st.number_input("Nilai IP Smt.1 =", 0.00)
    ip2 = st.number_input("Nilai IP Smt.2 =", 0.00)
    ip3 = st.number_input("Nilai IP Smt.3=", 0.00)
    ip4 = st.number_input("Nilai IP Smt.4=", 0.00)
    submit = st.button("Submit")
    
    if submit :
        if algoritma == "Random Forest" :           
            prediction_rf=rf.predict([[ip1,ip2,ip3,ip4]])
            score1 = rf.score(x_test, y_test)
            if prediction_rf[0] == 1:
                pred = ("Tepat Waktu")
            else:
                pred = ("Tidak Tepat Waktu")
            st.write('Prediksi :',nama ,"Lulus",pred)
            st.write("Test score: {0:.2f} %".format(100 * score1)) 
        elif algoritma == "Decision Tree" :
            prediction_dt=dt.predict([[ip1,ip2,ip3,ip4]])
            score1 = dt.score(x_test, y_test)
            if prediction_dt[0] == 1:
                pred = "Tepat Waktu"
            else:
                pred = "Tidak Tepat Waktu"
            st.write('Prediksi :',nama,"Lulus",pred)
            st.write("Test score: {0:.2f} %".format(100 * score1)) 
        elif algoritma == "KNN" :
            prediction_knn=knn.predict([[ip1,ip2,ip3,ip4]])
            score1 = knn.score(x_test, y_test)
            if prediction_knn[0] == 1:
                pred = "Tepat Waktu"
            else:
                pred = "Tidak Tepat Waktu"
            st.write('Prediksi :',nama,"Lulus",pred)
            st.write("Test score: {0:.2f} %".format(100 * score1)) 
        if algoritma == "Gaussian Naive Bayes" :
            prediction_gnb=gnb.predict([[ip1,ip2,ip3,ip4]])
            score1 = gnb.score(x_test, y_test)
            if prediction_gnb[0] == 1:
                pred = "Tepat Waktu"
            else:
                pred = "Tidak Tepat Waktu"
            st.write('Prediksi :',nama,"Lulus",pred)
            st.write("Test score: {0:.2f} %".format(100 * score1)) 
    
elif choose == "Dataset" :
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Dataset</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    st.write("Data kelulusan (https://www.kaggle.com/code/baladikaalhariri/klasifikasi-dengan-random-forest-97-test/data) ",dataset)


    
elif choose == "Data Training" :
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Data Training</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    st.write(x_train)
    
    
elif choose == "Data Testing" :
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font"> Data Testing</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )
    st.write(x_test)

elif choose == "Accuracy" :
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Accuracy</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(logo, width=130 )

    results = pd.DataFrame({
        'Model': [ 'KNN', 
                'Random Forest',
                'Naive Bayes',
                'Decision Tree'],
        'Accuracy': [ acc_knn,
                acc_rf,
                acc_gnb,
                acc_dt],})
    result_df = results.sort_values(by='Accuracy', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['KNN', 'Random Forest','Naive Bayes','Decision Tree'],[acc_knn, acc_rf, acc_gnb, acc_dt])
    plt.show()
    st.pyplot(fig)
        
else :
    gambar = Image.open('aqila.jpg')
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #00000;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">My Profile</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(gambar, width=130 )
    st.write ("Nama",':', "Dwi Aqilah Pradita")
    st.write ("NIM",':', "200411100044")
    st.write ("Kelas",':', "Penambangan Data B")
    st.write ("E-mail",':', "dwipradita56@gmail.com")
    
    
    
    
    