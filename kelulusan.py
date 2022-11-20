import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
import seaborn as sns
 
#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

st.write("""
# Prediksi Kelulusan Mahasiswa
### Menggunakan Beberapa Algoritma Yang Berbeda
Mana yang terbaik???
"""
)

img = Image.open('logo2.png')
st.image(img, use_column_width=False)

mhs = pd.read_csv("https://raw.githubusercontent.com/DwiAqilahP/kelulusan/main/lulus.csv")
st.write("Data kelulusan (https://raw.githubusercontent.com/DwiAqilahP/kelulusan/main/lulus.csv) ",mhs)

mhs['Lulus Cepat'].unique()

X=mhs.iloc[:,0:5].values
y=mhs.iloc[:,5].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Train and Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
st.write("Data Training", X_train)
st.write("Data Testing", X_test)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)
accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_prediction)
accuracy = accuracy_score(y_test,Y_prediction)
precision =precision_score(y_test, Y_prediction,average='micro')
recall =  recall_score(y_test, Y_prediction,average='micro')
f1 = f1_score(y_test,Y_prediction,average='micro')
print('Confusion matrix for Random Forest\n',cm)
print('accuracy_random_Forest : %.3f' %accuracy)
print('precision_random_Forest : %.3f' %precision)
print('recall_random_Forest : %.3f' %recall)
print('f1-score_random_Forest : %.3f' %f1)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test) 
accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for KNN\n',cm)
print('accuracy_KNN : %.3f' %accuracy)
print('precision_KNN : %.3f' %precision)
print('recall_KNN: %.3f' %recall)
print('f1-score_KNN : %.3f' %f1)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Naive Bayes\n',cm)
print('accuracy_Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1-score_Naive Bayes : %.3f' %f1)

# Decision Tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test) 
accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for DecisionTree\n',cm)
print('accuracy_DecisionTree: %.3f' %accuracy)
print('precision_DecisionTree: %.3f' %precision)
print('recall_DecisionTree: %.3f' %recall)
print('f1-score_DecisionTree : %.3f' %f1)

st.write("""
            #### Akurasi:"""
            )

results = pd.DataFrame({
    'Model': [ 'KNN', 
              'Random Forest',
              'Naive Bayes',
              'Decision Tree'],
    'Score': [ acc_knn,
              acc_random_forest,
              acc_gaussian,
              acc_decision_tree],
    "Accuracy_score":[accuracy_knn,
                      accuracy_rf,
                      accuracy_nb,
                      accuracy_dt
                     ]})
result_df = results.sort_values(by='Accuracy_score', ascending=False)
result_df = result_df.reset_index(drop=True)
result_df.head(9)
st.write(result_df)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['KNN', 'Random Forest','Naive Bayes','Decision Tree'],[accuracy_knn, accuracy_rf, accuracy_nb, accuracy_dt])
plt.show()
st.pyplot(fig)

# plt.subplots(figsize=(12,8))
# ax=sns.barplot(x='Model',y="Accuracy_score",data=result_df)
# labels = (result_df["Accuracy_score"])
# # add result numbers on barchart
# for i, v in enumerate(labels):
#     ax.text(i, v+1, str(v), horizontalalignment = 'center', size = 15, color = 'black')

# st.pyplot(fig)


st.write("""
            #### Pilih Metode Terbaik Menurut Prediksi Di atas :"""
            )
algoritma = st.selectbox(
'Pilih Algoritma',
('Random Forest', 'Decision Tree', 'KNN', 'Gaussian Naive Bayes')
)

ipk = st.sidebar.number_input("IPK =", 0.00)
ppd = st.sidebar.number_input("Pelatihan Pengembangan Diri =", 0)
prestasi = st.sidebar.number_input("Prestasi =", 0)
fkk = st.sidebar.number_input("Forum Komunikasi Kuliah =", 0)
ko = st.sidebar.number_input("Kegiatan Organisasi =", 0)
submit = st.sidebar.button("Submit")

if submit :
    if algoritma == 'KNN' :
        X_new = np.array([[ipk,	ppd, prestasi, fkk, ko]])
        prediksi = knn.predict(X_new)
        if prediksi == 1 :
            st.write("""# Menurut Prediksi anda Lulus Tepat Waktu""")
        else : 
            st.write("""# Menurut Prediksi anda Lulus Terlambat""")
    elif algoritma == 'Gaussian Naive Bayes' :
        X_new = np.array([[ipk,	ppd, prestasi, fkk, ko]])
        prediksi = gaussian.predict(X_new)
        if prediksi == 1 :
            st.write("""# Menurut Prediksi anda Lulus Tepat Waktu""")
        else : 
            st.write("""# Menurut Prediksi anda Lulus Terlambat""")
    elif algoritma == 'Random Forest' :
        X_new = np.array([[ipk,	ppd, prestasi, fkk, ko]])
        prediksi = random_forest.predict(X_new)
        if prediksi == 1 :
            st.write("""# Menurut Prediksi anda Lulus Tepat Waktu""")
        else : 
            st.write("""# Menurut Prediksi anda Lulus Terlambat""")
    else :
        X_new = np.array([[ipk,	ppd, prestasi, fkk, ko]])
        prediksi = decision_tree.predict(X_new)
        if prediksi == 1 :
            st.write("""# Menurut Prediksi anda Lulus Tepat Waktu""")
        else : 
            st.write("""# Menurut Prediksi anda Lulus Terlambat""")
