#ML-3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

digits=datasets.load_digits()

x=digits.data
y=digits.target

print("Data Shape :",x.shape)
print("Number of classes : ",len(set(y)))

plt.figure(figsize=(8,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(digits.images[i],cmap='gray')
    plt.title(f'Label {digits.target[i]}')
plt.show()
              
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
              
svm_clf=SVC(kernel='rbf',gamma=0.001,C=10)
svm_clf.fit(x_train_scaled,y_train)
              
y_pred=svm_clf.predict(x_test_scaled)

acc=accuracy_score(y_pred,y_test)
print(f"Model Accuracy : {acc*100:.2f}%\n")
print("Classification Report :")
print(classification_report(y_pred,y_test))

#ML-3
cm=confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix for SVM Digital Classification')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
              
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(8,8),cmap='gray')
    plt.title(f"Pred : {y_pred[i]} / True {y_test[i]}")
plt.tight_layout()
plt.show()

index=int(input("Enter index from (0 to "+str(len(y_test)-1)+")to check prediction"))
plt.imshow(x_test[index].reshape(8,8),cmap='gray')
plt.title(f"Predicted value : {y_pred[index]}, Actual Value : {y_test[index]}")

if y_pred[index]==y_test[index]:
    print("Correct Prediction")
else:
    print("Wrong Prediction")
