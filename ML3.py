import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

#load the handwritten dataset
digits = datasets.load_digits()

#features (pixel values) and labels (digits 0-9)
X= digits.data
y= digits.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(set(y)))

#visualize a few images
plt.figure(figsize=(8,4))
for i in range(8):
  plt.subplot(2, 4, i + 1))
  plt.imshow(digits.image[i], cmap='gray')
  plt.title(f"Label: {digits.target[i]}")
  plt.axis('off')
plt.show()

#Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Standardize the features
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train SVM classifier 
svm_clf = SVC(kernel='rbf', gamma=0.001, C=10)
svm_clf.fit(X_trained_scaled, y_train)

#Evaluate model
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of SVM Digit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Visualize predictions
plt.figure(figsize=(10,4))
for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
  plt.title(f"Pred: {y_pred[i]} / True: {y_test[i]}")
  plt.axis('off')
plt.tight_layout()
plt.show()
