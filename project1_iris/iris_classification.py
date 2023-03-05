# 1- Import the dataset: In this project we will use the Iris dataset from Scikit-learn:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

# 2- Explore the dataset by printing:
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

# 3- Split the dataset: Split the dataset into a training set and
# a testing set using Scikit-learn's train_test_split function:
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 4- Choose a model: For simplicity, let's use a k-nearest neighbors (KNN) classifier to predict the type of flower:
model = KNeighborsClassifier(n_neighbors=3)

# 5- Train the model: Train the model using the training data:
model.fit(X_train, y_train)

# 6- Make predictions: Use the trained model to make predictions on the test data:
y_pred = model.predict(X_test)

# 7- Evaluate the model: Evaluate the performance of the model by calculating the accuracy:
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# That's it! This is a very simple project that you can start with as a beginner! :)




