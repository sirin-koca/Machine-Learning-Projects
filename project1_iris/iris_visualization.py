"""In the first few lines of the code, we import the necessary libraries (load_iris() from scikit-learn,
train_test_split() from scikit-learn, sns from Seaborn, plt from Matplotlib, and KNeighborsClassifier from
scikit-learn). We then load the Iris dataset into the iris variable using load_iris(), and extract the feature matrix
X and target vector y from the iris variable.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

"""Next, we split the dataset into training and testing sets using train_test_split() from scikit-learn, with 80% of 
the data used for training and 20% for testing. We then choose a k-nearest neighbors classifier with k=3 and train it 
on the training data using fit()."""
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model and train it on the training data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

"""After that, we make predictions on the test data using the predict() method of the trained model, and store the 
predicted class labels in the y_pred variable."""
# Make predictions on the test data
y_pred = model.predict(X_test)

"""Finally, we use Seaborn's scatterplot() function to visualize the model's predictions, with the predicted class 
labels represented by different marker shapes and the actual class labels represented by different marker styles. We 
then use plt.show() to display the plot."""
# Visualize the model's predictions
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, style=y_test, markers=['o', 'X', 's'], s=100, alpha=0.8)

# Show the plots
plt.show()
