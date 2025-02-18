from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def train_knn(X_train, y_train):  
    # Creates and train kNN model
    # Knn initalised with 4 K value
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)

    return knn

def test_knn(knn, X_test, y_test):
    # Makes predictions on test set and evaluate the model
    predictions = knn.predict(X_test)
    accuracy = (predictions == y_test).mean() * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")

    return predictions