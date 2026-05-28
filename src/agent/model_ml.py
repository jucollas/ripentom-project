from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


models = {
    "SVM (RBF)":           SVC(kernel="rbf", C=10, gamma="scale"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
}