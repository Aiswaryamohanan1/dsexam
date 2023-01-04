from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

df = load_iris()
print(df.values())
x = df.data
y = df.target
plt.savefig("pne.png")

df1 = df.copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
y_pred = accuracy_score(x_test)
print(y_test, y_pred)

dtree = DecisionTreeClassifier()
c = df.dtree()
plt.show()