# Decision Tree Implementation



#  Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split Data into Train and Test
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Create and Train Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and Check Accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize Decision Tree
plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization - CODTECH Internship Task")
plt.show()
