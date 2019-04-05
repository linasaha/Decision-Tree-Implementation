# Decision Tree implementation

Part1:

Problem: Implementation of a fixed-depth decision tree algorithm, that is, the input to the ID3 algorithm will include
the training data and maximum depth of the tree to be learned. 

Data Sets: The MONK’s Problems were the basis of a first international comparison of learning algorithms1

The training and test files for the three problems are named monks-X.train and monks-X.test. There are six
attributes/features (columns 2–7), and binary class labels (column 1).

Visualization: The function render dot file() can be used to generate .png images of the trees learned by both scikit-learn and your code.

Part2:

Balanced Scale UCI dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data

Used scikit-learn’s DecisionTreeClassifier to learn a decision tree using criterion=’entropy’ for depth = 1, 3, 5 for UCI dataset and studied scikit-learn’s confusion matrix() function. 

Used my own implementation of decision tree and compared the results.
.
