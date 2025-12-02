# AI-ML-Projects
My AI/ML learning projects (Python, Pandas, TensorFlow)

Install Python 3.x, pip, pandas, tensorflow, scikit-learn, matplotlib, seaborn, VS Code(editor)

## python -m pip install tensorflow scikit-learn pandas matplotlib seaborn

## Day 1: Basics in Python
This project demonstrates basic Python concepts, including:

- Printing messages
- Taking user input
- Variables and arithmetic
- Conditional statements (`if`, `elif`, `else`)

## Day 2: Loops in Python

This project demonstrates basic loops in Python:
- `for` loop to calculate squares
- `while` loop countdown
- Summing numbers using loops


## Day 3: Functions in Python
This project demonstrates how to define and use **functions in Python**.  

### Concepts Covered
- Defining functions with `def`
- Passing arguments and returning values
- Reusing code with functions
- Combining functions with loops and conditionals

# Day 4: Python Lists and Dictionaries
This project demonstrates **basic Python data structures**:

### Concepts Covered
- Lists: creating, looping, appending, removing, and summing elements
- Dictionaries: creating, adding keys, accessing values, and looping
- Nested lists (2D lists) and list comprehension


# Day 5: Loops, List Comprehension, and Input

This project demonstrates **advanced loops and list comprehension in Python**.

### Concepts Covered
- Nested loops (e.g., multiplication tables)
- List comprehension for filtering and transforming
- Flattening nested lists (2D lists)
- Iterating through strings and lists

# Day 6: Python String Manipulation

This project demonstrates **basic string operations in Python**.

### Concepts Covered
- Counting vowels in a string
- Reversing a string using slicing
- Word frequency count
- Trimming, lowercasing, and replacing substrings

# Day 7: File Handling with CSV

This project demonstrates **reading, writing, and analyzing CSV files in Python**.

### Concepts Covered
- Writing CSV files using `csv.writer`
- Reading CSV files using `csv.reader`
- Reading CSV files into dictionaries using `csv.DictReader`
- Basic analysis: counting and filtering
- Finding longest names and students by city

# Day 8: NumPy Basics

This project demonstrates **numerical operations using NumPy**, essential for AI/ML.

### Concepts Covered
- Creating NumPy arrays
- Basic statistics: mean, sum, max, min, standard deviation
- Array operations
- Slicing arrays
- Working with 2D arrays (matrices) and dot product

# Day 9: Pandas Basics

This project demonstrates **data analysis using Pandas**, a key library for AI/ML.

### Concepts Covered
- Reading CSV files with `pd.read_csv`
- Accessing columns and filtering rows
- Adding new columns with `apply()`
- Summary statistics (`describe()`)
- Counting occurrences with `value_counts()`

# Day 10: Data Visualization with Matplotlib

This project demonstrates **visualizing data using Matplotlib**.

### Concepts Covered
- Bar charts
- Histograms
- Pie charts
- Using data from Pandas DataFrame

# Day 11: NumPy for ML Data Preparation

This project demonstrates **preparing numerical data for machine learning** using NumPy.

### Concepts Covered
- Creating NumPy arrays
- Combining features into a matrix
- Normalizing features (Min-Max scaling)
- Creating target labels

# Day 12: Basic Machine Learning with scikit-learn

This project demonstrates **building a simple ML model using scikit-learn**.

### Concepts Covered
- Preparing feature matrices and target labels
- Splitting data into train/test sets
- Feature scaling (Standardization)
- Training a K-Nearest Neighbors (KNN) classifier
- Evaluating model accuracy
- Making predictions

# Day 13: Simple Neural Network with TensorFlow/Keras

This project demonstrates **building a basic neural network** for binary classification.

### Concepts Covered
- Feature scaling using StandardScaler
- Creating a feed-forward neural network with Keras Sequential API
- Using Dense layers and activation functions (ReLU, Sigmoid)
- Compiling and training the model
- Evaluating model accuracy and making predictions

# Day 14: Multi-class Neural Network with TensorFlow/Keras

This project demonstrates **building a neural network for multi-class classification**.

### Concepts Covered
- Feature scaling using StandardScaler
- One-hot encoding target labels for multi-class output
- Creating a feed-forward neural network with Keras Sequential API
- Using Dense layers and activation functions (ReLU, Softmax)
- Compiling and training the model with categorical cross-entropy
- Evaluating model accuracy and making predictions

# Day 15: Neural Network Predictions and Activations

This project demonstrates **how to inspect neural network outputs** for multi-class classification.

### Concepts Covered
- Feature scaling
- One-hot encoding for multi-class targets
- Building a neural network (Sequential API)
- Softmax output: probabilities for each class
- Mapping probabilities to predicted class
- Comparing predicted vs actual labels

# Day 16

- Combine multiple features (numerical + categorical)
- Encode real-world text data for ML
- Scale data for neural networks
- Create multi-class labels
- Build a neural network with softmax output
- Train & evaluate a real ML model

# Day 17
- Why splitting data is essential
- How to evaluate ML properly
- What accuracy hides
- What precision, recall, and F1-score truly mean
- How to read a confusion matrix
- How to generate a classification report
- Real ML workflow used in industry

# Day 18
- Train/Test Split Is Not Enough
- k-Fold Cross-Validation
- cross_val_score()
- GridSearchCV for Hyperparameter Tuning
- Output: Best Params & Best Accuracy

# Day 19
- Logistic Regression is a classification algorithm (not regression).
- It uses the sigmoid function to convert numbers into probabilities (0–1).
- Predictions are made by comparing probability against a threshold (0.5).
- I trained a logistic regression classifier to detect “Minor vs Not-Minor”.
- I visualized the sigmoid curve + decision boundary, which shows exactly how the model separates the two classes.

# Day 20
- Learned the concept of linear regression and how models learn slope (m) and intercept (b).
- Studied mean squared error (MSE) as the loss function used for linear regression.
- Implemented gradient descent manually to update parameters.
- Observed how loss decreases over training, proving that learning is happening.
- Built a full training loop and plotted the loss curve.

# Day 21
- Feature scaling prevents some features from dominating others in ML models.
- Min-Max Scaling rescales data to 0–1 range, keeping distributions intact.
- Standardization rescales data to mean=0 and std=1, preserving outliers.
- Scaling is essential for gradient-based models (NN, Logistic Regression) and distance-based models (KNN, SVM).
- Always fit the scaler on training data and apply the same transformation to test

# Day 22
- KNN predicts labels based on the majority class of k nearest neighbors.
- Euclidean distance is the most common metric for finding neighbors.
- Feature scaling (standardization) is essential; unscaled features skew distance.
- Small k → overfitting; large k → underfitting.
- Evaluated model with accuracy, confusion matrix, classification report, and visualized decision boundary.

# Day 23
- Naive Bayes is a probabilistic classifier based on Bayes’ theorem.
- Assumes features are independent (“naive” assumption).
- GaussianNB is used for continuous numeric features; MultinomialNB/BernoulliNB for discrete/binary features.
- Predictions are made by computing probabilities and selecting the class with the highest probability.
- Evaluated model with accuracy, confusion matrix, classification report, and visualized decision boundaries.