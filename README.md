# Classification Algorithms in C

This project implements multiple classification approaches in C on small structured datasets. The main goal is to compare how different algorithms learn, predict, and are evaluated while following one shared reporting pattern through `ml_common.h` and `ml_common.c`.

## Datasets Used

### 1) Labelled human dataset
This is the core labelled dataset used for the human body classification task. It contains two numerical features and one label.

Files:
- `Data/train.csv`
- `Data/validate.csv`
- `Data/test.csv`

Structure:
- `height`
- `weight`
- `label`

Label goal:
- `FIT`
- `OBESE`

Purpose:
- `KNN`, `SVM`, and `Decision Tree` train and evaluate directly on this dataset.
- `Fuzzy C-Means` does not train on the labels, but these labelled files are still used afterward as a reference to compare clustering output with known class labels.

Split goal:
- `train.csv` is used for model fitting.
- `validate.csv` is used for model selection or hyperparameter choice where needed.
- `test.csv` is used for final held-out evaluation.

### 2) Unlabelled human dataset
This dataset is the unlabelled version of the human task and is used for clustering.

File:
- `Data/dataset_unlabelled.csv`

Structure:
- `height_cm`
- `weight_kg`

Goal:
- It is used by `cmeans_clustering.c` for unsupervised clustering.
- It follows the same human feature structure as the labelled data, but removes the label column.
- The cluster assignment is later mapped to `FIT` or `OBESE` based on the cluster center characteristics.

### 3) Fruit dataset
This dataset is used by the neural-network-based executable.

File:
- `Data/fruit_dataset.csv`

Structure:
- `shape`
- `texture`
- `weight`
- `label`

Label goal:
- `ORANGE`
- `APPLE`

Feature pattern:
- The feature values are bipolar and are stored as `+1` or `-1`.
- This makes the dataset suitable for Perceptron, Hamming, and Hopfield style pattern classification.

Purpose:
- It is used by `neural_net_classifier.c`.
- The executable creates train, validation, and test splits internally in a deterministic way.

## How to Compile & Run

From the project root, compile everything with:

```bash
make
```

This produces the following executables:
- `knn_classifier`
- `svm_classifier`
- `decision_tree_classifier`
- `cmeans_clustering`
- `neural_net_classifier`

Run them with default dataset paths:

```bash
./knn_classifier
./svm_classifier
./decision_tree_classifier
./cmeans_clustering
./neural_net_classifier
```

Run them with explicit paths if needed:

```bash
./knn_classifier Data/train.csv Data/validate.csv Data/test.csv
./svm_classifier Data/train.csv Data/validate.csv Data/test.csv
./decision_tree_classifier Data/train.csv Data/validate.csv Data/test.csv
./cmeans_clustering Data/dataset_unlabelled.csv Data/train.csv Data/validate.csv Data/test.csv
./neural_net_classifier Data/fruit_dataset.csv
```

All default file paths use the `Data/` prefix.

## Shared Evaluation Pattern

All algorithms follow the shared evaluation pattern defined in `ml_common.h` and `ml_common.c`. The report format is intentionally consistent across the project.

The shared evaluation metrics are:
- Accuracy
- Precision
- Recall
- F1-score
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Class support

For the human task, the reports are printed with `FIT` and `OBESE`.
For the fruit task, the same report pattern is reused with `ORANGE` and `APPLE`.

## Algorithms

### K-Nearest Neighbors (KNN)
KNN predicts a sample by checking the labels of the nearest training points. It is a distance-based method and does not build an explicit mathematical model during training.

Building process:
- Reads the labelled human training, validation, and test sets.
- Fits a min-max scaler on the training set.
- Scales train, validation, and test using the same scaler.
- Tries odd values of `k` from `1` up to `MAX_K_TO_TRY`.
- Selects the best `k` using validation accuracy.
- Uses leave-one-out style prediction for train evaluation.
- Uses the chosen `k` for validation, test, and interactive prediction.

Evaluation for KNN:
- Train, validation, and test reports are printed through `ml_common`.
- The train report is intentionally stricter because each training sample is predicted without using itself as a neighbor.

Short note:
- KNN is simple, intuitive, and useful when similar samples should have similar labels.
- Its quality depends strongly on feature scaling and the choice of `k`.

### Support Vector Machine (SVM)
SVM builds a separating hyperplane between the two classes. In this project it is a linear classifier trained on the human dataset.

Building process:
- Reads the labelled human training, validation, and test sets.
- Fits a min-max scaler on the training set.
- Scales all splits using the training scaler.
- Trains a linear SVM using the configured learning rate, regularization, and epochs from `ml_common.h`.
- Produces predictions for train, validation, test, and interactive user input.
- Prints the learned hyperplane.

Evaluation for SVM:
- Train, validation, and test reports are printed with the shared `ml_common` report structure.
- It uses the same binary evaluation style as KNN and Decision Tree.

Short note:
- SVM is good when the classes are separable by a strong boundary.
- It gives a compact model and usually behaves more smoothly than raw instance-based methods.

### Decision Tree
Decision Tree classifies by asking a sequence of feature-based questions. Each split tries to separate the data more cleanly.

Building process:
- Reads the labelled human training, validation, and test sets.
- Uses the raw feature values directly.
- Searches for the best split using Gini impurity.
- Tunes `max_depth` and `min_samples_split` using validation accuracy.
- Trains the final tree on the training set.
- Prints the learned tree structure.
- Predicts on train, validation, test, and interactive input.

Evaluation for Decision Tree:
- Train, validation, and test reports are printed through the same `ml_common` evaluation functions.
- The evaluation stays directly comparable with KNN and SVM because the output labels remain `FIT` and `OBESE`.

Short note:
- Decision Trees are easy to explain because the learned rule path can be read directly.
- They work well on small tabular data, but can become unstable if the split choices change too much.

### Fuzzy C-Means Clustering
Fuzzy C-Means is an unsupervised clustering method. Instead of assigning each sample fully to one cluster at first, it keeps soft membership values and gradually updates them.

Building process:
- Reads `Data/dataset_unlabelled.csv`.
- Applies min-max normalization to the unlabelled data.
- Initializes cluster memberships deterministically.
- Repeats center update and membership update steps until convergence or maximum iterations.
- Uses two clusters.
- Maps the heavier cluster to `OBESE` and the other cluster to `FIT`.
- Prints cluster centers, cluster counts, and interactive predictions.

Evaluation for Fuzzy C-Means:
- The clustering itself is unsupervised.
- For comparison with the supervised algorithms, the program builds a reference agreement report against the combined labelled human rows.
- The printed metrics still use the shared `ml_common` format so the output remains coherent with the rest of the project.

Short note:
- Fuzzy C-Means is helpful when boundaries are not perfectly sharp.
- It captures gradual membership rather than hard class assignment at the start.

### Neural Network Executable
The neural network executable contains three different pattern-classification approaches for the fruit dataset: Perceptron, Hamming Network, and Hopfield Network.

Building process:
- Reads `Data/fruit_dataset.csv`.
- Splits the fruit dataset internally into train, validation, and test sets.
- Computes train-set class prototypes where needed.
- Runs three models inside the same executable.
- Prints separate reports for each model.
- Reuses the shared metric pattern with custom class names `ORANGE` and `APPLE`.

#### Perceptron
Perceptron is a simple linear neural classifier. It learns weights and a bias from labelled examples.

Building process:
- Starts with zero weights and bias.
- Iteratively updates the weights on training mistakes.
- Monitors validation accuracy and keeps the best observed model.
- Uses the selected model for train, validation, and test reporting.

Evaluation for Perceptron:
- Accuracy, Precision, Recall, F1-score, Macro, Weighted, and Support are printed through the shared report functions.

Short note:
- Perceptron is a strong first neural model for binary classification.
- It is most suitable when the classes are close to linearly separable.

#### Hamming Network
Hamming Network is a prototype-based neural classifier. It compares the input with stored class prototypes and then resolves competition between candidate classes.

Building process:
- Computes one prototype for `ORANGE` and one prototype for `APPLE` from the training split.
- Builds feedforward weights from the prototypes.
- Uses a competitive MAXNET-style inhibition process to decide the winner.

Evaluation for Hamming Network:
- The same shared evaluation report is printed for train, validation, and test.
- Because it is prototype-based, its quality depends on how representative the stored prototypes are.

Short note:
- Hamming Network is useful when each class can be represented by a strong reference pattern.
- It is simple and fast once the prototypes are fixed.

#### Hopfield Network
Hopfield Network is an associative memory model. It tries to recall a stable stored pattern from an input pattern.

Building process:
- Builds a Hebbian weight matrix from the training prototypes.
- Iteratively updates the internal state until it stabilizes or reaches the maximum number of iterations.
- Checks whether the recalled state matches a stored prototype exactly.
- If not, it treats the state as spurious and maps it to the nearest prototype for reporting.

Evaluation for Hopfield Network:
- The same shared evaluation metrics are printed for train, validation, and test.
- The program also prints spurious recall counts because Hopfield networks can converge to states that were not stored exactly.

Short note:
- Hopfield is useful for associative recall and pattern memory experiments.
- It is more interesting as a memory model than as a general-purpose modern classifier.

## Overall Comparison Table

### Human-data algorithms

| Algorithm | Learning style | Uses labels for training | Main data used | Main preprocessing | Model selection / build style | Evaluation style | Main strength | Main limitation |
|---|---|---:|---|---|---|---|---|---|
| KNN | Instance-based supervised | Yes | `train/validate/test.csv` | Min-max scaling | Chooses best odd `k` by validation accuracy | Shared `ml_common` report on train/valid/test | Very simple and intuitive | Sensitive to scale and local noise |
| SVM | Linear supervised | Yes | `train/validate/test.csv` | Min-max scaling | Learns one separating hyperplane with configured training loop | Shared `ml_common` report on train/valid/test | Compact decision boundary | Limited if separation is strongly non-linear |
| Decision Tree | Rule-based supervised | Yes | `train/validate/test.csv` | No scaling required | Chooses splits by Gini and tunes tree size on validation | Shared `ml_common` report on train/valid/test | Easy to interpret | Can vary with split decisions |
| Fuzzy C-Means | Unsupervised clustering | No | `dataset_unlabelled.csv` | Min-max scaling | Iteratively updates memberships and cluster centers | Shared report used as reference agreement against labelled rows | Handles soft membership naturally | Cluster labels must be mapped after clustering |

### Neural-network-related algorithms

| Algorithm | Learning style | Main data used | Core representation | Build style | Evaluation style | Main strength | Main limitation |
|---|---|---|---|---|---|---|---|
| Perceptron | Supervised linear neural classifier | `fruit_dataset.csv` | Learned weights and bias | Error-driven updates with best validation checkpoint | Shared custom-name report on train/valid/test | Simple and effective for linearly separable patterns | Cannot represent complex non-linear boundaries well |
| Hamming Network | Prototype-based neural classifier | `fruit_dataset.csv` | Stored class prototypes | Prototype comparison plus competitive inhibition | Shared custom-name report on train/valid/test | Fast and direct prototype matching | Depends strongly on prototype quality |
| Hopfield Network | Associative memory network | `fruit_dataset.csv` | Stored stable patterns | Hebbian memory matrix plus iterative recall | Shared custom-name report on train/valid/test plus spurious recall counts | Good for memory-style pattern recall | Can converge to spurious states |

## Short Notes for Every Algorithm

### KNN
- Best when nearby points are expected to share the same class.
- Needs careful scaling.
- No real training phase, but prediction can be slower because it compares against stored training points.

### SVM
- Best when a clean separating boundary exists.
- Produces a compact model using weights and bias.
- Usually easier to deploy than memory-based methods.

### Decision Tree
- Best when the relationship can be explained as a sequence of simple rules.
- Very readable and interpretable.
- Can overfit if tree growth is not controlled well.

### Fuzzy C-Means
- Best when samples may partially belong to more than one group.
- Useful for exploratory structure discovery.
- Not fully label-aware during training.

### Perceptron
- Best as a first neural baseline for two-class pattern separation.
- Easy to implement and understand.
- Limited to simple linear boundaries.

### Hamming Network
- Best when each class has a meaningful representative pattern.
- Computationally light after the prototypes are built.
- Less flexible when class variation is large.

### Hopfield Network
- Best for demonstrating associative memory and stable state recall.
- Interesting for pattern restoration experiments.
- Not as general or as robust as modern supervised neural classifiers.

## Author

**Md. Kais**
Dept. of CSE 
University Of Chittagong 
Course: CSE 816 (Machine Learning Lab)
