CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2
LDFLAGS = -lm

all: svm_classifier knn_classifier decision_tree_classifier cmeans_clustering neural_net_classifier

svm_classifier: svm_classifier.c ml_common.c ml_common.h
	$(CC) $(CFLAGS) svm_classifier.c ml_common.c -o svm_classifier $(LDFLAGS)

knn_classifier: knn_classifier.c ml_common.c ml_common.h
	$(CC) $(CFLAGS) knn_classifier.c ml_common.c -o knn_classifier $(LDFLAGS)

decision_tree_classifier: decision_tree_classifier.c ml_common.c ml_common.h
	$(CC) $(CFLAGS) decision_tree_classifier.c ml_common.c -o decision_tree_classifier $(LDFLAGS)

cmeans_clustering: cmeans_clustering.c ml_common.c ml_common.h
	$(CC) $(CFLAGS) cmeans_clustering.c ml_common.c -o cmeans_clustering $(LDFLAGS)

neural_net_classifier: neural_net_classifier.c ml_common.c ml_common.h
	$(CC) $(CFLAGS) neural_net_classifier.c ml_common.c -o neural_net_classifier $(LDFLAGS)

clean:
	rm -f svm_classifier knn_classifier decision_tree_classifier cmeans_clustering neural_net_classifier
