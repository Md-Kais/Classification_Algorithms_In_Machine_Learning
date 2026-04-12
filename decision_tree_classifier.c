#include "ml_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DT_MIN_DEPTH_TO_TRY 2
#define DT_MAX_DEPTH_TO_TRY 8

static const int MIN_SAMPLES_OPTIONS[] = {2, 4, 6, 8};

typedef struct TreeNode
{
    int is_leaf;
    int predicted_label;
    int feature_index;
    double threshold;
    double gini;
    int sample_count;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

typedef struct
{
    int max_depth;
    int min_samples_split;
} TreeHyperParams;

typedef struct
{
    int feature_index;
    double threshold;
    double score;
    int left_count;
    int right_count;
} SplitResult;

static int compareDouble(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

static double giniForCounts(int fit_count, int obese_count)
{
    int total = fit_count + obese_count;
    if (total == 0)
        return 0.0;

    double p_fit = safeDivide((double)fit_count, (double)total);
    double p_obese = safeDivide((double)obese_count, (double)total);
    return 1.0 - (p_fit * p_fit + p_obese * p_obese);
}

static double giniForSubset(const Dataset *ds, const int *indices, int count)
{
    int fit_count = 0;
    int obese_count = 0;

    for (int i = 0; i < count; i++)
    {
        if (ds->data[indices[i]].label == LABEL_FIT)
            fit_count++;
        else
            obese_count++;
    }

    return giniForCounts(fit_count, obese_count);
}

static int majorityLabel(const Dataset *ds, const int *indices, int count)
{
    int fit_count = 0;
    int obese_count = 0;

    for (int i = 0; i < count; i++)
    {
        if (ds->data[indices[i]].label == LABEL_FIT)
            fit_count++;
        else
            obese_count++;
    }

    return (fit_count >= obese_count) ? LABEL_FIT : LABEL_OBESE;
}

static int isPureSubset(const Dataset *ds, const int *indices, int count)
{
    if (count <= 1)
        return 1;
    int first_label = ds->data[indices[0]].label;
    for (int i = 1; i < count; i++)
    {
        if (ds->data[indices[i]].label != first_label)
            return 0;
    }
    return 1;
}

static void partitionIndices(const Dataset *ds,
                             const int *indices,
                             int count,
                             int feature_index,
                             double threshold,
                             int *left_indices,
                             int *left_count,
                             int *right_indices,
                             int *right_count)
{
    *left_count = 0;
    *right_count = 0;

    for (int i = 0; i < count; i++)
    {
        int idx = indices[i];
        if (ds->data[idx].features[feature_index] <= threshold)
        {
            left_indices[(*left_count)++] = idx;
        }
        else
        {
            right_indices[(*right_count)++] = idx;
        }
    }
}

static SplitResult findBestSplit(const Dataset *ds, const int *indices, int count)
{
    SplitResult best;
    best.feature_index = -1;
    best.threshold = 0.0;
    best.score = 1e100;
    best.left_count = 0;
    best.right_count = 0;

    if (count <= 1)
        return best;

    int *left_indices = (int *)malloc(count * sizeof(int));
    int *right_indices = (int *)malloc(count * sizeof(int));
    double *values = (double *)malloc(count * sizeof(double));

    if (!left_indices || !right_indices || !values)
    {
        fprintf(stderr, "FATAL: malloc failed in findBestSplit\n");
        free(left_indices);
        free(right_indices);
        free(values);
        exit(EXIT_FAILURE);
    }

    for (int f = 0; f < NUM_FEATURES; f++)
    {
        for (int i = 0; i < count; i++)
        {
            values[i] = ds->data[indices[i]].features[f];
        }
        qsort(values, count, sizeof(double), compareDouble);

        for (int i = 0; i < count - 1; i++)
        {
            if (fabs(values[i + 1] - values[i]) < 1e-12)
                continue;
            double threshold = 0.5 * (values[i] + values[i + 1]);

            int left_count = 0;
            int right_count = 0;
            partitionIndices(ds, indices, count, f, threshold,
                             left_indices, &left_count,
                             right_indices, &right_count);

            if (left_count == 0 || right_count == 0)
                continue;

            double left_gini = giniForSubset(ds, left_indices, left_count);
            double right_gini = giniForSubset(ds, right_indices, right_count);
            double weighted = safeDivide((double)left_count, (double)count) * left_gini +
                              safeDivide((double)right_count, (double)count) * right_gini;

            if (weighted + 1e-12 < best.score)
            {
                best.feature_index = f;
                best.threshold = threshold;
                best.score = weighted;
                best.left_count = left_count;
                best.right_count = right_count;
            }
        }
    }

    free(left_indices);
    free(right_indices);
    free(values);
    return best;
}

static TreeNode *buildTreeRecursive(const Dataset *ds,
                                    const int *indices,
                                    int count,
                                    int depth,
                                    const TreeHyperParams *hp)
{
    TreeNode *node = (TreeNode *)calloc(1, sizeof(TreeNode));
    if (!node)
    {
        fprintf(stderr, "FATAL: calloc failed in buildTreeRecursive\n");
        exit(EXIT_FAILURE);
    }

    node->sample_count = count;
    node->predicted_label = majorityLabel(ds, indices, count);
    node->gini = giniForSubset(ds, indices, count);

    if (count == 0 ||
        depth >= hp->max_depth ||
        count < hp->min_samples_split ||
        isPureSubset(ds, indices, count) ||
        node->gini <= 1e-12)
    {
        node->is_leaf = 1;
        return node;
    }

    SplitResult best = findBestSplit(ds, indices, count);
    if (best.feature_index < 0)
    {
        node->is_leaf = 1;
        return node;
    }

    int *left_indices = (int *)malloc(count * sizeof(int));
    int *right_indices = (int *)malloc(count * sizeof(int));
    if (!left_indices || !right_indices)
    {
        fprintf(stderr, "FATAL: malloc failed while splitting tree node\n");
        free(left_indices);
        free(right_indices);
        free(node);
        exit(EXIT_FAILURE);
    }

    int left_count = 0;
    int right_count = 0;
    partitionIndices(ds, indices, count, best.feature_index, best.threshold,
                     left_indices, &left_count,
                     right_indices, &right_count);

    if (left_count == 0 || right_count == 0)
    {
        node->is_leaf = 1;
        free(left_indices);
        free(right_indices);
        return node;
    }

    node->is_leaf = 0;
    node->feature_index = best.feature_index;
    node->threshold = best.threshold;
    node->left = buildTreeRecursive(ds, left_indices, left_count, depth + 1, hp);
    node->right = buildTreeRecursive(ds, right_indices, right_count, depth + 1, hp);

    free(left_indices);
    free(right_indices);
    return node;
}

static TreeNode *trainDecisionTree(const Dataset *train, const TreeHyperParams *hp)
{
    int *indices = (int *)malloc(train->size * sizeof(int));
    if (!indices)
    {
        fprintf(stderr, "FATAL: malloc failed in trainDecisionTree\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < train->size; i++)
        indices[i] = i;
    TreeNode *root = buildTreeRecursive(train, indices, train->size, 0, hp);
    free(indices);
    return root;
}

static void freeTree(TreeNode *node)
{
    if (!node)
        return;
    freeTree(node->left);
    freeTree(node->right);
    free(node);
}

static int predictDecisionTree(const TreeNode *node, const double *features)
{
    const TreeNode *cur = node;
    while (cur && !cur->is_leaf)
    {
        if (features[cur->feature_index] <= cur->threshold)
            cur = cur->left;
        else
            cur = cur->right;
    }
    return cur ? cur->predicted_label : LABEL_FIT;
}

static void batchPredictDecisionTree(const TreeNode *tree, const Dataset *ds, int *predictions)
{
    for (int i = 0; i < ds->size; i++)
    {
        predictions[i] = predictDecisionTree(tree, ds->data[i].features);
    }
}

static void printTreeRecursive(const TreeNode *node, int depth)
{
    for (int i = 0; i < depth; i++)
        printf("    ");

    if (node->is_leaf)
    {
        printf("[LEAF] -> %-5s  (samples=%d, gini=%.4f)\n",
               labelToStr(node->predicted_label), node->sample_count, node->gini);
        return;
    }

    printf("%s <= %.2f  (samples=%d, gini=%.4f)\n",
           (node->feature_index == 0) ? "Height" : "Weight",
           node->threshold,
           node->sample_count,
           node->gini);

    for (int i = 0; i < depth; i++)
        printf("    ");
    printf("|-- True  (<=):\n");
    printTreeRecursive(node->left, depth + 1);

    for (int i = 0; i < depth; i++)
        printf("    ");
    printf("|-- False (>):\n");
    printTreeRecursive(node->right, depth + 1);
}

static TreeHyperParams selectBestDecisionTreeParams(const Dataset *train, const Dataset *valid)
{
    TreeHyperParams best = {5, 4};
    double best_acc = -1.0;
    int *preds = (int *)malloc(valid->size * sizeof(int));
    if (!preds)
    {
        fprintf(stderr, "FATAL: malloc failed in selectBestDecisionTreeParams\n");
        exit(EXIT_FAILURE);
    }

    printf("\n  Validation search for best Decision Tree hyperparameters:\n");

    for (int depth = DT_MIN_DEPTH_TO_TRY; depth <= DT_MAX_DEPTH_TO_TRY; depth++)
    {
        for (size_t i = 0; i < sizeof(MIN_SAMPLES_OPTIONS) / sizeof(MIN_SAMPLES_OPTIONS[0]); i++)
        {
            TreeHyperParams hp;
            hp.max_depth = depth;
            hp.min_samples_split = MIN_SAMPLES_OPTIONS[i];

            TreeNode *tree = trainDecisionTree(train, &hp);
            batchPredictDecisionTree(tree, valid, preds);
            double acc = computeAccuracy(valid, preds);
            printf("    max_depth = %-2d  min_samples_split = %-2d  ->  validation accuracy = %6.2f%%\n",
                   hp.max_depth, hp.min_samples_split, acc);

            if (acc > best_acc + 1e-12)
            {
                best_acc = acc;
                best = hp;
            }
            freeTree(tree);
        }
    }

    printf("  Chosen Decision Tree settings: max_depth = %d, min_samples_split = %d\n",
           best.max_depth, best.min_samples_split);

    free(preds);
    return best;
}

static void interactiveDecisionTreeCLI(const TreeNode *tree)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  DECISION TREE INTERACTIVE PREDICTION  (q to quit)         \n");
    printf("══════════════════════════════════════════════════════════════\n");

    char buf[64];

    while (1)
    {
        double height_raw, weight_raw;

        printf("\n  Enter Height (cm) [or 'q' to quit]: ");
        if (fgets(buf, sizeof(buf), stdin) == NULL)
            break;
        if (buf[0] == 'q' || buf[0] == 'Q')
            break;
        if (sscanf(buf, "%lf", &height_raw) != 1)
        {
            printf("  [!] Invalid input. Please enter a numeric value.\n");
            continue;
        }

        printf("  Enter Weight (kg): ");
        if (fgets(buf, sizeof(buf), stdin) == NULL)
            break;
        if (buf[0] == 'q' || buf[0] == 'Q')
            break;
        if (sscanf(buf, "%lf", &weight_raw) != 1)
        {
            printf("  [!] Invalid input. Please enter a numeric value.\n");
            continue;
        }

        if (height_raw < 50 || height_raw > 250 || weight_raw < 10 || weight_raw > 300)
        {
            printf("  [!] Values look unrealistic. Please try again.\n");
            continue;
        }

        double features[NUM_FEATURES] = {height_raw, weight_raw};
        int pred = predictDecisionTree(tree, features);
        double bmi = weight_raw / ((height_raw / 100.0) * (height_raw / 100.0));

        printf("\n");
        printf("  ┌─────────────────────────────────────────────────┐\n");
        printf("  │           DECISION TREE PREDICTION             │\n");
        printf("  ├──────────────┬──────────────────────────────────┤\n");
        printf("  │ Height       │ %.1f cm                          │\n", height_raw);
        printf("  │ Weight       │ %.1f kg                          │\n", weight_raw);
        printf("  │ BMI          │ %.2f                             │\n", bmi);
        printf("  ├──────────────┼──────────────────────────────────┤\n");
        printf("  │ Prediction   │ %-5s                             │\n", labelToStr(pred));
        printf("  └──────────────┴──────────────────────────────────┘\n");
    }

    printf("\n  Goodbye!\n\n");
}

int main(int argc, char *argv[])
{
    printCommonHeader("FIT / OBESE CLASSIFIER  —  DECISION TREE  (C99)",
                      "Train / Validate / Test from CSV with readable reports");

    const char *train_csv = (argc >= 2) ? argv[1] : DEFAULT_TRAIN_CSV;
    const char *valid_csv = (argc >= 3) ? argv[2] : DEFAULT_VALID_CSV;
    const char *test_csv = (argc >= 4) ? argv[3] : DEFAULT_TEST_CSV;

    Dataset train = loadDatasetFromCSV(train_csv);
    Dataset valid = loadDatasetFromCSV(valid_csv);
    Dataset test = loadDatasetFromCSV(test_csv);

    printCSVInfo(train_csv, valid_csv, test_csv, &train, &valid, &test);

    TreeHyperParams hp = selectBestDecisionTreeParams(&train, &valid);
    TreeNode *tree = trainDecisionTree(&train, &hp);

    printf("\n  Learned tree structure:\n");
    printTreeRecursive(tree, 1);

    int *train_preds = (int *)malloc(train.size * sizeof(int));
    int *valid_preds = (int *)malloc(valid.size * sizeof(int));
    int *test_preds = (int *)malloc(test.size * sizeof(int));

    if (!train_preds || !valid_preds || !test_preds)
    {
        fprintf(stderr, "FATAL: malloc failed for prediction arrays\n");
        free(train_preds);
        free(valid_preds);
        free(test_preds);
        freeTree(tree);
        freeDataset(&train);
        freeDataset(&valid);
        freeDataset(&test);
        return EXIT_FAILURE;
    }

    batchPredictDecisionTree(tree, &train, train_preds);
    batchPredictDecisionTree(tree, &valid, valid_preds);
    batchPredictDecisionTree(tree, &test, test_preds);

    ClassificationReport train_report = buildClassificationReport(&train, train_preds);
    ClassificationReport valid_report = buildClassificationReport(&valid, valid_preds);
    ClassificationReport test_report = buildClassificationReport(&test, test_preds);

    printClassificationReport("Decision Tree", "Train report", &train_report);
    printClassificationReport("Decision Tree", "Validation report", &valid_report);
    printClassificationReport("Decision Tree", "Test report", &test_report);

    free(train_preds);
    free(valid_preds);
    free(test_preds);

    interactiveDecisionTreeCLI(tree);

    freeTree(tree);
    freeDataset(&train);
    freeDataset(&valid);
    freeDataset(&test);
    return EXIT_SUCCESS;
}
