#include "ml_common.h"

#include <math.h>
#include <stdlib.h>

typedef struct {
    double distance;
    int    label;
} DistLabel;

static void bubbleSort(DistLabel *arr, int n)
{
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j].distance > arr[j + 1].distance) {
                DistLabel tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

static int classifyKNNCore(const Dataset *train, const double *features, int k, int skip_index)
{
    DistLabel *dl = (DistLabel *)malloc(train->size * sizeof(DistLabel));
    if (!dl) {
        fprintf(stderr, "FATAL: malloc failed in classifyKNNCore\n");
        exit(EXIT_FAILURE);
    }

    int used = 0;
    for (int i = 0; i < train->size; i++) {
        if (i == skip_index) continue;

        double sum_sq = 0.0;
        for (int f = 0; f < NUM_FEATURES; f++) {
            double diff = features[f] - train->data[i].features[f];
            sum_sq += diff * diff;
        }
        dl[used].distance = sqrt(sum_sq);
        dl[used].label = train->data[i].label;
        used++;
    }

    bubbleSort(dl, used);

    int votes_fit = 0;
    int votes_obese = 0;
    for (int i = 0; i < k && i < used; i++) {
        if (dl[i].label == LABEL_FIT) votes_fit++;
        else                          votes_obese++;
    }

    free(dl);
    return (votes_fit >= votes_obese) ? LABEL_FIT : LABEL_OBESE;
}

static int classifyKNN(const Dataset *train, const double *features, int k)
{
    return classifyKNNCore(train, features, k, -1);
}

static void batchPredictKNN(const Dataset *train, const Dataset *ds, int *predictions, int k)
{
    for (int i = 0; i < ds->size; i++) {
        predictions[i] = classifyKNN(train, ds->data[i].features, k);
    }
}

static void batchPredictKNNLeaveOneOut(const Dataset *train, int *predictions, int k)
{
    for (int i = 0; i < train->size; i++) {
        predictions[i] = classifyKNNCore(train, train->data[i].features, k, i);
    }
}

static int selectBestK(const Dataset *train, const Dataset *valid)
{
    int best_k = DEFAULT_K;
    double best_acc = -1.0;

    int *preds = (int *)malloc(valid->size * sizeof(int));
    if (!preds) {
        fprintf(stderr, "FATAL: malloc failed in selectBestK\n");
        exit(EXIT_FAILURE);
    }

    printf("\n  Validation search for best k in KNN:\n");
    printf("    k = 1, 3, 5, ... up to %d\n", MAX_K_TO_TRY);

    int max_k = MAX_K_TO_TRY;
    if (max_k > train->size) max_k = train->size;
    if (max_k % 2 == 0) max_k--;

    for (int k = 1; k <= max_k; k += 2) {
        batchPredictKNN(train, valid, preds, k);
        double acc = computeAccuracy(valid, preds);
        printf("    k = %-2d  ->  validation accuracy = %6.2f%%\n", k, acc);

        if (acc > best_acc + 1e-12) {
            best_acc = acc;
            best_k = k;
        }
    }

    free(preds);

    printf("  Chosen k = %d (highest validation accuracy)\n", best_k);
    return best_k;
}

static void interactiveKNNCLI(const Dataset *train, const ScalerParams *sp, int knn_k)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  KNN INTERACTIVE PREDICTION  (enter 'q' at any prompt)     \n");
    printf("══════════════════════════════════════════════════════════════\n");

    char buf[64];

    while (1) {
        double height_raw, weight_raw;

        printf("\n  Enter Height (cm) [or 'q' to quit]: ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%lf", &height_raw) != 1) {
            printf("  [!] Invalid input. Please enter a numeric value.\n");
            continue;
        }

        printf("  Enter Weight (kg): ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%lf", &weight_raw) != 1) {
            printf("  [!] Invalid input. Please enter a numeric value.\n");
            continue;
        }

        if (height_raw < 50 || height_raw > 250 || weight_raw < 10 || weight_raw > 300) {
            printf("  [!] Values look unrealistic. Please try again.\n");
            continue;
        }

        double raw_q[NUM_FEATURES] = {height_raw, weight_raw};
        double norm_q[NUM_FEATURES];
        transformPoint(raw_q, norm_q, sp);

        int pred = classifyKNN(train, norm_q, knn_k);
        double bmi = weight_raw / ((height_raw / 100.0) * (height_raw / 100.0));

        printf("\n");
        printf("  ┌─────────────────────────────────────────────────┐\n");
        printf("  │                KNN PREDICTION                   │\n");
        printf("  ├──────────────┬──────────────────────────────────┤\n");
        printf("  │ Height       │ %.1f cm                          │\n", height_raw);
        printf("  │ Weight       │ %.1f kg                          │\n", weight_raw);
        printf("  │ BMI          │ %.2f                             │\n", bmi);
        printf("  ├──────────────┼──────────────────────────────────┤\n");
        printf("  │ k            │ %-5d                             │\n", knn_k);
        printf("  │ Prediction   │ %-5s                             │\n", labelToStr(pred));
        printf("  └──────────────┴──────────────────────────────────┘\n");
    }

    printf("\n  Goodbye!\n\n");
}

int main(int argc, char *argv[])
{
    printCommonHeader("FIT / OBESE CLASSIFIER  —  KNN ONLY  (C99)",
                      "Train / Validate / Test from CSV with readable reports");

    const char *train_csv = (argc >= 2) ? argv[1] : DEFAULT_TRAIN_CSV;
    const char *valid_csv = (argc >= 3) ? argv[2] : DEFAULT_VALID_CSV;
    const char *test_csv  = (argc >= 4) ? argv[3] : DEFAULT_TEST_CSV;

    Dataset train = loadDatasetFromCSV(train_csv);
    Dataset valid = loadDatasetFromCSV(valid_csv);
    Dataset test  = loadDatasetFromCSV(test_csv);

    printCSVInfo(train_csv, valid_csv, test_csv, &train, &valid, &test);
    printf("\n  Note: KNN train accuracy uses leave-one-out evaluation, so each train sample is predicted without using itself as a neighbor.\n");

    ScalerParams sp = fitScaler(&train);
    printScalerInfo(&sp);

    transformDataset(&train, &sp);
    transformDataset(&valid, &sp);
    transformDataset(&test, &sp);

    int best_k = selectBestK(&train, &valid);

    int *train_preds = (int *)malloc(train.size * sizeof(int));
    int *valid_preds = (int *)malloc(valid.size * sizeof(int));
    int *test_preds  = (int *)malloc(test.size  * sizeof(int));

    if (!train_preds || !valid_preds || !test_preds) {
        fprintf(stderr, "FATAL: malloc failed for prediction arrays\n");
        free(train_preds);
        free(valid_preds);
        free(test_preds);
        freeDataset(&train);
        freeDataset(&valid);
        freeDataset(&test);
        return EXIT_FAILURE;
    }

    batchPredictKNNLeaveOneOut(&train, train_preds, best_k);
    batchPredictKNN(&train, &valid, valid_preds, best_k);
    batchPredictKNN(&train, &test,  test_preds,  best_k);

    ClassificationReport train_report = buildClassificationReport(&train, train_preds);
    ClassificationReport valid_report = buildClassificationReport(&valid, valid_preds);
    ClassificationReport test_report  = buildClassificationReport(&test,  test_preds);

    printMetricGuide();
    printClassificationReport("KNN", "Train report", &train_report);
    printClassificationReport("KNN", "Validation report", &valid_report);
    printClassificationReport("KNN", "Test report", &test_report);

    free(train_preds);
    free(valid_preds);
    free(test_preds);

    interactiveKNNCLI(&train, &sp, best_k);

    freeDataset(&train);
    freeDataset(&valid);
    freeDataset(&test);
    return EXIT_SUCCESS;
}
