#include "ml_common.h"

#include <stdlib.h>

typedef struct
{
    double w[NUM_FEATURES];
    double bias;
} SVMModel;

static SVMModel trainSVM(const Dataset *train, double lr, double lambda, int epochs)
{
    SVMModel model;
    for (int f = 0; f < NUM_FEATURES; f++)
        model.w[f] = 0.0;
    model.bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < train->size; i++)
        {
            double y_svm = (train->data[i].label == LABEL_FIT) ? 1.0 : -1.0;
            const double *x = train->data[i].features;

            double score = model.bias;
            for (int f = 0; f < NUM_FEATURES; f++)
            {
                score += model.w[f] * x[f];
            }

            double margin = y_svm * score;

            if (margin < 1.0)
            {
                for (int f = 0; f < NUM_FEATURES; f++)
                {
                    model.w[f] += lr * (y_svm * x[f] - 2.0 * lambda * model.w[f]);
                }
                model.bias += lr * y_svm;
            }
            else
            {
                for (int f = 0; f < NUM_FEATURES; f++)
                {
                    model.w[f] -= lr * 2.0 * lambda * model.w[f];
                }
            }
        }
    }

    return model;
}

static int predictSVM(const SVMModel *model, const double *features)
{
    double score = model->bias;
    for (int f = 0; f < NUM_FEATURES; f++)
    {
        score += model->w[f] * features[f];
    }
    return (score > 0.0) ? LABEL_FIT : LABEL_OBESE;
}

static void batchPredictSVM(const SVMModel *model, const Dataset *ds, int *predictions)
{
    for (int i = 0; i < ds->size; i++)
    {
        predictions[i] = predictSVM(model, ds->data[i].features);
    }
}

static void printSVMWeights(const SVMModel *model)
{
    printf("\n  SVM hyperplane:\n");
    printf("    %.6f * height_norm + %.6f * weight_norm + (%.6f) = 0\n",
           model->w[0], model->w[1], model->bias);
}

static void interactiveSVMCLI(const SVMModel *model, const ScalerParams *sp)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  SVM INTERACTIVE PREDICTION  (enter 'q' at any prompt)     \n");
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

        double raw_q[NUM_FEATURES] = {height_raw, weight_raw};
        double norm_q[NUM_FEATURES];
        transformPoint(raw_q, norm_q, sp);

        int pred = predictSVM(model, norm_q);
        double bmi = weight_raw / ((height_raw / 100.0) * (height_raw / 100.0));

        printf("\n");
        printf("  ┌─────────────────────────────────────────────────┐\n");
        printf("  │                SVM PREDICTION                   │\n");
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
    printCommonHeader("FIT / OBESE CLASSIFIER  —  SVM ONLY  (C99)",
                      "Train / Validate / Test from CSV with readable reports");

    const char *train_csv = (argc >= 2) ? argv[1] : DEFAULT_TRAIN_CSV;
    const char *valid_csv = (argc >= 3) ? argv[2] : DEFAULT_VALID_CSV;
    const char *test_csv = (argc >= 4) ? argv[3] : DEFAULT_TEST_CSV;

    Dataset train = loadDatasetFromCSV(train_csv);
    Dataset valid = loadDatasetFromCSV(valid_csv);
    Dataset test = loadDatasetFromCSV(test_csv);

    printCSVInfo(train_csv, valid_csv, test_csv, &train, &valid, &test);

    ScalerParams sp = fitScaler(&train);
    printScalerInfo(&sp);

    transformDataset(&train, &sp);
    transformDataset(&valid, &sp);
    transformDataset(&test, &sp);

    printf("\n  Training SVM (eta=%.4f, lambda=%.4f, epochs=%d) ...\n",
           LEARNING_RATE, LAMBDA_PARAM, EPOCHS);
    SVMModel svm = trainSVM(&train, LEARNING_RATE, LAMBDA_PARAM, EPOCHS);
    printf("  SVM training complete.\n");
    printSVMWeights(&svm);

    int *train_preds = (int *)malloc(train.size * sizeof(int));
    int *valid_preds = (int *)malloc(valid.size * sizeof(int));
    int *test_preds = (int *)malloc(test.size * sizeof(int));

    if (!train_preds || !valid_preds || !test_preds)
    {
        fprintf(stderr, "FATAL: malloc failed for prediction arrays\n");
        free(train_preds);
        free(valid_preds);
        free(test_preds);
        freeDataset(&train);
        freeDataset(&valid);
        freeDataset(&test);
        return EXIT_FAILURE;
    }

    batchPredictSVM(&svm, &train, train_preds);
    batchPredictSVM(&svm, &valid, valid_preds);
    batchPredictSVM(&svm, &test, test_preds);

    ClassificationReport train_report = buildClassificationReport(&train, train_preds);
    ClassificationReport valid_report = buildClassificationReport(&valid, valid_preds);
    ClassificationReport test_report = buildClassificationReport(&test, test_preds);
    printClassificationReport("SVM", "Train report", &train_report);
    printClassificationReport("SVM", "Validation report", &valid_report);
    printClassificationReport("SVM", "Test report", &test_report);

    free(train_preds);
    free(valid_preds);
    free(test_preds);

    interactiveSVMCLI(&svm, &sp);

    freeDataset(&train);
    freeDataset(&valid);
    freeDataset(&test);
    return EXIT_SUCCESS;
}
