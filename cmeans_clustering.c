#include "ml_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FCM_CLUSTERS       2
#define FCM_FUZZINESS      2.0
#define FCM_MAX_ITER       1000
#define FCM_TOLERANCE      1e-6
#define FCM_SEED           815

typedef struct {
    double features[NUM_FEATURES];
} UnlabelledPerson;

typedef struct {
    UnlabelledPerson *data;
    int size;
} UnlabelledDataset;

typedef struct {
    double min_val[NUM_FEATURES];
    double max_val[NUM_FEATURES];
} UnlabelledScaler;

typedef struct {
    double memberships[FCM_CLUSTERS];
} MembershipRow;

typedef struct {
    double centers[FCM_CLUSTERS][NUM_FEATURES];
    MembershipRow *U;
    int iterations;
    int obese_cluster;
    int fit_cluster;
} FCMModel;

static UnlabelledDataset loadUnlabelledDataset(const char *filename)
{
    UnlabelledDataset ds;
    ds.data = NULL;
    ds.size = 0;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "FATAL: could not open unlabeled CSV file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    int capacity = INITIAL_CAPACITY;
    ds.data = (UnlabelledPerson *)malloc(capacity * sizeof(UnlabelledPerson));
    if (!ds.data) {
        fprintf(stderr, "FATAL: malloc failed in loadUnlabelledDataset\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    char line[256];
    int line_no = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        line_no++;
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        char *height_tok = strtok(line, ",");
        char *weight_tok = strtok(NULL, ",\r\n");
        if (!height_tok || !weight_tok) {
            fprintf(stderr, "FATAL: invalid unlabeled CSV format in %s at line %d\n", filename, line_no);
            free(ds.data);
            fclose(fp);
            exit(EXIT_FAILURE);
        }

        if (line_no == 1) {
            char *endptr = NULL;
            strtod(height_tok, &endptr);
            if (endptr == height_tok || (*endptr != '\0' && *endptr != '\r' && *endptr != '\n')) {
                continue;
            }
        }

        if (ds.size == capacity) {
            capacity *= 2;
            UnlabelledPerson *tmp = (UnlabelledPerson *)realloc(ds.data, capacity * sizeof(UnlabelledPerson));
            if (!tmp) {
                fprintf(stderr, "FATAL: realloc failed in loadUnlabelledDataset\n");
                free(ds.data);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
            ds.data = tmp;
        }

        ds.data[ds.size].features[0] = atof(height_tok);
        ds.data[ds.size].features[1] = atof(weight_tok);
        ds.size++;
    }

    fclose(fp);

    if (ds.size == 0) {
        fprintf(stderr, "FATAL: unlabeled CSV file '%s' has no usable rows\n", filename);
        free(ds.data);
        exit(EXIT_FAILURE);
    }

    return ds;
}

static void freeUnlabelledDataset(UnlabelledDataset *ds)
{
    free(ds->data);
    ds->data = NULL;
    ds->size = 0;
}

static Dataset concatenateLabelledDatasets(const Dataset *train, const Dataset *valid, const Dataset *test)
{
    Dataset merged;
    merged.size = train->size + valid->size + test->size;
    merged.data = (Person *)malloc(merged.size * sizeof(Person));
    if (!merged.data) {
        fprintf(stderr, "FATAL: malloc failed in concatenateLabelledDatasets\n");
        exit(EXIT_FAILURE);
    }

    int pos = 0;
    for (int i = 0; i < train->size; i++) merged.data[pos++] = train->data[i];
    for (int i = 0; i < valid->size; i++) merged.data[pos++] = valid->data[i];
    for (int i = 0; i < test->size; i++)  merged.data[pos++] = test->data[i];
    return merged;
}

static UnlabelledScaler fitUnlabelledScaler(const UnlabelledDataset *ds)
{
    UnlabelledScaler sp;
    for (int f = 0; f < NUM_FEATURES; f++) {
        sp.min_val[f] = ds->data[0].features[f];
        sp.max_val[f] = ds->data[0].features[f];
    }

    for (int i = 1; i < ds->size; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            double v = ds->data[i].features[f];
            if (v < sp.min_val[f]) sp.min_val[f] = v;
            if (v > sp.max_val[f]) sp.max_val[f] = v;
        }
    }
    return sp;
}

static void transformUnlabelledDataset(UnlabelledDataset *ds, const UnlabelledScaler *sp)
{
    for (int i = 0; i < ds->size; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            double range = sp->max_val[f] - sp->min_val[f];
            if (range < 1e-9) range = 1e-9;
            ds->data[i].features[f] = (ds->data[i].features[f] - sp->min_val[f]) / range;
        }
    }
}

static void transformUnlabelledPoint(const double *raw_features, double *scaled, const UnlabelledScaler *sp)
{
    for (int f = 0; f < NUM_FEATURES; f++) {
        double range = sp->max_val[f] - sp->min_val[f];
        if (range < 1e-9) range = 1e-9;
        scaled[f] = (raw_features[f] - sp->min_val[f]) / range;
    }
}

static double pointDistance2D(const double *a, const double *b)
{
    double d0 = a[0] - b[0];
    double d1 = a[1] - b[1];
    return sqrt(d0 * d0 + d1 * d1);
}

static void initializeMemberships(MembershipRow *U, int n)
{
    srand(FCM_SEED);
    for (int i = 0; i < n; i++) {
        double r = safeDivide((double)rand(), (double)RAND_MAX);
        if (r < 1e-6) r = 1e-6;
        if (r > 1.0 - 1e-6) r = 1.0 - 1e-6;
        U[i].memberships[0] = r;
        U[i].memberships[1] = 1.0 - r;
    }
}

static void updateCenters(const UnlabelledDataset *ds, const MembershipRow *U, double centers[FCM_CLUSTERS][NUM_FEATURES])
{
    for (int c = 0; c < FCM_CLUSTERS; c++) {
        double numerator[NUM_FEATURES] = {0.0, 0.0};
        double denominator = 0.0;

        for (int i = 0; i < ds->size; i++) {
            double u_m = pow(U[i].memberships[c], FCM_FUZZINESS);
            for (int f = 0; f < NUM_FEATURES; f++) {
                numerator[f] += u_m * ds->data[i].features[f];
            }
            denominator += u_m;
        }

        for (int f = 0; f < NUM_FEATURES; f++) {
            centers[c][f] = safeDivide(numerator[f], denominator);
        }
    }
}

static double updateMemberships(const UnlabelledDataset *ds,
                                MembershipRow *U,
                                const double centers[FCM_CLUSTERS][NUM_FEATURES])
{
    double max_change = 0.0;
    double exponent = 2.0 / (FCM_FUZZINESS - 1.0);

    for (int i = 0; i < ds->size; i++) {
        double distances[FCM_CLUSTERS];
        int zero_cluster = -1;

        for (int c = 0; c < FCM_CLUSTERS; c++) {
            distances[c] = pointDistance2D(ds->data[i].features, centers[c]);
            if (distances[c] < 1e-12) {
                zero_cluster = c;
            }
        }

        double new_u[FCM_CLUSTERS];
        if (zero_cluster >= 0) {
            for (int c = 0; c < FCM_CLUSTERS; c++) new_u[c] = 0.0;
            new_u[zero_cluster] = 1.0;
        } else {
            for (int c = 0; c < FCM_CLUSTERS; c++) {
                double denom = 0.0;
                for (int k = 0; k < FCM_CLUSTERS; k++) {
                    denom += pow(distances[c] / distances[k], exponent);
                }
                new_u[c] = safeDivide(1.0, denom);
            }
        }

        for (int c = 0; c < FCM_CLUSTERS; c++) {
            double diff = fabs(new_u[c] - U[i].memberships[c]);
            if (diff > max_change) max_change = diff;
            U[i].memberships[c] = new_u[c];
        }
    }

    return max_change;
}

static int predictCluster(const FCMModel *model, const double *scaled_features)
{
    double memberships[FCM_CLUSTERS];
    double distances[FCM_CLUSTERS];
    int zero_cluster = -1;

    for (int c = 0; c < FCM_CLUSTERS; c++) {
        distances[c] = pointDistance2D(scaled_features, model->centers[c]);
        if (distances[c] < 1e-12) zero_cluster = c;
    }

    if (zero_cluster >= 0) return zero_cluster;

    double exponent = 2.0 / (FCM_FUZZINESS - 1.0);
    for (int c = 0; c < FCM_CLUSTERS; c++) {
        double denom = 0.0;
        for (int k = 0; k < FCM_CLUSTERS; k++) {
            denom += pow(distances[c] / distances[k], exponent);
        }
        memberships[c] = safeDivide(1.0, denom);
    }

    return (memberships[0] >= memberships[1]) ? 0 : 1;
}

static int clusterToLabel(const FCMModel *model, int cluster)
{
    return (cluster == model->obese_cluster) ? LABEL_OBESE : LABEL_FIT;
}

static FCMModel trainFCM(const UnlabelledDataset *ds)
{
    FCMModel model;
    model.U = (MembershipRow *)malloc(ds->size * sizeof(MembershipRow));
    if (!model.U) {
        fprintf(stderr, "FATAL: malloc failed in trainFCM\n");
        exit(EXIT_FAILURE);
    }

    initializeMemberships(model.U, ds->size);
    model.iterations = 0;

    for (int iter = 1; iter <= FCM_MAX_ITER; iter++) {
        updateCenters(ds, model.U, model.centers);
        double max_change = updateMemberships(ds, model.U, model.centers);
        model.iterations = iter;
        if (max_change < FCM_TOLERANCE) break;
    }

    int heavier_cluster = (model.centers[0][1] >= model.centers[1][1]) ? 0 : 1;
    model.obese_cluster = heavier_cluster;
    model.fit_cluster = 1 - heavier_cluster;
    return model;
}

static void freeFCMModel(FCMModel *model)
{
    free(model->U);
    model->U = NULL;
}

static void interactiveFCMCLI(const FCMModel *model, const UnlabelledScaler *sp)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  FUZZY C-MEANS INTERACTIVE PREDICTION  (q to quit)         \n");
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

        double raw[NUM_FEATURES] = {height_raw, weight_raw};
        double scaled[NUM_FEATURES];
        transformUnlabelledPoint(raw, scaled, sp);

        int cluster = predictCluster(model, scaled);
        int pred = clusterToLabel(model, cluster);
        double bmi = weight_raw / ((height_raw / 100.0) * (height_raw / 100.0));

        printf("\n");
        printf("  ┌─────────────────────────────────────────────────┐\n");
        printf("  │          FUZZY C-MEANS PREDICTION              │\n");
        printf("  ├──────────────┬──────────────────────────────────┤\n");
        printf("  │ Height       │ %.1f cm                          │\n", height_raw);
        printf("  │ Weight       │ %.1f kg                          │\n", weight_raw);
        printf("  │ BMI          │ %.2f                             │\n", bmi);
        printf("  ├──────────────┼──────────────────────────────────┤\n");
        printf("  │ Cluster      │ %-5d                             │\n", cluster);
        printf("  │ Prediction   │ %-5s                             │\n", labelToStr(pred));
        printf("  └──────────────┴──────────────────────────────────┘\n");
    }

    printf("\n  Goodbye!\n\n");
}

int main(int argc, char *argv[])
{
    printCommonHeader("FIT / OBESE CLUSTERING  —  FUZZY C-MEANS  (C99)",
                      "Unlabelled data with reference agreement report from the labelled source");

    const char *unlabelled_csv = (argc >= 2) ? argv[1] : DEFAULT_UNLABELLED_CSV;
    const char *train_csv      = (argc >= 3) ? argv[2] : DEFAULT_TRAIN_CSV;
    const char *valid_csv      = (argc >= 4) ? argv[3] : DEFAULT_VALID_CSV;
    const char *test_csv       = (argc >= 5) ? argv[4] : DEFAULT_TEST_CSV;

    UnlabelledDataset unlabeled = loadUnlabelledDataset(unlabelled_csv);
    Dataset train = loadDatasetFromCSV(train_csv);
    Dataset valid = loadDatasetFromCSV(valid_csv);
    Dataset test  = loadDatasetFromCSV(test_csv);
    Dataset reference = concatenateLabelledDatasets(&train, &valid, &test);

    if (reference.size != unlabeled.size) {
        fprintf(stderr, "FATAL: unlabeled/reference size mismatch (%d vs %d)\n", unlabeled.size, reference.size);
        freeUnlabelledDataset(&unlabeled);
        freeDataset(&train);
        freeDataset(&valid);
        freeDataset(&test);
        freeDataset(&reference);
        return EXIT_FAILURE;
    }

    printf("\n  Loaded CSV files:\n");
    printf("    Unlabelled : %s\n", unlabelled_csv);
    printf("    Reference  : %s + %s + %s\n", train_csv, valid_csv, test_csv);
    printf("\n  Total samples: %d\n", unlabeled.size);

    UnlabelledScaler sp = fitUnlabelledScaler(&unlabeled);
    printf("\n  Scaler fitted on the full unlabeled dataset:\n");
    printf("    Height range : [%.1f, %.1f] cm\n", sp.min_val[0], sp.max_val[0]);
    printf("    Weight range : [%.1f, %.1f] kg\n", sp.min_val[1], sp.max_val[1]);
    printf("  Min-max normalization applied before clustering.\n");

    transformUnlabelledDataset(&unlabeled, &sp);

    FCMModel model = trainFCM(&unlabeled);

    printf("\n  Training complete.\n");
    printf("    Iterations used : %d\n", model.iterations);
    printf("    Cluster %d -> FIT\n", model.fit_cluster);
    printf("    Cluster %d -> OBESE\n", model.obese_cluster);
    printf("\n  Cluster centres (normalized):\n");
    for (int c = 0; c < FCM_CLUSTERS; c++) {
        printf("    Cluster %d  |  height = %.4f  |  weight = %.4f\n",
               c, model.centers[c][0], model.centers[c][1]);
    }

    int *predictions = (int *)malloc(reference.size * sizeof(int));
    if (!predictions) {
        fprintf(stderr, "FATAL: malloc failed for FCM predictions\n");
        freeFCMModel(&model);
        freeUnlabelledDataset(&unlabeled);
        freeDataset(&train);
        freeDataset(&valid);
        freeDataset(&test);
        freeDataset(&reference);
        return EXIT_FAILURE;
    }

    int fit_count = 0;
    int obese_count = 0;
    for (int i = 0; i < unlabeled.size; i++) {
        int cluster = predictCluster(&model, unlabeled.data[i].features);
        predictions[i] = clusterToLabel(&model, cluster);
        if (predictions[i] == LABEL_FIT) fit_count++;
        else                             obese_count++;
    }

    printf("\n  Clustered label distribution:\n");
    printf("    FIT   : %d (%.2f%%)\n", fit_count, safeDivide(100.0 * fit_count, (double)reference.size));
    printf("    OBESE : %d (%.2f%%)\n", obese_count, safeDivide(100.0 * obese_count, (double)reference.size));

    ClassificationReport agreement_report = buildClassificationReport(&reference, predictions);
    printMetricGuide();
    printClassificationReport("Fuzzy C-Means", "Reference agreement report", &agreement_report);

    free(predictions);

    interactiveFCMCLI(&model, &sp);

    freeFCMModel(&model);
    freeUnlabelledDataset(&unlabeled);
    freeDataset(&train);
    freeDataset(&valid);
    freeDataset(&test);
    freeDataset(&reference);
    return EXIT_SUCCESS;
}
