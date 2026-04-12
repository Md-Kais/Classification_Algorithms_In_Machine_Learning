#include "ml_common.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

double safeDivide(double num, double den)
{
    return (den > 0.0) ? (num / den) : 0.0;
}

static void trimWhitespace(char *s)
{
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }

    size_t start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;

    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }
}

static int parseLabelToken(const char *token)
{
    if (strcmp(token, "FIT") == 0 || strcmp(token, "fit") == 0 || strcmp(token, "0") == 0) {
        return LABEL_FIT;
    }
    if (strcmp(token, "OBESE") == 0 || strcmp(token, "obese") == 0 || strcmp(token, "1") == 0) {
        return LABEL_OBESE;
    }

    fprintf(stderr, "FATAL: unknown label '%s' in CSV\n", token);
    exit(EXIT_FAILURE);
}

Dataset loadDatasetFromCSV(const char *filename)
{
    Dataset ds;
    ds.size = 0;
    ds.data = NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "FATAL: could not open CSV file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    int capacity = INITIAL_CAPACITY;
    ds.data = (Person *)malloc(capacity * sizeof(Person));
    if (!ds.data) {
        fprintf(stderr, "FATAL: malloc failed in loadDatasetFromCSV\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    char line[256];
    int line_no = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        line_no++;
        trimWhitespace(line);

        if (line[0] == '\0') continue;

        char line_copy[256];
        strncpy(line_copy, line, sizeof(line_copy) - 1);
        line_copy[sizeof(line_copy) - 1] = '\0';

        char *height_tok = strtok(line_copy, ",");
        char *weight_tok = strtok(NULL, ",");
        char *label_tok  = strtok(NULL, ",");

        if (!height_tok || !weight_tok || !label_tok) {
            fprintf(stderr, "FATAL: invalid CSV format in %s at line %d\n", filename, line_no);
            fclose(fp);
            free(ds.data);
            exit(EXIT_FAILURE);
        }

        trimWhitespace(height_tok);
        trimWhitespace(weight_tok);
        trimWhitespace(label_tok);

        if (line_no == 1) {
            char *endptr_h = NULL;
            strtod(height_tok, &endptr_h);
            if (endptr_h == height_tok || *endptr_h != '\0') {
                continue;
            }
        }

        if (ds.size == capacity) {
            capacity *= 2;
            Person *tmp = (Person *)realloc(ds.data, capacity * sizeof(Person));
            if (!tmp) {
                fprintf(stderr, "FATAL: realloc failed in loadDatasetFromCSV\n");
                fclose(fp);
                free(ds.data);
                exit(EXIT_FAILURE);
            }
            ds.data = tmp;
        }

        ds.data[ds.size].features[0] = atof(height_tok);
        ds.data[ds.size].features[1] = atof(weight_tok);
        ds.data[ds.size].label = parseLabelToken(label_tok);
        ds.size++;
    }

    fclose(fp);

    if (ds.size == 0) {
        fprintf(stderr, "FATAL: CSV file '%s' has no usable rows\n", filename);
        free(ds.data);
        exit(EXIT_FAILURE);
    }

    Person *tmp = (Person *)realloc(ds.data, ds.size * sizeof(Person));
    if (tmp) ds.data = tmp;

    return ds;
}

void freeDataset(Dataset *ds)
{
    free(ds->data);
    ds->data = NULL;
    ds->size = 0;
}

int countLabel(const Dataset *ds, int label)
{
    int count = 0;
    for (int i = 0; i < ds->size; i++) {
        if (ds->data[i].label == label) count++;
    }
    return count;
}

void printDatasetSummary(const char *name, const Dataset *ds)
{
    int fit_count = countLabel(ds, LABEL_FIT);
    int obese_count = countLabel(ds, LABEL_OBESE);

    printf("  %-10s : %3d samples  |  FIT: %3d  |  OBESE: %3d\n",
           name, ds->size, fit_count, obese_count);
}

ScalerParams fitScaler(const Dataset *train)
{
    ScalerParams sp;
    for (int f = 0; f < NUM_FEATURES; f++) {
        sp.min_val[f] = train->data[0].features[f];
        sp.max_val[f] = train->data[0].features[f];
    }

    for (int i = 1; i < train->size; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            double v = train->data[i].features[f];
            if (v < sp.min_val[f]) sp.min_val[f] = v;
            if (v > sp.max_val[f]) sp.max_val[f] = v;
        }
    }
    return sp;
}

void transformDataset(Dataset *ds, const ScalerParams *sp)
{
    for (int i = 0; i < ds->size; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            double range = sp->max_val[f] - sp->min_val[f];
            if (range < 1e-9) range = 1e-9;
            ds->data[i].features[f] =
                (ds->data[i].features[f] - sp->min_val[f]) / range;
        }
    }
}

void transformPoint(const double *raw_features, double *scaled, const ScalerParams *sp)
{
    for (int f = 0; f < NUM_FEATURES; f++) {
        double range = sp->max_val[f] - sp->min_val[f];
        if (range < 1e-9) range = 1e-9;
        scaled[f] = (raw_features[f] - sp->min_val[f]) / range;
    }
}

double computeAccuracy(const Dataset *dataset, const int *predictions)
{
    int correct = 0;
    for (int i = 0; i < dataset->size; i++) {
        if (predictions[i] == dataset->data[i].label) correct++;
    }
    return safeDivide(100.0 * correct, (double)dataset->size);
}

double computeAccuracyFromArrays(const int *truth, const int *predictions, int n)
{
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (truth[i] == predictions[i]) correct++;
    }
    return safeDivide(100.0 * correct, (double)n);
}

ClassificationReport buildClassificationReportFromArrays(const int *truth, const int *predictions, int n)
{
    ClassificationReport r;

    for (int c = 0; c < NUM_CLASSES; c++) {
        r.precision[c] = 0.0;
        r.recall[c] = 0.0;
        r.f1[c] = 0.0;
        r.support[c] = 0;
    }

    r.accuracy = 0.0;
    r.macro_precision = 0.0;
    r.macro_recall = 0.0;
    r.macro_f1 = 0.0;
    r.weighted_precision = 0.0;
    r.weighted_recall = 0.0;
    r.weighted_f1 = 0.0;
    r.total_support = n;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int support = 0;

        for (int i = 0; i < n; i++) {
            if (truth[i] == c) support++;
            if (truth[i] == c && predictions[i] == c) tp++;
            if (truth[i] != c && predictions[i] == c) fp++;
            if (truth[i] == c && predictions[i] != c) fn++;
        }

        r.support[c] = support;
        r.precision[c] = safeDivide((double)tp, (double)(tp + fp));
        r.recall[c]    = safeDivide((double)tp, (double)(tp + fn));
        r.f1[c]        = safeDivide(2.0 * r.precision[c] * r.recall[c],
                                    r.precision[c] + r.recall[c]);
    }

    r.accuracy = safeDivide(computeAccuracyFromArrays(truth, predictions, n), 100.0);

    for (int c = 0; c < NUM_CLASSES; c++) {
        r.macro_precision += r.precision[c];
        r.macro_recall    += r.recall[c];
        r.macro_f1        += r.f1[c];

        double weight = safeDivide((double)r.support[c], (double)r.total_support);
        r.weighted_precision += weight * r.precision[c];
        r.weighted_recall    += weight * r.recall[c];
        r.weighted_f1        += weight * r.f1[c];
    }

    r.macro_precision /= NUM_CLASSES;
    r.macro_recall    /= NUM_CLASSES;
    r.macro_f1        /= NUM_CLASSES;

    return r;
}

ClassificationReport buildClassificationReport(const Dataset *ds, const int *predictions)
{
    int *truth = (int *)malloc(ds->size * sizeof(int));
    if (!truth) {
        fprintf(stderr, "FATAL: malloc failed in buildClassificationReport\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < ds->size; i++) truth[i] = ds->data[i].label;

    ClassificationReport r = buildClassificationReportFromArrays(truth, predictions, ds->size);
    free(truth);
    return r;
}

const char *labelToStr(int label)
{
    return (label == LABEL_FIT) ? "FIT" : "OBESE";
}

const char *labelToStrCustom(int label, const char *class_zero, const char *class_one)
{
    return (label == 0) ? class_zero : class_one;
}

void printMetricGuide(void)
{
    printf("\n");
    printf("  Metric guide:\n");
    printf("    Precision : When the model predicts a class, how often that prediction is right.\n");
    printf("    Recall    : Of all real samples in a class, how many the model found.\n");
    printf("    F1-Score  : Balance between precision and recall.\n");
    printf("    Support   : Number of real samples belonging to that class.\n");
    printf("    Macro Avg : Simple average across classes.\n");
    printf("    Weighted  : Average weighted by class support.\n");
}

void printClassificationReportWithNames(const char *model_name,
                                        const char *split_name,
                                        const ClassificationReport *r,
                                        const char *class_zero,
                                        const char *class_one)
{
    printf("\n");
    printf("┌────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-26s | %-43s │\n", model_name, split_name);
    printf("├────────────────┬────────────┬────────────┬────────────┬────────────┤\n");
    printf("│ Class          │ Precision  │ Recall     │ F1-Score   │ Support    │\n");
    printf("├────────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    printf("│ %-14s │ %10.3f │ %10.3f │ %10.3f │ %10d │\n",
           class_zero,
           r->precision[0],
           r->recall[0],
           r->f1[0],
           r->support[0]);
    printf("│ %-14s │ %10.3f │ %10.3f │ %10.3f │ %10d │\n",
           class_one,
           r->precision[1],
           r->recall[1],
           r->f1[1],
           r->support[1]);
    printf("├────────────────┼────────────┼────────────┼────────────┼────────────┤\n");
    printf("│ %-14s │ %10.3f │ %10.3f │ %10.3f │ %10d │\n",
           "Macro Avg",
           r->macro_precision,
           r->macro_recall,
           r->macro_f1,
           r->total_support);
    printf("│ %-14s │ %10.3f │ %10.3f │ %10.3f │ %10d │\n",
           "Weighted Avg",
           r->weighted_precision,
           r->weighted_recall,
           r->weighted_f1,
           r->total_support);
    printf("├────────────────┴────────────┴────────────┴────────────┬────────────┤\n");
    printf("│ Accuracy                                              │ %10.3f │\n",
           r->accuracy);
    printf("└───────────────────────────────────────────────────────┴────────────┘\n");
}

void printClassificationReport(const char *model_name,
                               const char *split_name,
                               const ClassificationReport *r)
{
    printClassificationReportWithNames(model_name, split_name, r, "FIT", "OBESE");
}

void printCommonHeader(const char *title_line, const char *subtitle_line)
{
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║ %-74s ║\n", title_line);
    printf("║ %-74s ║\n", subtitle_line);
    printf("╚════════════════════════════════════════════════════════════════════════════╝\n");
}

void printCSVInfo(const char *train_csv, const char *valid_csv, const char *test_csv,
                  const Dataset *train, const Dataset *valid, const Dataset *test)
{
    printf("\n  Loaded CSV files:\n");
    printf("    Train    : %s\n", train_csv);
    printf("    Validate : %s\n", valid_csv);
    printf("    Test     : %s\n", test_csv);

    printf("\n  Dataset summary:\n");
    printDatasetSummary("Train", train);
    printDatasetSummary("Validate", valid);
    printDatasetSummary("Test", test);
}

void printScalerInfo(const ScalerParams *sp)
{
    printf("\n  Scaler fitted on TRAIN only:\n");
    printf("    Height range : [%.1f, %.1f] cm\n", sp->min_val[0], sp->max_val[0]);
    printf("    Weight range : [%.1f, %.1f] kg\n", sp->min_val[1], sp->max_val[1]);
    printf("  Min-max normalization applied to train, validation, and test.\n");
}
