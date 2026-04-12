#ifndef ML_COMMON_H
#define ML_COMMON_H

#include <stdio.h>

#define NUM_FEATURES 2
#define DEFAULT_K 3
#define MAX_K_TO_TRY 11
#define INITIAL_CAPACITY 64

#define DEFAULT_TRAIN_CSV "./Data/train.csv"
#define DEFAULT_VALID_CSV "./Data/validate.csv"
#define DEFAULT_TEST_CSV "./Data/test.csv"
#define DEFAULT_UNLABELLED_CSV "./Data/dataset_unlabelled.csv"
#define DEFAULT_FRUIT_CSV "./Data/fruit_dataset.csv"

#define LEARNING_RATE 0.005
#define LAMBDA_PARAM 0.001
#define EPOCHS 20000

#define LABEL_FIT 0
#define LABEL_OBESE 1
#define NUM_CLASSES 2

typedef struct
{
    double features[NUM_FEATURES];
    int label;
} Person;

typedef struct
{
    Person *data;
    int size;
} Dataset;

typedef struct
{
    double min_val[NUM_FEATURES];
    double max_val[NUM_FEATURES];
} ScalerParams;

typedef struct
{
    double precision[NUM_CLASSES];
    double recall[NUM_CLASSES];
    double f1[NUM_CLASSES];
    int support[NUM_CLASSES];

    double accuracy;

    double macro_precision;
    double macro_recall;
    double macro_f1;

    double weighted_precision;
    double weighted_recall;
    double weighted_f1;

    int total_support;
} ClassificationReport;

double safeDivide(double num, double den);
Dataset loadDatasetFromCSV(const char *filename);
void freeDataset(Dataset *ds);
int countLabel(const Dataset *ds, int label);
void printDatasetSummary(const char *name, const Dataset *ds);
ScalerParams fitScaler(const Dataset *train);
void transformDataset(Dataset *ds, const ScalerParams *sp);
void transformPoint(const double *raw_features, double *scaled, const ScalerParams *sp);
double computeAccuracy(const Dataset *dataset, const int *predictions);
double computeAccuracyFromArrays(const int *truth, const int *predictions, int n);
ClassificationReport buildClassificationReport(const Dataset *ds, const int *predictions);
ClassificationReport buildClassificationReportFromArrays(const int *truth, const int *predictions, int n);
const char *labelToStr(int label);
const char *labelToStrCustom(int label, const char *class_zero, const char *class_one);
void printMetricGuide(void);
void printClassificationReport(const char *model_name,
                               const char *split_name,
                               const ClassificationReport *r);
void printClassificationReportWithNames(const char *model_name,
                                        const char *split_name,
                                        const ClassificationReport *r,
                                        const char *class_zero,
                                        const char *class_one);
void printCommonHeader(const char *title_line, const char *subtitle_line);
void printCSVInfo(const char *train_csv, const char *valid_csv, const char *test_csv,
                  const Dataset *train, const Dataset *valid, const Dataset *test);
void printScalerInfo(const ScalerParams *sp);

#endif
