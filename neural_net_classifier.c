#include "ml_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FRUIT_NUM_FEATURES     3
#define FRUIT_LABEL_ORANGE     0
#define FRUIT_LABEL_APPLE      1
#define FRUIT_TRAIN_RATIO      0.60
#define FRUIT_VALID_RATIO      0.20
#define FRUIT_SPLIT_SEED       815
#define PERCEPTRON_LR          0.10
#define PERCEPTRON_MAX_EPOCHS  250
#define HOPFIELD_MAX_ITER      25
#define MAXNET_ITER            30

typedef struct {
    double x[FRUIT_NUM_FEATURES];
    int label;
} FruitSample;

typedef struct {
    FruitSample *data;
    int size;
} FruitDataset;

typedef struct {
    int *idx;
    int size;
    int *truth;
} FruitSplit;

typedef struct {
    double w[FRUIT_NUM_FEATURES];
    double b;
} Perceptron;

typedef struct {
    double W[NUM_CLASSES][FRUIT_NUM_FEATURES];
    double bias[NUM_CLASSES];
    double proto[NUM_CLASSES][FRUIT_NUM_FEATURES];
    double epsilon;
} HammingNet;

typedef struct {
    double W[FRUIT_NUM_FEATURES][FRUIT_NUM_FEATURES];
    double proto[NUM_CLASSES][FRUIT_NUM_FEATURES];
} HopfieldNet;

typedef struct {
    int spurious_count;
} HopfieldEvalInfo;

static int parseFruitLabel(const char *token)
{
    if (strcmp(token, "ORANGE") == 0 || strcmp(token, "orange") == 0 ||
        strcmp(token, "1") == 0 || strcmp(token, "+1") == 0) {
        return FRUIT_LABEL_ORANGE;
    }
    if (strcmp(token, "APPLE") == 0 || strcmp(token, "apple") == 0 ||
        strcmp(token, "-1") == 0 || strcmp(token, "0") == 0) {
        return FRUIT_LABEL_APPLE;
    }

    fprintf(stderr, "FATAL: unknown fruit label '%s'\n", token);
    exit(EXIT_FAILURE);
}

static FruitDataset loadFruitDataset(const char *filename)
{
    FruitDataset ds;
    ds.data = NULL;
    ds.size = 0;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "FATAL: could not open fruit CSV file '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    int capacity = INITIAL_CAPACITY;
    ds.data = (FruitSample *)malloc(capacity * sizeof(FruitSample));
    if (!ds.data) {
        fprintf(stderr, "FATAL: malloc failed in loadFruitDataset\n");
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    char line[256];
    int line_no = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        line_no++;
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') continue;

        char *shape_tok = strtok(line, ",");
        char *texture_tok = strtok(NULL, ",");
        char *weight_tok = strtok(NULL, ",");
        char *label_tok = strtok(NULL, ",\r\n");

        if (!shape_tok || !texture_tok || !weight_tok || !label_tok) {
            fprintf(stderr, "FATAL: invalid fruit CSV format in %s at line %d\n", filename, line_no);
            free(ds.data);
            fclose(fp);
            exit(EXIT_FAILURE);
        }

        if (line_no == 1) {
            char *endptr = NULL;
            strtod(shape_tok, &endptr);
            if (endptr == shape_tok || (*endptr != '\0' && *endptr != '\r' && *endptr != '\n')) {
                continue;
            }
        }

        if (ds.size == capacity) {
            capacity *= 2;
            FruitSample *tmp = (FruitSample *)realloc(ds.data, capacity * sizeof(FruitSample));
            if (!tmp) {
                fprintf(stderr, "FATAL: realloc failed in loadFruitDataset\n");
                free(ds.data);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
            ds.data = tmp;
        }

        ds.data[ds.size].x[0] = atof(shape_tok);
        ds.data[ds.size].x[1] = atof(texture_tok);
        ds.data[ds.size].x[2] = atof(weight_tok);
        ds.data[ds.size].label = parseFruitLabel(label_tok);
        ds.size++;
    }

    fclose(fp);

    if (ds.size == 0) {
        fprintf(stderr, "FATAL: fruit CSV file '%s' has no usable rows\n", filename);
        free(ds.data);
        exit(EXIT_FAILURE);
    }

    return ds;
}

static void freeFruitDataset(FruitDataset *ds)
{
    free(ds->data);
    ds->data = NULL;
    ds->size = 0;
}

static void shuffleIndices(int *idx, int n)
{
    srand(FRUIT_SPLIT_SEED);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

static FruitSplit makeSplit(const FruitDataset *ds, const int *indices, int start, int count)
{
    FruitSplit split;
    split.size = count;
    split.idx = (int *)malloc(count * sizeof(int));
    split.truth = (int *)malloc(count * sizeof(int));
    if (!split.idx || !split.truth) {
        fprintf(stderr, "FATAL: malloc failed while creating fruit split\n");
        free(split.idx);
        free(split.truth);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < count; i++) {
        split.idx[i] = indices[start + i];
        split.truth[i] = ds->data[split.idx[i]].label;
    }
    return split;
}

static void freeFruitSplit(FruitSplit *split)
{
    free(split->idx);
    free(split->truth);
    split->idx = NULL;
    split->truth = NULL;
    split->size = 0;
}

static int toBipolarLabel(int label)
{
    return (label == FRUIT_LABEL_ORANGE) ? +1 : -1;
}

static int bipolarStep(double net)
{
    return (net >= 0.0) ? +1 : -1;
}

static void computeClassPrototypes(const FruitDataset *ds,
                                   const FruitSplit *train,
                                   double proto_orange[FRUIT_NUM_FEATURES],
                                   double proto_apple[FRUIT_NUM_FEATURES])
{
    double orange_sum[FRUIT_NUM_FEATURES] = {0.0, 0.0, 0.0};
    double apple_sum[FRUIT_NUM_FEATURES] = {0.0, 0.0, 0.0};
    int orange_count = 0;
    int apple_count = 0;

    for (int i = 0; i < train->size; i++) {
        const FruitSample *s = &ds->data[train->idx[i]];
        if (s->label == FRUIT_LABEL_ORANGE) {
            orange_count++;
            for (int f = 0; f < FRUIT_NUM_FEATURES; f++) orange_sum[f] += s->x[f];
        } else {
            apple_count++;
            for (int f = 0; f < FRUIT_NUM_FEATURES; f++) apple_sum[f] += s->x[f];
        }
    }

    for (int f = 0; f < FRUIT_NUM_FEATURES; f++) {
        proto_orange[f] = (safeDivide(orange_sum[f], (double)orange_count) >= 0.0) ? +1.0 : -1.0;
        proto_apple[f] = (safeDivide(apple_sum[f], (double)apple_count) >= 0.0) ? +1.0 : -1.0;
    }
}

static void perceptronInit(Perceptron *p)
{
    for (int f = 0; f < FRUIT_NUM_FEATURES; f++) p->w[f] = 0.0;
    p->b = 0.0;
}

static int perceptronPredictRaw(const Perceptron *p, const double x[FRUIT_NUM_FEATURES])
{
    double net = p->b;
    for (int f = 0; f < FRUIT_NUM_FEATURES; f++) net += p->w[f] * x[f];
    return (bipolarStep(net) >= 0) ? FRUIT_LABEL_ORANGE : FRUIT_LABEL_APPLE;
}

static double evaluatePerceptronAccuracy(const Perceptron *p, const FruitDataset *ds, const FruitSplit *split)
{
    int correct = 0;
    for (int i = 0; i < split->size; i++) {
        int pred = perceptronPredictRaw(p, ds->data[split->idx[i]].x);
        if (pred == split->truth[i]) correct++;
    }
    return safeDivide(100.0 * correct, (double)split->size);
}

static Perceptron trainPerceptron(const FruitDataset *ds, const FruitSplit *train, const FruitSplit *valid)
{
    Perceptron current;
    Perceptron best;
    perceptronInit(&current);
    best = current;

    double best_valid_acc = -1.0;

    for (int epoch = 0; epoch < PERCEPTRON_MAX_EPOCHS; epoch++) {
        int updates = 0;

        for (int i = 0; i < train->size; i++) {
            const FruitSample *s = &ds->data[train->idx[i]];
            int y = toBipolarLabel(s->label);
            int pred_bipolar = (perceptronPredictRaw(&current, s->x) == FRUIT_LABEL_ORANGE) ? +1 : -1;

            if (pred_bipolar != y) {
                for (int f = 0; f < FRUIT_NUM_FEATURES; f++) current.w[f] += PERCEPTRON_LR * y * s->x[f];
                current.b += PERCEPTRON_LR * y;
                updates++;
            }
        }

        double valid_acc = evaluatePerceptronAccuracy(&current, ds, valid);
        if (valid_acc > best_valid_acc + 1e-12) {
            best_valid_acc = valid_acc;
            best = current;
        }

        if (updates == 0) break;
    }

    return best;
}

static void collectPerceptronPredictions(const Perceptron *p,
                                         const FruitDataset *ds,
                                         const FruitSplit *split,
                                         int *predictions)
{
    for (int i = 0; i < split->size; i++) {
        predictions[i] = perceptronPredictRaw(p, ds->data[split->idx[i]].x);
    }
}

static void hammingInit(HammingNet *h,
                        const double proto_orange[FRUIT_NUM_FEATURES],
                        const double proto_apple[FRUIT_NUM_FEATURES])
{
    for (int f = 0; f < FRUIT_NUM_FEATURES; f++) {
        h->W[0][f] = proto_orange[f] / 2.0;
        h->W[1][f] = proto_apple[f] / 2.0;
        h->proto[0][f] = proto_orange[f];
        h->proto[1][f] = proto_apple[f];
    }
    h->bias[0] = h->bias[1] = (double)FRUIT_NUM_FEATURES / 2.0;
    h->epsilon = 1.0 / (NUM_CLASSES + 1.0);
}

static int hammingPredict(const HammingNet *h, const double x[FRUIT_NUM_FEATURES])
{
    double a[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        a[c] = h->bias[c];
        for (int f = 0; f < FRUIT_NUM_FEATURES; f++) a[c] += h->W[c][f] * x[f];
    }

    for (int iter = 0; iter < MAXNET_ITER; iter++) {
        double next_a[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++) {
            double inhibit = 0.0;
            for (int k = 0; k < NUM_CLASSES; k++) {
                if (k != c) inhibit += a[k];
            }
            next_a[c] = a[c] - h->epsilon * inhibit;
            if (next_a[c] < 0.0) next_a[c] = 0.0;
        }
        memcpy(a, next_a, sizeof(a));
        int active = 0;
        for (int c = 0; c < NUM_CLASSES; c++) if (a[c] > 0.0) active++;
        if (active <= 1) break;
    }

    return (a[0] >= a[1]) ? FRUIT_LABEL_ORANGE : FRUIT_LABEL_APPLE;
}

static void collectHammingPredictions(const HammingNet *h,
                                      const FruitDataset *ds,
                                      const FruitSplit *split,
                                      int *predictions)
{
    for (int i = 0; i < split->size; i++) {
        predictions[i] = hammingPredict(h, ds->data[split->idx[i]].x);
    }
}

static void hopfieldTrain(HopfieldNet *hf,
                          const double proto_orange[FRUIT_NUM_FEATURES],
                          const double proto_apple[FRUIT_NUM_FEATURES])
{
    const double *protos[NUM_CLASSES] = {proto_orange, proto_apple};

    for (int i = 0; i < FRUIT_NUM_FEATURES; i++) {
        for (int j = 0; j < FRUIT_NUM_FEATURES; j++) {
            hf->W[i][j] = 0.0;
        }
    }

    for (int k = 0; k < NUM_CLASSES; k++) {
        for (int i = 0; i < FRUIT_NUM_FEATURES; i++) {
            for (int j = 0; j < FRUIT_NUM_FEATURES; j++) {
                if (i != j) hf->W[i][j] += protos[k][i] * protos[k][j];
            }
        }
    }

    for (int i = 0; i < FRUIT_NUM_FEATURES; i++) {
        for (int j = 0; j < FRUIT_NUM_FEATURES; j++) {
            hf->W[i][j] /= FRUIT_NUM_FEATURES;
        }
    }

    for (int f = 0; f < FRUIT_NUM_FEATURES; f++) {
        hf->proto[0][f] = proto_orange[f];
        hf->proto[1][f] = proto_apple[f];
    }
}

static int nearestPrototypeLabel(const HopfieldNet *hf, const double state[FRUIT_NUM_FEATURES])
{
    int best_label = FRUIT_LABEL_ORANGE;
    int best_match = -1;

    for (int c = 0; c < NUM_CLASSES; c++) {
        int match = 0;
        for (int f = 0; f < FRUIT_NUM_FEATURES; f++) {
            if ((int)state[f] == (int)hf->proto[c][f]) match++;
        }
        if (match > best_match) {
            best_match = match;
            best_label = (c == 0) ? FRUIT_LABEL_ORANGE : FRUIT_LABEL_APPLE;
        }
    }

    return best_label;
}

static int hopfieldPredict(const HopfieldNet *hf,
                           const double x[FRUIT_NUM_FEATURES],
                           int *was_spurious)
{
    double state[FRUIT_NUM_FEATURES];
    memcpy(state, x, sizeof(state));

    for (int iter = 0; iter < HOPFIELD_MAX_ITER; iter++) {
        double next_state[FRUIT_NUM_FEATURES];
        int changed = 0;

        for (int i = 0; i < FRUIT_NUM_FEATURES; i++) {
            double net = 0.0;
            for (int j = 0; j < FRUIT_NUM_FEATURES; j++) net += hf->W[i][j] * state[j];
            next_state[i] = bipolarStep(net);
            if ((int)next_state[i] != (int)state[i]) changed++;
        }

        memcpy(state, next_state, sizeof(state));
        if (!changed) break;
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        int exact_match = 1;
        for (int f = 0; f < FRUIT_NUM_FEATURES; f++) {
            if ((int)state[f] != (int)hf->proto[c][f]) {
                exact_match = 0;
                break;
            }
        }
        if (exact_match) {
            if (was_spurious) *was_spurious = 0;
            return (c == 0) ? FRUIT_LABEL_ORANGE : FRUIT_LABEL_APPLE;
        }
    }

    if (was_spurious) *was_spurious = 1;
    return nearestPrototypeLabel(hf, state);
}

static void collectHopfieldPredictions(const HopfieldNet *hf,
                                       const FruitDataset *ds,
                                       const FruitSplit *split,
                                       int *predictions,
                                       HopfieldEvalInfo *info)
{
    info->spurious_count = 0;
    for (int i = 0; i < split->size; i++) {
        int was_spurious = 0;
        predictions[i] = hopfieldPredict(hf, ds->data[split->idx[i]].x, &was_spurious);
        info->spurious_count += was_spurious;
    }
}

static void printFruitHeader(const char *dataset_csv, const FruitDataset *ds,
                             const FruitSplit *train, const FruitSplit *valid, const FruitSplit *test)
{
    printf("\n  Loaded CSV file:\n");
    printf("    Fruit dataset : %s\n", dataset_csv);
    printf("\n  Dataset summary:\n");
    printf("    Total samples : %d\n", ds->size);
    printf("    Train         : %d\n", train->size);
    printf("    Validate      : %d\n", valid->size);
    printf("    Test          : %d\n", test->size);
}

static void printFruitPartA(const double proto_orange[FRUIT_NUM_FEATURES],
                            const double proto_apple[FRUIT_NUM_FEATURES])
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  PART A  —  CLASS PROTOTYPES FROM TRAIN SPLIT              \n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  Orange prototype : [%+.0f %+.0f %+.0f]\n",
           proto_orange[0], proto_orange[1], proto_orange[2]);
    printf("  Apple prototype  : [%+.0f %+.0f %+.0f]\n",
           proto_apple[0], proto_apple[1], proto_apple[2]);
}

static void interactiveFruitCLI(const Perceptron *p, const HammingNet *h, const HopfieldNet *hf)
{
    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  FRUIT NETWORK INTERACTIVE PREDICTION  (q to quit)         \n");
    printf("══════════════════════════════════════════════════════════════\n");

    char buf[64];

    while (1) {
        int shape, texture, weight;
        int choice;

        printf("\n  Choose network [1=Perceptron, 2=Hamming, 3=Hopfield, 0=quit]: ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%d", &choice) != 1 || choice < 0 || choice > 3) {
            printf("  [!] Invalid option.\n");
            continue;
        }
        if (choice == 0) break;

        printf("  Enter shape   (+1 or -1): ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%d", &shape) != 1 || (shape != 1 && shape != -1)) {
            printf("  [!] Shape must be +1 or -1.\n");
            continue;
        }

        printf("  Enter texture (+1 or -1): ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%d", &texture) != 1 || (texture != 1 && texture != -1)) {
            printf("  [!] Texture must be +1 or -1.\n");
            continue;
        }

        printf("  Enter weight  (+1 or -1): ");
        if (fgets(buf, sizeof(buf), stdin) == NULL) break;
        if (buf[0] == 'q' || buf[0] == 'Q') break;
        if (sscanf(buf, "%d", &weight) != 1 || (weight != 1 && weight != -1)) {
            printf("  [!] Weight must be +1 or -1.\n");
            continue;
        }

        double x[FRUIT_NUM_FEATURES] = {(double)shape, (double)texture, (double)weight};
        int pred = FRUIT_LABEL_ORANGE;
        const char *network_name = "Perceptron";
        int spurious = 0;

        if (choice == 1) {
            pred = perceptronPredictRaw(p, x);
            network_name = "Perceptron";
        } else if (choice == 2) {
            pred = hammingPredict(h, x);
            network_name = "Hamming";
        } else {
            pred = hopfieldPredict(hf, x, &spurious);
            network_name = "Hopfield";
        }

        printf("\n");
        printf("  ┌─────────────────────────────────────────────────┐\n");
        printf("  │              FRUIT CLASSIFICATION               │\n");
        printf("  ├──────────────┬──────────────────────────────────┤\n");
        printf("  │ Network      │ %-32s │\n", network_name);
        printf("  │ shape        │ %+d                                │\n", shape);
        printf("  │ texture      │ %+d                                │\n", texture);
        printf("  │ weight       │ %+d                                │\n", weight);
        printf("  ├──────────────┼──────────────────────────────────┤\n");
        printf("  │ Prediction   │ %-32s │\n", labelToStrCustom(pred, "ORANGE", "APPLE"));
        if (choice == 3) {
            printf("  │ Spurious     │ %-32s │\n", spurious ? "YES (nearest prototype used)" : "NO");
        }
        printf("  └──────────────┴──────────────────────────────────┘\n");
    }

    printf("\n  Goodbye!\n\n");
}

int main(int argc, char *argv[])
{
    printCommonHeader("FRUIT CLASSIFIER  —  PERCEPTRON / HAMMING / HOPFIELD  (C99)",
                      "Single fruit dataset with deterministic train / validate / test split");

    const char *fruit_csv = (argc >= 2) ? argv[1] : DEFAULT_FRUIT_CSV;
    FruitDataset ds = loadFruitDataset(fruit_csv);

    int *indices = (int *)malloc(ds.size * sizeof(int));
    if (!indices) {
        fprintf(stderr, "FATAL: malloc failed for fruit indices\n");
        freeFruitDataset(&ds);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < ds.size; i++) indices[i] = i;
    shuffleIndices(indices, ds.size);

    int train_size = (int)(ds.size * FRUIT_TRAIN_RATIO);
    int valid_size = (int)(ds.size * FRUIT_VALID_RATIO);
    int test_size = ds.size - train_size - valid_size;

    FruitSplit train = makeSplit(&ds, indices, 0, train_size);
    FruitSplit valid = makeSplit(&ds, indices, train_size, valid_size);
    FruitSplit test  = makeSplit(&ds, indices, train_size + valid_size, test_size);

    free(indices);

    printFruitHeader(fruit_csv, &ds, &train, &valid, &test);

    double proto_orange[FRUIT_NUM_FEATURES];
    double proto_apple[FRUIT_NUM_FEATURES];
    computeClassPrototypes(&ds, &train, proto_orange, proto_apple);
    printFruitPartA(proto_orange, proto_apple);

    printMetricGuide();

    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  PART B  —  PERCEPTRON NETWORK                           \n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  Learning rate : %.2f\n", PERCEPTRON_LR);
    printf("  Max epochs    : %d\n", PERCEPTRON_MAX_EPOCHS);

    Perceptron p = trainPerceptron(&ds, &train, &valid);
    printf("  Final weights : [%.4f, %.4f, %.4f]  bias = %.4f\n",
           p.w[0], p.w[1], p.w[2], p.b);

    int *train_preds = (int *)malloc(train.size * sizeof(int));
    int *valid_preds = (int *)malloc(valid.size * sizeof(int));
    int *test_preds  = (int *)malloc(test.size  * sizeof(int));
    if (!train_preds || !valid_preds || !test_preds) {
        fprintf(stderr, "FATAL: malloc failed for neural prediction arrays\n");
        free(train_preds);
        free(valid_preds);
        free(test_preds);
        freeFruitSplit(&train);
        freeFruitSplit(&valid);
        freeFruitSplit(&test);
        freeFruitDataset(&ds);
        return EXIT_FAILURE;
    }

    collectPerceptronPredictions(&p, &ds, &train, train_preds);
    collectPerceptronPredictions(&p, &ds, &valid, valid_preds);
    collectPerceptronPredictions(&p, &ds, &test, test_preds);

    ClassificationReport perceptron_train = buildClassificationReportFromArrays(train.truth, train_preds, train.size);
    ClassificationReport perceptron_valid = buildClassificationReportFromArrays(valid.truth, valid_preds, valid.size);
    ClassificationReport perceptron_test  = buildClassificationReportFromArrays(test.truth,  test_preds,  test.size);

    printClassificationReportWithNames("Perceptron", "Train report", &perceptron_train, "ORANGE", "APPLE");
    printClassificationReportWithNames("Perceptron", "Validation report", &perceptron_valid, "ORANGE", "APPLE");
    printClassificationReportWithNames("Perceptron", "Test report", &perceptron_test, "ORANGE", "APPLE");

    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  PART C  —  HAMMING NETWORK                              \n");
    printf("══════════════════════════════════════════════════════════════\n");
    HammingNet h;
    hammingInit(&h, proto_orange, proto_apple);
    printf("  Feedforward weights (prototype / 2):\n");
    printf("    Orange row : [%+.2f %+.2f %+.2f]\n", h.W[0][0], h.W[0][1], h.W[0][2]);
    printf("    Apple row  : [%+.2f %+.2f %+.2f]\n", h.W[1][0], h.W[1][1], h.W[1][2]);

    collectHammingPredictions(&h, &ds, &train, train_preds);
    collectHammingPredictions(&h, &ds, &valid, valid_preds);
    collectHammingPredictions(&h, &ds, &test, test_preds);

    ClassificationReport hamming_train = buildClassificationReportFromArrays(train.truth, train_preds, train.size);
    ClassificationReport hamming_valid = buildClassificationReportFromArrays(valid.truth, valid_preds, valid.size);
    ClassificationReport hamming_test  = buildClassificationReportFromArrays(test.truth,  test_preds,  test.size);

    printClassificationReportWithNames("Hamming", "Train report", &hamming_train, "ORANGE", "APPLE");
    printClassificationReportWithNames("Hamming", "Validation report", &hamming_valid, "ORANGE", "APPLE");
    printClassificationReportWithNames("Hamming", "Test report", &hamming_test, "ORANGE", "APPLE");

    printf("\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  PART D  —  HOPFIELD NETWORK                             \n");
    printf("══════════════════════════════════════════════════════════════\n");
    HopfieldNet hf;
    hopfieldTrain(&hf, proto_orange, proto_apple);
    printf("  Weight matrix W:\n");
    for (int i = 0; i < FRUIT_NUM_FEATURES; i++) {
        printf("    [ ");
        for (int j = 0; j < FRUIT_NUM_FEATURES; j++) {
            printf("%+.4f ", hf.W[i][j]);
        }
        printf("]\n");
    }

    HopfieldEvalInfo hop_train_info;
    HopfieldEvalInfo hop_valid_info;
    HopfieldEvalInfo hop_test_info;

    collectHopfieldPredictions(&hf, &ds, &train, train_preds, &hop_train_info);
    collectHopfieldPredictions(&hf, &ds, &valid, valid_preds, &hop_valid_info);
    collectHopfieldPredictions(&hf, &ds, &test, test_preds, &hop_test_info);

    ClassificationReport hopfield_train = buildClassificationReportFromArrays(train.truth, train_preds, train.size);
    ClassificationReport hopfield_valid = buildClassificationReportFromArrays(valid.truth, valid_preds, valid.size);
    ClassificationReport hopfield_test  = buildClassificationReportFromArrays(test.truth,  test_preds,  test.size);

    printClassificationReportWithNames("Hopfield", "Train report", &hopfield_train, "ORANGE", "APPLE");
    printf("  Spurious recalls on train split      : %d\n", hop_train_info.spurious_count);
    printClassificationReportWithNames("Hopfield", "Validation report", &hopfield_valid, "ORANGE", "APPLE");
    printf("  Spurious recalls on validation split : %d\n", hop_valid_info.spurious_count);
    printClassificationReportWithNames("Hopfield", "Test report", &hopfield_test, "ORANGE", "APPLE");
    printf("  Spurious recalls on test split       : %d\n", hop_test_info.spurious_count);

    free(train_preds);
    free(valid_preds);
    free(test_preds);

    interactiveFruitCLI(&p, &h, &hf);

    freeFruitSplit(&train);
    freeFruitSplit(&valid);
    freeFruitSplit(&test);
    freeFruitDataset(&ds);
    return EXIT_SUCCESS;
}
