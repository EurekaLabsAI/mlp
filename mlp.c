#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Helper wrapper functions

extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

extern inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

extern inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

extern inline void *calloc_check(size_t nmemb, size_t size, const char *file, int line) {
    void *ptr = calloc(nmemb, size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", nmemb * size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define callocCheck(nmemb, size) calloc_check(nmemb, size, __FILE__, __LINE__)

// -----------------------------------------------------------------------------
// MLP model

// Pictorial representation of the model for:
// batch size 1 (B=1), 4-gram input (T=4), vocab size 3 (V=1), embedding size 2 (E=2), hidden size 4 (H=4)

// [[c1, c2, c3, c4]] | shape: (1, 4)
//   |   |   |   |
//   v   v   v   v (lookup in wte)
// [[[e11, e12], [e21, e22], [e31, e32], [e41, e42]]] | shape: (1, 4, 2)
//   |   |   |   |
//   v   v   v   v (reshape to (1, 8))
// [[e11, e12, e21, e22, e31, e32, e41, e42]] | shape: (1, 8)
//   |   |   |   |
//   v   v   v   v (matmul with fc1_weights, add fc1_bias, tanh)
// [[h1, h2, h3, h4]] | shape: (1, 4)
//   |   |   |   |
//   v   v   v   v (matmul with fc2_weights, add fc2_bias)
// [[l1, l2, l3]] | shape: (1, 3)
//   |   |   |
//   v   v   v (softmax)
// [[p1, p2, p3]] | shape: (1, 3)

typedef struct {
    size_t vocab_size;  // (V) number of tokens in the vocabulary
    size_t context_length;  // (T) number of tokens in the context, e.g. 4 means 4-gram passed as input
    size_t embedding_size;  // (E) size of the token embeddings
    size_t hidden_size;  // (H) size of MLP the hidden layer
    size_t batch_size;  // (B) batch size
} MLPConfig;

#define NUM_PARAMETER_TENSORS 5
typedef struct {
    float *wte;  // (V, E)
    float *fc1_weights;  // (T*E, H)
    float *fc1_bias;  // (H,)
    float *fc2_weights;  // (H, V)
    float *fc2_bias;  // (V,)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, MLPConfig config) {
    size_t V, T, E, H, B;
    V = config.vocab_size;
    T = config.context_length;
    E = config.embedding_size;
    H = config.hidden_size;
    B = config.batch_size;

    param_sizes[0] = V * E;
    param_sizes[1] = T * E * H;
    param_sizes[2] = H;
    param_sizes[3] = H * V;
    param_sizes[4] = V;
}

// allocate memory for the parameters and point the individual pointers to the right locations
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, size_t num_parameters) {
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the pointers to the right locations
    float** ptrs[] = {
        &params->wte, &params->fc1_weights, &params->fc1_bias, &params->fc2_weights, &params->fc2_bias
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 5
typedef struct {
    float* emb; // (B, T*E)
    float* fc1; // (B, H)
    float* h; // (B, H)
    float* logits; // (B, V)
    float* probs; // (B, V)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, MLPConfig config) {
    size_t V, T, E, H, B;
    V = config.vocab_size;
    T = config.context_length;
    E = config.embedding_size;
    H = config.hidden_size;
    B = config.batch_size;

    act_sizes[0] = B * T * E;
    act_sizes[1] = B * H;
    act_sizes[2] = B * H;
    act_sizes[3] = B * V;
    act_sizes[4] = B * V;
}

// allocate memory for the activations and point the individual pointers to the right locations
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes, size_t num_activations) {
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->emb, &acts->fc1, &acts->h, &acts->logits, &acts->probs
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    MLPConfig config;
    // parameters
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    // activations
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // grads
    ParameterTensors grads;
    float* grads_memory;
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
} MLP;

void mlp_build_from_checkpoint(MLP *model, const char *filename) {
    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(filename, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240719) { printf("Bad magic model file\n"); exit(1); }

    // read in hyperparameters
    size_t V, T, E, H;
    model->config.vocab_size = V = model_header[1];
    model->config.context_length = T = model_header[2];
    model->config.embedding_size = E = model_header[3];
    model->config.hidden_size = H = model_header[4];
    printf("[MLP]\n");
    printf("vocab_size: %zu\n", V);
    printf("context_length: %zu\n", T);
    printf("embedding_size: %zu\n", E);
    printf("hidden_size: %zu\n", H);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, num_parameters);
    // read in all the parameters from file
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);
}

void mlp_free(MLP *model) {
    // free up all dynamically allocated memory
    free(model->params_memory);
    free(model->acts_memory);
    free(model->grads_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

// -----------------------------------------------------------------------------
// forward pass

void encoder_forward(float *act_emb, int *inputs, float *weight_wte, int B, int T, int E) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx = inputs[b*T + t];
            // copy the embedding vector of size E for the token at index idx
            memcpy(&act_emb[(b*T + t)*E], &weight_wte[idx*E], E * sizeof(float));
        }
    }
}

void matmul_forward(float *c, float *a, float *b, float *bias, int m, int n, int k) {
    // C = A * B + bias;
    // shapes: (m, k) = (m, n) * (n, k) + (k,); note: (k,) is broadcasted to (m, k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            c[i*k + j] = 0;
            // dot product of row i of A and column j of B
            for (int l = 0; l < n; l++) {
                c[i*k + j] += a[i*n + l] * b[l*k + j];
            }

            // add bias
            c[i*k + j] += bias[j];
        }
    }
}

void tanh_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = tanhf(x[i]);
    }
}

void softmax_forward(float *probs, float *logits, int batch_size, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        float max_val = -INFINITY;
        // find the max value for this sample in the batch - this is to avoid numerical instability
        for (int j = 0; j < vocab_size; j++) {
            if (logits[b*vocab_size + j] > max_val) {
                max_val = logits[b*vocab_size + j];
            }
        }
        float sum = 0.0f;
        // compute the unnormalized probabilities and softmax denominator
        for (int j = 0; j < vocab_size; j++) {
            probs[b*vocab_size + j] = expf(logits[b*vocab_size + j] - max_val);
            sum += probs[b*vocab_size + j];
        }
        // normalize the probabilities
        for (int j = 0; j < vocab_size; j++) {
            probs[b*vocab_size + j] /= sum;
        }
    }
}

float cross_entropy(float *probs, int *targets, int batch_size, int vocab_size) {
    double loss = 0;
    for (int i = 0; i < batch_size; i++) {
        int target = targets[i];
        loss += -logf(probs[i*vocab_size + target]);
    }
    return (float)(loss / (double)batch_size);
}

float forward(MLP *model) {

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience shortcuts
    int B = model->config.batch_size;
    int T = model->config.context_length;
    int E = model->config.embedding_size;
    int H = model->config.hidden_size;
    int V = model->config.vocab_size;

    if (model->acts_memory == NULL) {  // lazy initialization of activations the first time we call forward
        fill_in_activation_sizes(model->act_sizes, model->config);

        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;

        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes, num_activations);
    }

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // encode all the tokens using the embedding table
    // inputs are the input tokens, a (B, T) array of integers
    encoder_forward(acts.emb, model->inputs, params.wte, B, T, E);  // (B, T) -> (B, T, E)

    // forward through the first linear layer
    matmul_forward(acts.h, acts.emb, params.fc1_weights, params.fc1_bias, B, T * E, H);  // (B, T*E) @ (T*E, H) = (B, H)
    tanh_forward(acts.h, B * H);  // (B, H)

    // forward through the second linear layer
    matmul_forward(acts.logits, acts.h, params.fc2_weights, params.fc2_bias, B, H, V);  // (B, H) @ (H, V) = (B, V)
    softmax_forward(acts.probs, acts.logits, B, V);  // (B, V)

    float loss = cross_entropy(acts.probs, model->targets, B, V);  // scalar
    return loss;
}

// -----------------------------------------------------------------------------
// backward pass

void crossentropy_softmax_backward(float* grad_logits, float* act_probs, int* targets, int B, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < V; i++) {
            float p = act_probs[b*V + i];
            float indicator = i == targets[b] ? 1.0f : 0.0f;
            grad_logits[b*V + i] += (p - indicator) * (1.f / B);  // divide by B as the loss is a mean over the batch
        }
    }
}

// simplest possible matmul backward, can be further optimized with pragma directives, etc.
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int C, int OC) {
    // backward into input
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < C; i++) {
            for (int o = 0; o < OC; o++) {
                // dinp = dout @ weight^T
                // (B,C) = (B,OC) @ (OC,C)
                // weight's layout is (C,OC) but here we transpose it using i*OC + o
                dinp[b*C + i] += dout[b*OC + o] * weight[i*OC + o];
            }
        }
    }
    // backward into weight
    for (int i = 0; i < C; i++) {
        for (int o = 0; o < OC; o++) {
            for (int b = 0; b < B; b++) {
                // dweight = inp^T @ dout
                // (C,OC) = (C,B) @ (B,OC)
                // inp's layout is (B,C) but here we transpose it using b*C + i
                dweight[i*OC + o] += inp[b*C + i] * dout[b*OC + o];
            }
        }
    }
    // backward into bias
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            dbias[o] += dout[b*OC + o];
        }
    }
}

void tanh_backward(float* dinp, float* dout, float* inp, int size) {
    for (int i = 0; i < size; i++) {
        dinp[i] = dout[i] * (1 - inp[i] * inp[i]);
    }
}

void encoder_backward(float* dinp,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < C; i++) {
                dinp[inp[b*T + t]*C + i] += dout[b*T*C + t*C + i];
            }
        }
    }
}

void backward(MLP *model) {
    // convenience shortcuts
    int B = model->config.batch_size;
    int T = model->config.context_length;
    int E = model->config.embedding_size;
    int H = model->config.hidden_size;
    int V = model->config.vocab_size;

    if (model->grads_memory == NULL) {  // lazy initialization of gradients the first time we call backward
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, model->num_parameters);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes, model->num_activations);
    }

    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // backprop through softmax and crossentropy
    crossentropy_softmax_backward(grads_acts.logits, acts.probs, model->targets, B, V);

    // backprop through the second linear layer
    matmul_backward(grads_acts.h, grads.fc2_weights, grads.fc2_bias,
                    grads_acts.logits, acts.h, params.fc2_weights, B, H, V);

    // backprop through tanh
    tanh_backward(grads_acts.fc1, grads_acts.h, acts.h, B * H);

    // backprop through the first linear layer
    matmul_backward(grads_acts.emb, grads.fc1_weights, grads.fc1_bias,
                    grads_acts.fc1, acts.emb, params.fc1_weights, B, T * E, H);

    // backprop through the embedding layer
    encoder_backward(grads.wte, grads_acts.emb, model->inputs, B, T, E);
}

void zero_grad(MLP *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

// -----------------------------------------------------------------------------
// AdamW optimizer

typedef struct {
    float lr;  // learning rate
    float beta1;  // exponential decay rate for the first moment estimates
    float beta2;  // exponential decay rate for the second moment estimates
    float weight_decay;  // weight decay (L2 penalty)
    float eps;  // a small constant to avoid division by zero
    int t;  // timestep - used for bias correction
    float *m;  // first moment estimates
    float *v;  // second moment estimates
} AdamW;

void adamw_init(AdamW* optimizer, size_t num_parameters, float lr, float beta1, float beta2, float weight_decay, float eps) {
    optimizer->lr = lr;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->weight_decay = weight_decay;
    optimizer->eps = eps;
    optimizer->t = 0;
    optimizer->m = (float*)callocCheck(num_parameters, sizeof(float));
    optimizer->v = (float*)callocCheck(num_parameters, sizeof(float));
}

void update(AdamW *optimizer, MLP* model) {
    // convenience shortcuts
    float *grads = model->grads_memory;
    float *params = model->params_memory;

    optimizer->t += 1;
    for (int i = 0; i < model->num_parameters; i++) {
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + (1 - optimizer->beta1) * grads[i];  // exponential moving average of the gradient (momentum)
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + (1 - optimizer->beta2) * grads[i] * grads[i];  // exponential moving average of the squared gradient (uncentered variance)
        float m_hat = optimizer->m[i] / (1 - pow(optimizer->beta1, optimizer->t));  // bias correction
        float v_hat = optimizer->v[i] / (1 - pow(optimizer->beta2, optimizer->t));  // bias correction
        params[i] -= optimizer->lr * (m_hat / (sqrt(v_hat) + optimizer->eps) + optimizer->weight_decay * params[i]);  // update with weight decay
    }
}

void adam_free(AdamW *optimizer) {
    // free up all dynamically allocated memory
    free(optimizer->m);
    free(optimizer->v);
}

// -----------------------------------------------------------------------------
// simple DataLoader that iterates over all the n-grams

void dataloader(MLP* model, int *tokens, int context_length, int batch_size, int token_cnt, int *pos) {
    if (model->acts_memory == NULL) {  // lazy initialization the first time we call dataloader
        model->inputs = (int*)mallocCheck(batch_size * model->config.context_length * sizeof(int));
        model->targets = (int*)mallocCheck(batch_size * sizeof(int));
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < context_length; j++) {
            model->inputs[i*context_length + j] = tokens[*pos + j];
        }
        model->targets[i] = tokens[*pos + context_length];
        *pos += 1;
        if (*pos + context_length >= token_cnt) {
            *pos = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// evaluation function

float eval_split(MLP *model, int *tokens, int token_cnt, int max_batches, int batch_size) {
    float total_loss = 0;
    int num_batches = token_cnt / batch_size;
    num_batches = max_batches ? (num_batches < max_batches ? num_batches : max_batches) : num_batches;
    int pos = 0;
    for (int i = 0; i < num_batches; i++) {
        dataloader(model, tokens, model->config.context_length, batch_size, token_cnt, &pos);
        float loss = forward(model);
        total_loss += loss;
    }
    float mean_loss = total_loss / num_batches;
    return mean_loss;
}

// -----------------------------------------------------------------------------
// let's train!

int main() {
    FILE *train_file = fopenCheck("data/train.txt", "r");

    // ensure that the training data only contains lowercase letters and newlines
    char c;
    while ((c = fgetc(train_file)) != EOF) {
        if (c != '\n' && !(c >= 'a' && c <= 'z')) {
            fprintf(stderr, "Error: Invalid character in training data: %c\n", c);
            exit(EXIT_FAILURE);
        }
    }

    fseek(train_file, 0, SEEK_SET);  // Reset the train txt file pointer
    FILE *val_file = fopenCheck("data/val.txt", "r");
    FILE *test_file = fopenCheck("data/test.txt", "r");

    // allocate memory for the tokens
    int TRAIN_SIZE = 213796;
    int VAL_SIZE = 7179;
    int TEST_SIZE = 7170;

    int train_tokens[TRAIN_SIZE];
    int val_tokens[VAL_SIZE];
    int test_tokens[TEST_SIZE];

    int train_token_count = 0;
    int val_token_count = 0;
    int test_token_count = 0;

    // pre-tokenize all the splits one time up here
    // \n -> 0, a -> 1, b -> 2, ..., z -> 26
    while ((c = fgetc(train_file)) != EOF) {
        train_tokens[train_token_count++] = c == '\n' ? 0 : c - 'a' + 1;
    }
    while ((c = fgetc(val_file)) != EOF) {
        val_tokens[val_token_count++] = c == '\n' ? 0 : c - 'a' + 1;
    }
    while ((c = fgetc(test_file)) != EOF) {
        test_tokens[test_token_count++] = c == '\n' ? 0 : c - 'a' + 1;
    }

    // close the files
    fcloseCheck(train_file);
    fcloseCheck(val_file);
    fcloseCheck(test_file);

    // create the model
    MLP model;
    mlp_build_from_checkpoint(&model, "mlp_weights.bin");

    // optimizer
    AdamW optimizer;
    float learning_rate = 1e-3;
    float weight_decay = 1e-4;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    adamw_init(&optimizer, model.num_parameters, learning_rate, beta1, beta2, weight_decay, eps);

    // training loop
    int batch_size = 64;
    int num_steps = 50000;
    model.config.batch_size = batch_size;
    printf("num_steps %d, num_epochs %.2f\n", num_steps, num_steps * batch_size / (float)train_token_count);
    int pos = 0;
    for (int step = 0; step < num_steps; step++) {
        // cosine learning rate schedule, from max lr to 0
        float lr = learning_rate * 0.5 * (1 + cosf(M_PI * step / num_steps));
        // every now and then evaluate the validation loss
        int last_step = step == num_steps - 1;
        if (step % 200 == 0 || last_step) {
            float train_loss = eval_split(&model, train_tokens, train_token_count, 20, batch_size);
            float val_loss = eval_split(&model, val_tokens, val_token_count, 0, batch_size);
            printf("step %d/%d | train_loss %.4f | val_loss %.4f | lr %.6f\n", step, num_steps, train_loss, val_loss, lr);
        }
        // get the next batch of training data
        dataloader(&model, train_tokens, model.config.context_length, batch_size, train_token_count, &pos);
        // forward through the model
        float loss = forward(&model);
        // zero out the gradients so we know we have the right gradient state just before the backward pass
        zero_grad(&model);
        // compute the gradients
        backward(&model);
        // update the weights
        update(&optimizer, &model);
    }

    // free all dynamically allocated memory
    mlp_free(&model);
    adam_free(&optimizer);

    return 0;
}
