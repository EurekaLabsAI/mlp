/*
The C version of our MLP.
Compile and run as:

$ gcc -O3 -o mlp mlp.c && ./mlp
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

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
// Random number utilities

// Box-Muller transform function
void box_muller_transform(float u1, float u2, float *z1, float *z2) {
    // This is using the Basic form of the Box-Muller transform
    // u1 and u2 are simple floats in [0, 1)
    // z1 and z2 are standard normal random variables
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;
    *z1 = r * cosf(theta);
    *z2 = r * sinf(theta);
}

// RNG class equivalent
typedef struct {
    uint64_t state;
} RNG;

void rng_init(RNG *rng, uint64_t seed) {
    rng->state = seed;
}

uint32_t rng_random_u32(RNG *rng) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng->state ^= (rng->state >> 12);
    rng->state ^= (rng->state << 25);
    rng->state ^= (rng->state >> 27);
    return ((rng->state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF;
}

float rng_random(RNG *rng) {
    // random float32 from Uniform(0, 1), i.e. interval [0, 1)
    return (float)((rng_random_u32(rng) >> 8) / 16777216.0);
}

void rng_rand(RNG *rng, int n, float a, float b, float *out) {
    // return n random float32 from Uniform(a, b), in a list
    for (int i = 0; i < n; ++i) {
        out[i] = rng_random(rng) * (b - a) + a;
    }
}

void rng_randn(RNG *rng, int n, float mu, float sigma, float *out) {
    // return n random float32 from Normal(0, 1), in a list
    // (note box-muller transform returns two numbers at a time)
    for (int i = 0; i < (n + 1) / 2; ++i) {
        float u1 = rng_random(rng);
        float u2 = rng_random(rng);
        float z1, z2;
        box_muller_transform(u1, u2, &z1, &z2);
        out[2 * i] = z1 * sigma + mu;
        if (2 * i + 1 < n) {
            out[2 * i + 1] = z2 * sigma + mu;
        }
    }
}

// -----------------------------------------------------------------------------
// Timer utilities

typedef struct {
    double ema_alpha;
    double ema_time;
    double corrected_ema_time;
    clock_t start_time;
    int step;
} StepTimer;

void init_timer(StepTimer *timer, double ema_alpha) {
    timer->ema_alpha = ema_alpha;
    timer->ema_time = 0;
    timer->corrected_ema_time = 0.0;
    timer->step = 0;
}

void start_timer(StepTimer *timer) {
    timer->start_time = clock();
}

void stop_timer(StepTimer *timer) {
    clock_t end_time = clock();
    double iteration_time = (double)(end_time - timer->start_time) / CLOCKS_PER_SEC;
    timer->ema_time = timer->ema_alpha * timer->ema_time + (1 - timer->ema_alpha) * iteration_time;
    timer->step += 1;
    timer->corrected_ema_time = timer->ema_time / (1 - pow(timer->ema_alpha, timer->step));
}

double get_dt(StepTimer *timer) {
    return timer->corrected_ema_time;
}

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
    int vocab_size;      // (V) number of tokens in the vocabulary
    int context_length;  // (T) number of tokens in the context, e.g. 4 means 4-gram passed as input
    int embedding_size;  // (E) size of the token embeddings
    int hidden_size;     // (H) size of MLP the hidden layer
} MLPConfig;

#define NUM_PARAMETER_TENSORS 5
typedef struct {
    float *wte;          // (V, E)
    float *fc1_weights;  // (T*E, H)
    float *fc1_bias;     // (H,)
    float *fc2_weights;  // (H, V)
    float *fc2_bias;     // (V,)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, MLPConfig config) {
    // casting to size_t so that the arithmetic below does not overflow
    size_t V, T, E, H, B;
    V = (size_t) config.vocab_size;
    T = (size_t) config.context_length;
    E = (size_t) config.embedding_size;
    H = (size_t) config.hidden_size;
    param_sizes[0] = V * E;
    param_sizes[1] = T * E * H;
    param_sizes[2] = H;
    param_sizes[3] = H * V;
    param_sizes[4] = V;
}

// allocate memory for the parameters once, then point the individual pointers to the right locations
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, size_t num_parameters) {
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the pointers to the right locations
    float** ptrs[] = {
        &params->wte, &params->fc1_weights, &params->fc1_bias, &params->fc2_weights, &params->fc2_bias
    };
    float* params_memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 5
typedef struct {
    float* emb;     // (B, T*E)
    float* fc1;     // (B, H)
    float* h;       // (B, H)
    float* logits;  // (B, V)
    float* probs;   // (B, V)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, MLPConfig config, int batch_size) {
    // casting to size_t so that the arithmetic below does not overflow
    size_t V, T, E, H, B;
    V = (size_t) config.vocab_size;
    T = (size_t) config.context_length;
    E = (size_t) config.embedding_size;
    H = (size_t) config.hidden_size;
    B = (size_t) batch_size;
    // the sizes of the activation tensors
    act_sizes[0] = B * T * E;
    act_sizes[1] = B * H;
    act_sizes[2] = B * H;
    act_sizes[3] = B * V;
    act_sizes[4] = B * V;
}

// allocate memory for the activations and point the individual pointers to the right locations
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes, size_t num_activations) {
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {&acts->emb, &acts->fc1, &acts->h, &acts->logits, &acts->probs};
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
    int batch_size; // passed in at the time of forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
} MLP;

void mlp_random_init(MLP *model, RNG *init_rng) {
    size_t V = (size_t) model->config.vocab_size;
    size_t T = (size_t) model->config.context_length;
    size_t E = (size_t) model->config.embedding_size;
    size_t H = (size_t) model->config.hidden_size;
    printf("[MLP]\n");
    printf("vocab_size: %zu\n", V);
    printf("context_length: %zu\n", T);
    printf("embedding_size: %zu\n", E);
    printf("hidden_size: %zu\n", H);
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters and allocate their memory
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, num_parameters);

    // Let's match the PyTorch default initialization:
    // Embedding init with N(0,1)
    rng_randn(init_rng, model->param_sizes[0], 0.0f, 1.0f, model->params.wte);

    // Linear (both W,b) with U(-K, K) where K = 1/sqrt(fan_in)
    // This part gets a bit tricky only because we want to have identical init to PyTorch.
    // // we need (T*E, H) in order to keep equivalence with PyTorch but we have (H, T*E)
    // => right after we generate the weights, transpose them.
    assert(model->param_sizes[1] == E * T * H);
    float* tmp_buffer = (float*)mallocCheck(E * T * H * sizeof(float));
    float k1 = 1.0f / sqrtf(E * T);
    rng_rand(init_rng, model->param_sizes[1], -k1, k1, tmp_buffer);
    // set fc1_weights as transposed tmp_buffer (both are row-major)
    for (size_t i = 0; i < T * E; i++) {
        for (size_t j = 0; j < H; j++) {
            model->params.fc1_weights[i * H + j] = tmp_buffer[j * T * E + i];
        }
    }
    free(tmp_buffer);
    // PyTorch initializes the biases of Linear layers also with U(-K, K)
    rng_rand(init_rng, model->param_sizes[2], -k1, k1, model->params.fc1_bias);

    // Second Linear, same as above
    assert(model->param_sizes[3] == H * V);
    float* tmp_buffer2 = (float*)mallocCheck(H * V * sizeof(float));
    float k2 = 1.0f / sqrtf(model->config.hidden_size);
    rng_rand(init_rng, model->param_sizes[3], -k2, k2, tmp_buffer2);
    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < V; j++) {
            model->params.fc2_weights[i * V + j] = tmp_buffer2[j * H + i];
        }
    }
    free(tmp_buffer2);
    rng_rand(init_rng, model->param_sizes[4], -k2, k2, model->params.fc2_bias);

    // explicitly initialize all of these pointers to NULL, as this is not guaranteed default on all platforms
    // these will all be initialized lazily when/if we call forward/backward/update functions
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0; // passed in at the time of the first forward pass
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
// the individual layer forward/backward passes
// TODO/NOTE: we're not being super careful with our size_t <-> int
// doing a lot of arithmetic here (for convenience) that could overflow in int for large models

void encoder_forward(float *out, const int *inputs, const float *wte, const int B, const int T, const int E) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx = inputs[b*T + t];
            // copy the embedding vector of size E for the token at index idx
            memcpy(&out[(b*T + t)*E], &wte[idx*E], E * sizeof(float));
        }
    }
}

void encoder_backward(float* dwte, const float* dout, const int* inputs, const int B, const int T, const int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx = inputs[b*T + t];
            for (int i = 0; i < C; i++) {
                dwte[idx*C + i] += dout[b*T*C + t*C + i];
            }
        }
    }
}

void matmul_forward(float *out, float *a, float *b, float *bias, int m, int n, int k) {
    // out = A * B + bias;
    // shapes: (m, k) = (m, n) * (n, k) + (k,); note: (k,) is broadcasted to (m, k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            out[i*k + j] = 0;
            // dot product of row i of A and column j of B
            for (int l = 0; l < n; l++) {
                out[i*k + j] += a[i*n + l] * b[l*k + j];
            }
            // add bias
            out[i*k + j] += bias[j];
        }
    }
}

// simplest possible matmul backward, can be further optimized with pragma directives, etc.
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int C, int OC) {
    // note that all of these could be done in a single loop but we do them
    // separately because we'd want to later parallelize in different dimensions for each one.
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

void tanh_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = tanhf(x[i]);
    }
}

void tanh_backward(float* dinp, float* dout, float* inp, int size) {
    for (int i = 0; i < size; i++) {
        float inpi = inp[i];
        dinp[i] = dout[i] * (1 - inpi * inpi);
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
    double loss = 0; // note: doing accumulation in double for better precision
    for (int i = 0; i < batch_size; i++) {
        int target = targets[i];
        loss += (double) -logf(probs[i*vocab_size + target]);
    }
    return (float)(loss / (double)batch_size);
}

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

// -----------------------------------------------------------------------------
// the model forward/backward pass

float forward(MLP *model, int batch_size) {

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // lazy initialization of activations the first time we call forward
    if (model->acts_memory == NULL) {
        fill_in_activation_sizes(model->act_sizes, model->config, batch_size);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes, num_activations);
        model->batch_size = batch_size;
    } else {
        // validate that the batch size the caller is consistent with the amount of memory we allocated before
        // we can tolerate no more than model->batch_size batch, or we'd need to reallocate memory
        assert(batch_size > 0 && batch_size <= model->batch_size);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->config.context_length;
    int E = model->config.embedding_size;
    int H = model->config.hidden_size;
    int V = model->config.vocab_size;
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
    float loss = -1.0f;
    if (model->targets != NULL) {
        loss = cross_entropy(acts.probs, model->targets, B, V);
    }
    return loss;
}

void backward(MLP *model) {

    // lazy initialization of gradients the first time we call backward
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, model->num_parameters);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes, model->num_activations);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->config.context_length;
    int E = model->config.embedding_size;
    int H = model->config.hidden_size;
    int V = model->config.vocab_size;
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
        // exponential moving average of the gradient (momentum)
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + (1 - optimizer->beta1) * grads[i];
        // exponential moving average of the squared gradient (uncentered variance)
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + (1 - optimizer->beta2) * grads[i] * grads[i];
        // bias correction
        float m_hat = optimizer->m[i] / (1 - pow(optimizer->beta1, optimizer->t));
        float v_hat = optimizer->v[i] / (1 - pow(optimizer->beta2, optimizer->t));
        // update with weight decay
        params[i] -= optimizer->lr * (m_hat / (sqrt(v_hat) + optimizer->eps) + optimizer->weight_decay * params[i]);
    }
}

void adam_free(AdamW *optimizer) {
    free(optimizer->m);
    free(optimizer->v);
}

// -----------------------------------------------------------------------------
// simple DataLoader that iterates over all the n-grams

void dataloader(MLP* model, int *tokens, int context_length, int batch_size, int token_cnt, int *pos) {
    if (model->inputs == NULL) {  // lazy initialization the first time we call dataloader
        model->inputs = (int*)mallocCheck(batch_size * model->config.context_length * sizeof(int));
    }
    if (model->targets == NULL) {
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
        float loss = forward(model, batch_size);
        total_loss += loss;
    }
    float mean_loss = total_loss / num_batches;
    return mean_loss;
}

// -----------------------------------------------------------------------------
// sampling from the model

int sample_discrete(float* probabilities, int n, float coinf) {
    // sample from a discrete distribution
    // coin is a random number in [0, 1)
    // probabilities is an array of n probabilities that sum to 1
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coinf < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

void tokenize(const char *filename, int **tokens, int *token_count) {
    // reads filename and tokenizes all of its characters into a tokens[] int array
    FILE *file = fopenCheck(filename, "r");
    // as we are character-level, each character = 1 byte. count bytes in the file
    fseek(file, 0, SEEK_END);
    long count = ftell(file);
    fseek(file, 0, SEEK_SET);
    // now allocate the tokens array and tokenize
    *tokens = (int*)mallocCheck(count * sizeof(int));
    *token_count = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        // validate character c
        if (c != '\n' && !(c >= 'a' && c <= 'z')) {
            fprintf(stderr, "Error: Invalid character in training data: %c\n", c);
            exit(EXIT_FAILURE);
        }
        // tokenize character c
        (*tokens)[(*token_count)++] = c == '\n' ? 0 : c - 'a' + 1;
    }
    fcloseCheck(file);
}

// -----------------------------------------------------------------------------
// let's train!

int main() {
    RNG init_rng;
    rng_init(&init_rng, 1337);

    // read and tokenize the training, validation, and test data
    int *train_tokens, *val_tokens, *test_tokens;
    int train_token_count, val_token_count, test_token_count;
    tokenize("data/train.txt", &train_tokens, &train_token_count);
    tokenize("data/val.txt", &val_tokens, &val_token_count);
    tokenize("data/test.txt", &test_tokens, &test_token_count);
    // just being defensive, if you're trying different files, ok to remove the asserts here
    assert(train_token_count == 213796);
    assert(val_token_count == 7179);
    assert(test_token_count == 7170);

    // create the model
    MLP model;
    model.config.vocab_size = 27;
    model.config.context_length = 3; // if 3 tokens predict the 4th, it is a 4-gram model
    model.config.embedding_size = 48;
    model.config.hidden_size = 512;
    mlp_random_init(&model, &init_rng);

    // optimizer
    AdamW optimizer;
    float learning_rate = 7e-4;
    float weight_decay = 1e-4;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    adamw_init(&optimizer, model.num_parameters, learning_rate, beta1, beta2, weight_decay, eps);

    // training loop
    StepTimer timer;
    init_timer(&timer, 0.9);
    int batch_size = 128;
    int num_steps = 50000;
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
            printf("step %d/%d | train_loss %.6f | val_loss %.6f | lr %.6f | time/step %.4fms\n", step, num_steps, train_loss, val_loss, lr, get_dt(&timer) * 1000);
        }
        start_timer(&timer);
        // get the next batch of training data
        dataloader(&model, train_tokens, model.config.context_length, batch_size, train_token_count, &pos);
        // forward through the model
        float loss = forward(&model, batch_size);
        // zero out the gradients so we know we have the right gradient state just before the backward pass
        zero_grad(&model);
        // compute the gradients
        backward(&model);
        // update the weights
        update(&optimizer, &model);
        stop_timer(&timer);
    }

    // model inference
    // hardcode a prompt from which we'll continue the text
    RNG sample_rng;
    rng_init(&sample_rng, 42);
    char prompt[] = "\nrichard";
    int context_length = model.config.context_length;
    int context[context_length];
    int prompt_length = strlen(prompt);
    // take the last context_length tokens from the prompt
    for (int i = 0; i < context_length; i++) {
        context[i] = prompt[prompt_length - context_length + i] - 'a' + 1;
    }
    printf("%s", prompt);
    fflush(stdout);
    free(model.inputs);
    free(model.targets);  // free the targets so we don't compute the loss in fwd pass
    model.inputs = context;
    model.targets = NULL;
    // now let's sample 200 more tokens that follow
    for (int i = 0; i < 200; i++) {
        // take the last context_length tokens and predict the next one
        forward(&model, 1);
        float *probs = model.acts.probs;
        float coinf = rng_random(&sample_rng);
        int next_token = sample_discrete(probs, model.config.vocab_size, coinf);
        // shift the context to the left and append the next token
        for (int j = 0; j < context_length - 1; j++) {
            context[j] = context[j + 1];
        }
        context[context_length - 1] = next_token;
        printf("%c", next_token == 0 ? 10 : next_token + 'a' - 1);
        fflush(stdout);
    }

    // and finally report the test loss
    float test_loss = eval_split(&model, test_tokens, test_token_count, 0, batch_size);
    printf("test_loss %.4f\n", test_loss);

    // free all dynamically allocated memory
    mlp_free(&model);
    adam_free(&optimizer);

    return 0;
}
