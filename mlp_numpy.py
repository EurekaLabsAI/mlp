"""
Implements a simple n-gram language model in NumPy.
"""
import math
import numpy as np

# -----------------------------------------------------------------------------
# The NumPy Module

class MLP:
    """
    Takes the previous n tokens, encodes them with a lookup table,
    concatenates the vectors, and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, filename="mlp_weights.bin"):
        with open(filename, "rb") as file:
            header = np.frombuffer(file.read(256 * 4), dtype=np.int32)
            assert header[0] == 20240719
            self.vocab_size = header[1]
            self.context_length = header[2]
            self.embedding_size = header[3]
            self.hidden_size = header[4]

            def read_fun(shape, file):
                array = np.frombuffer(file.read(np.prod(shape) * 4), dtype=np.float32).reshape(shape)
                return array.copy()

            self.wte = read_fun((self.vocab_size, self.embedding_size), file)
            self.fc1_weights = read_fun((self.hidden_size, self.embedding_size * self.context_length), file)
            self.fc1_bias = read_fun((self.hidden_size,), file)
            self.fc2_weights = read_fun((self.vocab_size, self.hidden_size), file)
            self.fc2_bias = read_fun((self.vocab_size,), file)

        self.act_cache = {}  # cache for the activations for the backward pass

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def relu(self, x):
        return np.maximum(x, 0)

    def parameters(self):
        return {
            'wte': self.wte,
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias
        }

    def __call__(self, idx, targets):
        return self.forward(idx, targets)

    def forward(self, idx, targets):
        # idx are the input tokens, (B, T) numpy array of integers
        # targets are the target tokens, (B, ) tensor of integers
        B, T = idx.shape
        # encode all the tokens using the embedding table
        emb = self.wte[idx] # (B, T, embedding_size)
        # concat all of the embeddings together
        emb = emb.reshape(B, -1) # (B, T * embedding_size)
        # forward through the MLP
        h = self.relu(np.dot(emb, self.fc1_weights.T) + self.fc1_bias)
        logits = np.dot(h, self.fc2_weights.T) + self.fc2_bias

        self.act_cache['idx'] = idx
        self.act_cache['targets'] = targets
        self.act_cache['emb'] = emb
        self.act_cache['h'] = h

        loss = None
        if targets is not None:
            # cross-entropy loss
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            correct_logprobs = -np.log(probs[np.arange(len(targets)), targets])
            loss = np.sum(correct_logprobs) / len(targets)
            self.act_cache['probs'] = probs

        return logits, loss

    def backward(self):
        # extract the activations from the forward pass
        idx = self.act_cache['idx']
        targets = self.act_cache['targets']
        emb = self.act_cache['emb']
        h = self.act_cache['h']
        probs = self.act_cache['probs']
        B, T = idx.shape

        # compute the gradients
        dL_dlogits = probs
        dL_dlogits[np.arange(len(targets)), targets] -= 1
        dL_dlogits /= len(targets)

        dL_dfc2_weights = np.dot(dL_dlogits.T, h)
        dL_dfc2_bias = np.sum(dL_dlogits, axis=0)

        dL_dh = np.dot(dL_dlogits, self.fc2_weights)
        dL_dfc1 = dL_dh * (h > 0)

        dL_dfc1_weights = np.dot(dL_dfc1.T, emb)
        dL_dfc1_bias = np.sum(dL_dfc1, axis=0)

        dL_emb = np.dot(dL_dfc1, self.fc1_weights).reshape(B, T, self.embedding_size)
        dL_dwte = np.zeros_like(self.wte)
        for i in range(B):
            for j in range(T):
                dL_dwte[idx[i, j]] += dL_emb[i, j]

        return {
            'wte': dL_dwte,
            'fc1_weights': dL_dfc1_weights,
            'fc1_bias': dL_dfc1_bias,
            'fc2_weights': dL_dfc2_weights,
            'fc2_bias': dL_dfc2_bias
        }

class AdamW:

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-4, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.params = params

    def step(self, grads):
        self.t += 1
        for k in self.params.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                self.params[k] -= lr * self.weight_decay * self.params[k]

# -----------------------------------------------------------------------------
# simple DataLoader that iterates over all the n-grams

def dataloader(tokens, context_length, batch_size):
    # returns inputs, targets as torch Tensors of shape (B, T), (B, )
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        # simple sliding window over the tokens, of size context_length + 1
        window = tokens[pos:pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        # once we've collected a batch, emit it
        if len(inputs) == batch_size:
            yield (np.array(inputs), np.array(targets))
            inputs, targets = [], []
        # advance the position and wrap around if we reach the end
        pos += 1
        if pos + context_length >= n:
            pos = 0

# -----------------------------------------------------------------------------
# evaluation function

def eval_split(model, tokens, max_batches=None):
    # calculate the loss on the given tokens
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        total_loss += loss
    mean_loss = total_loss / num_batches
    return mean_loss

# -----------------------------------------------------------------------------
# let's train!

# "train" the Tokenizer, so we're able to map between characters and tokens
train_text = open('data/train.txt', 'r').read()
assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
uchars = sorted(list(set(train_text))) # unique characters we see in the input
vocab_size = len(uchars)
char_to_token = {c: i for i, c in enumerate(uchars)}
token_to_char = {i: c for i, c in enumerate(uchars)}
EOT_TOKEN = char_to_token['\n'] # designate \n as the delimiting <|endoftext|> token
# pre-tokenize all the splits one time up here
test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]

# create the model
model = MLP()
context_length = model.context_length
embedding_size = model.embedding_size
hidden_size = model.hidden_size

# optimizer
learning_rate = 1e-3
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# training loop
batch_size = 64
num_steps = 50000
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
train_data_iter = dataloader(train_tokens, context_length, batch_size)
for step in range(num_steps):
    # cosine learning rate schedule, from max lr to 0
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    # every now and then evaluate the validation loss
    last_step = step == num_steps - 1
    if step % 200 == 0 or last_step:
        train_loss = eval_split(model, train_tokens, max_batches=20)
        val_loss = eval_split(model, val_tokens)
        print(f'step {step}/{num_steps} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {lr:e}')
    # get the next batch of training data
    inputs, targets = next(train_data_iter)
    # forward through the model
    logits, loss = model(inputs, targets)
    # backpropagate and update the weights
    grads = model.backward()
    # step the optimizer
    optimizer.step(grads)

