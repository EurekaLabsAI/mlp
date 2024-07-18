"""
Implements a simple n-gram language model in MLX.
Acts as the correctness reference for all the other versions.
"""
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from common import RNG

# -----------------------------------------------------------------------------
# The MLX Module

class MLP(nn.Module):
    """
    Takes the previous n tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, vocab_size, context_length, embedding_size, hidden_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size)  # token embedding table
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),
            nn.Tanh(),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size)
        )

    def __call__(self, idx, targets=None):
        # idx are the input tokens, (B, T) array of integers
        # targets are the target tokens, (B, ) array of integers
        B, T = idx.shape
        # encode all the tokens using the embedding table
        emb = self.wte(idx)  # (B, T, embedding_size)
        # concat all of the embeddings together
        emb = emb.reshape(B, -1)  # (B, T * embedding_size)
        # forward through the MLP
        logits = self.mlp(emb)
        # if we are given desired targets, also calculate the loss
        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(logits, targets)
        return logits, loss

# -----------------------------------------------------------------------------
# simple DataLoader that iterates over all the n-grams

def dataloader(tokens, context_length, batch_size):
    # returns inputs, targets as MLX arrays of shape (B, T), (B, )
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
            yield (mx.array(inputs), mx.array(targets))
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
        print(f"Eval, model is type: {type(model)}")
        logits, loss = model(inputs, targets)
        total_loss += loss
    # maybe loss should be scaled with the batch size?
    # probably not
    mean_loss = mx.mean(total_loss).item()
    return mean_loss

# -----------------------------------------------------------------------------
# let's train!

random = RNG(1337)
# TODO: actually use this rng for the model initialization

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
context_length = 3 # if 3 tokens predict the 4th, this is a 4-gram model
embedding_size = 24
hidden_size = 512
model = MLP(vocab_size, context_length, embedding_size, hidden_size)

# create the optimizer
learning_rate = 1e-3
# Adam optimizer in MLX doesn't have weight decay.
# Besides, AdamW should be better then Adam.
optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)

# training loop
batch_size = 64
num_steps = 50000
print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
train_data_iter = dataloader(train_tokens, context_length, batch_size)

# MLX doesn't allow passing functions to @mx.compile
# so we have to define the loss function outside
def loss_fn(model, inputs, targets):
    print(f"Loss, model is type: {type(model)}")
    logits, loss = model(inputs, targets)
    return loss

@mx.compile
def train_step(model, inputs, targets):
    print(f"Train, model is type: {type(model)}")
    loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
    return loss, grads

for step in range(num_steps):
    # cosine learning rate schedule, from max lr to 0
    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    optimizer.learning_rate = lr
    # every now and then evaluate the validation loss
    last_step = step == num_steps - 1
    if step % 200 == 0 or last_step:
        train_loss = eval_split(model, train_tokens, max_batches=20)
        val_loss = eval_split(model, val_tokens)
        print(f'step {step} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {lr:e}')
    # get the next batch of training data
    inputs, targets = next(train_data_iter)
    
    # perform a training step
    print(f"step in train loop: Model is type: {type(model)}")
    loss, grads = train_step(model, inputs, targets)
    
    # update model parameters
    print(f"Before optimizer update, model is type: {type(model)}")
    optimizer.update(model, grads)

    print(f"After optimizer update, model is type: {type(model)}")
    mx.eval(model.parameters(), optimizer.state)
    print(f"After eval, model is type: {type(model)}")