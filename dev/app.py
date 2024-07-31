import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from common import RNG

# Vocabulary and Tokenization
# "train" the Tokenizer, so we're able to map between characters and tokens
train_text = open("./data/train.txt", "r").read()
assert all(c == "\n" or ("a" <= c <= "z") for c in train_text)
uchars = sorted(list(set(train_text)))  # unique characters we see in the input
vocab_size = len(uchars)
char_to_token = {c: i for i, c in enumerate(uchars)}
token_to_char = {i: c for i, c in enumerate(uchars)}
EOT_TOKEN = char_to_token["\n"]  # designate \n as the delimiting <|endoftext|> token
# pre-tokenize all the splits one time up here
test_tokens = [char_to_token[c] for c in open("./data/test.txt", "r").read()]
val_tokens = [char_to_token[c] for c in open("./data/val.txt", "r").read()]
train_tokens = [char_to_token[c] for c in open("./data/train.txt", "r").read()]


def dataloader(tokens, context_length, batch_size):
    # returns inputs, targets as torch Tensors of shape (B, T), (B, )
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        # simple sliding window over the tokens, of size context_length + 1
        window = tokens[pos : pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        # once we've collected a batch, emit it
        if len(inputs) == batch_size:
            yield (torch.tensor(inputs), torch.tensor(targets))
            inputs, targets = [], []
        # advance the position and wrap around if we reach the end
        pos += 1
        if pos + context_length >= n:
            pos = 0


# Define the MLP class
class MLP(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size)  # token embedding table
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )
        self.reinit(rng)

    @torch.no_grad()
    def reinit(self, rng):
        # This function is a bit of a hack and would not be present in
        # typical PyTorch code. Basically:
        # - we want to use our own RNG to initialize the weights.
        # - but we don't want to change idiomatic PyTorch code (above).
        # So here in this function we overwrite the weights using our own RNG.
        # This ensures that we have full control over the initialization and
        # can easily compare the results with other implementations.

        def reinit_tensor_randn(w, mu, sigma):
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        # Let's match the PyTorch default initialization:
        # Embedding with N(0,1)
        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)

        # Linear (both W,b) with U(-K, K) where K = 1/sqrt(fan_in)
        scale = (self.mlp[0].in_features) ** -0.5
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        scale = (self.mlp[2].in_features) ** -0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        emb = self.wte(idx)  # (B, T, embedding_size)
        emb = emb.view(B, -1)  # (B, T * embedding_size)
        logits = self.mlp(emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


def softmax(logits):
    # logits here is a (1D) torch.Tensor of shape (V,)
    maxval = torch.max(logits)  # subtract max for numerical stability
    exps = torch.exp(logits - maxval)
    probs = exps / torch.sum(exps)
    return probs


def sample_discrete(probs, coinf):
    # sample from a discrete distribution
    # probs is a torch.Tensor of shape (V,)
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1  # in case of rounding errors


# Function to perform a single training step
def train_step(model, train_data_iter, optimizer):
    model.train()

    idx, targets = next(train_data_iter)
    logits, loss = model(idx, targets)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return idx, targets, logits, loss.item()


def visualize_logits(input_tokens, target_tokens):
    model.eval()

    logits, _ = model(input_tokens)

    # Convert logits to probabilities
    probs = softmax(logits)

    figs = []

    for i, probs in enumerate(probs):
        # Visualization using Plotly
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[token_to_char[i] for i in range(len(probs))],
                y=probs.detach().numpy(),
                marker=dict(color=probs.detach().numpy(), colorscale="Viridis"),
            )
        )

        fig.update_layout(
            title=f"""Logits as probabilities of the next token<br>Input: {''.join(token_to_char[t.item()] for t in input_tokens[i])} | Target: {token_to_char[target_tokens[i].item()]}
            """,
            xaxis_title="Token",
            yaxis_title="Probability",
        )

        figs.append(fig)

    return figs


# Streamlit app
st.title("MLP Training Step Visualization")

# Parameters
#   vocab_size = st.sidebar.number_input("Vocab Size", value=100)
context_length = st.sidebar.number_input("Context Length", value=10)
embedding_size = st.sidebar.number_input("Embedding Size", value=32)
hidden_size = st.sidebar.number_input("Hidden Size", value=64)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.001)
batch_size = st.sidebar.number_input("Batch Size", value=1)

# Random number generator
rng = RNG(1337)

# Initialize model and optimizer
model = MLP(vocab_size, context_length, embedding_size, hidden_size, rng)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# First batch of data to visualize through the training steps
train_data_iter = dataloader(train_tokens, context_length, batch_size)
idx_prime, targets_prime = next(train_data_iter)

# Actual training data iterator
train_data_iter = dataloader(train_tokens, context_length, batch_size)

if st.button("Train"):
    idx, targets, logits, loss = train_step(model, train_data_iter, optimizer)

if st.button("Train 1000 steps"):
    for _ in range(1000):
        idx, targets, logits, loss = train_step(model, train_data_iter, optimizer)

st.dataframe(
    {
        "Input": [
            "".join(token_to_char[t.item()] for t in tokens) for tokens in idx_prime
        ],
        "Targets": [token_to_char[t.item()] for t in targets_prime],
    },
    use_container_width=True,
)

figs = visualize_logits(idx_prime, targets_prime)


tabs = st.tabs([f"Example {i}" for i in range(len(figs))])
for i, fig in enumerate(figs):
    with tabs[i]:
        st.plotly_chart(fig, use_container_width=True)

st.write("Loss:", loss)
