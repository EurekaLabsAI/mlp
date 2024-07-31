import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RNG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    layout="wide",
    page_title="n-gram MLP Playground",
    page_icon="🧠",
)

LEARNING_RATE = 7e-4
TOTAL_STEPS = 50000


# Vocabulary and Tokenization
@st.cache_resource
def load_data():
    train_text = open("./data/train.txt", "r").read()
    assert all(c == "\n" or ("a" <= c <= "z") for c in train_text)
    uchars = sorted(list(set(train_text)))
    vocab_size = len(uchars)
    char_to_token = {c: i for i, c in enumerate(uchars)}
    token_to_char = {i: c for i, c in enumerate(uchars)}
    EOT_TOKEN = char_to_token["\n"]
    test_tokens = [char_to_token[c] for c in open("./data/test.txt", "r").read()]
    val_tokens = [char_to_token[c] for c in open("./data/val.txt", "r").read()]
    train_tokens = [char_to_token[c] for c in open("./data/train.txt", "r").read()]
    return (
        vocab_size,
        char_to_token,
        token_to_char,
        EOT_TOKEN,
        test_tokens,
        val_tokens,
        train_tokens,
    )


(
    vocab_size,
    char_to_token,
    token_to_char,
    EOT_TOKEN,
    test_tokens,
    val_tokens,
    train_tokens,
) = load_data()


def dataloader(tokens, context_length, batch_size):
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        window = tokens[pos : pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        if len(inputs) == batch_size:
            yield (torch.tensor(inputs), torch.tensor(targets))
            inputs, targets = [], []
        pos += 1
        if pos + context_length >= n:
            pos = 0


class MLP(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )
        self.reinit(rng)

    @torch.no_grad()
    def reinit(self, rng):
        def reinit_tensor_randn(w, mu, sigma):
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)
        scale = (self.mlp[0].in_features) ** -0.5
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        scale = (self.mlp[2].in_features) ** -0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        emb = self.wte(idx)
        emb = emb.view(B, -1)
        logits = self.mlp(emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


@torch.inference_mode()
def eval_split(model, tokens, max_batches=None):
    # calculate the loss on the given tokens
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    mean_loss = total_loss / num_batches
    return mean_loss


def get_lr(step, total_steps, lr_max):
    return lr_max * 0.5 * (1 + math.cos(math.pi * step / total_steps))


def train_step(model, idx, targets, optimizer):
    lr = get_lr(st.session_state.step_count, TOTAL_STEPS, LEARNING_RATE)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model.train()
    logits, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return logits, loss.item(), lr


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


@torch.inference_mode()
def generate_text(model, prompt, max_new_tokens, rng):
    context = [char_to_token[c] for c in prompt]
    assert len(context) >= context_length
    context = context[-context_length:]  # crop to context_length

    for _ in range(max_new_tokens):
        # take the last context_length tokens and predict the next one
        model.eval()
        context_tensor = torch.tensor(context).unsqueeze(0).to(device)  # (1, T)
        logits, _ = model(context_tensor)  # (1, V)
        probs = softmax(logits[0])  # (V, )
        coinf = rng.random()  # "coin flip", float32 in range [0, 1)
        next_token = sample_discrete(probs, coinf)
        context = context[1:] + [next_token]  # update the token tape

        yield token_to_char[next_token] if next_token != EOT_TOKEN else "  \n"


def visualize_logits(input_tokens, target_tokens, model):
    model.eval()
    logits, _ = model(input_tokens)
    probs = softmax(logits)
    figs = []

    for i, prob in enumerate(probs):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[token_to_char[i] for i in range(len(prob))],
                y=prob.cpu().detach().numpy(),
                marker=dict(color=prob.cpu().detach().numpy(), colorscale="Viridis"),
            )
        )
        fig.update_layout(
            title=f"""Logits as probabilities of the next token | Step: {st.session_state.step_count}<br>Input: {''.join(token_to_char[t.item()] for t in input_tokens[i])} | Target: {token_to_char[target_tokens[i].item()]}
            """,
            xaxis_title="Token",
            yaxis_title="Probability",
        )
        figs.append(fig)
    return figs


def visualize_loss(loss_history):
    df = pd.DataFrame(
        loss_history,
        columns=["Step", "Train Loss", "Val Loss"],
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Train Loss"],
            mode="lines",
            name="Train Loss",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Step"],
            y=df["Val Loss"],
            mode="lines",
            name="Val Loss",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Training and Validation Loss", xaxis_title="Step", yaxis_title="Loss"
    )
    return fig


# Streamlit app
st.title("MLP - n-gram Playground")

# Parameters
context_length = st.sidebar.number_input(
    "Context Length", value=3, min_value=1, max_value=100
)
embedding_size = st.sidebar.number_input(
    "Embedding Size", value=48, min_value=1, max_value=1024
)
hidden_size = st.sidebar.number_input(
    "Hidden Size", value=512, min_value=1, max_value=1024
)
batch_size = st.sidebar.number_input(
    "Batch Size", value=128, min_value=1, max_value=128
)

st.sidebar.write(f"Using device: ```{device}```")


# Initialize or reset session state
def init_session_state():
    rng = RNG(1337)
    st.session_state.model = MLP(
        vocab_size, context_length, embedding_size, hidden_size, rng
    ).to(device)
    st.session_state.optimizer = torch.optim.AdamW(
        st.session_state.model.parameters(), lr=7e-4, weight_decay=1e-4
    )
    st.session_state.first_train_data_iter = dataloader(
        train_tokens, context_length, min(8, batch_size)
    )
    st.session_state.first_batch = next(st.session_state.first_train_data_iter)
    st.session_state.train_data_iter = dataloader(
        train_tokens, context_length, batch_size
    )
    st.session_state.loss_history = []
    st.session_state.step_count = 0
    st.session_state.context_length = context_length
    st.session_state.batch_size = batch_size
    st.session_state.embedding_size = embedding_size
    st.session_state.hidden_size = hidden_size


# Check if we need to reset the session state
if (
    "model" not in st.session_state
    or st.session_state.context_length != context_length
    or st.session_state.batch_size != batch_size
    or st.session_state.embedding_size != embedding_size
    or st.session_state.hidden_size != hidden_size
):
    init_session_state()

# Training buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Train 1 Step"):
        idx, targets = next(st.session_state.train_data_iter)
        idx, targets = idx.to(device), targets.to(device)
        _, loss, lr = train_step(
            st.session_state.model, idx, targets, st.session_state.optimizer
        )

        st.session_state.step_count += 1

with col2:
    if st.button("Train 2000 Steps"):
        with st.status(f"Training | Step: {st.session_state.step_count}") as status:
            for _ in range(2000):
                idx, targets = next(st.session_state.train_data_iter)
                idx, targets = idx.to(device), targets.to(device)
                _, loss, lr = train_step(
                    st.session_state.model, idx, targets, st.session_state.optimizer
                )

                if (
                    st.session_state.step_count % 200 == 0
                    or st.session_state.step_count == TOTAL_STEPS - 1
                ):
                    train_loss = eval_split(
                        st.session_state.model, train_tokens, max_batches=20
                    )
                    val_loss = eval_split(st.session_state.model, val_tokens)

                    st.session_state.loss_history.append(
                        (st.session_state.step_count, train_loss, val_loss)
                    )

                    st.write(
                        f"Step: {st.session_state.step_count} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.6e}"
                    )

                status.update(
                    label=f"Training | Step: {st.session_state.step_count}",
                    state="running",
                )
                st.session_state.step_count += 1

            status.update(state="complete")

with st.expander("Model Parameters"):
    st.write(st.session_state.model)

# Visualize first batch
with st.expander("Visualize First Batch", expanded=True):
    st.subheader("Visualization of First Batch (max 8 samples)")
    first_idx, first_targets = st.session_state.first_batch

    st.dataframe(
        {
            "Input": [
                "".join(token_to_char[t.item()] for t in tokens) for tokens in first_idx
            ],
            "Targets": [token_to_char[t.item()] for t in first_targets],
        },
        use_container_width=True,
    )

    figs = visualize_logits(
        first_idx.to(device), first_targets.to(device), st.session_state.model
    )

    tabs = st.tabs([f"Example {i}" for i in range(len(figs))])
    for i, fig in enumerate(figs):
        with tabs[i]:
            st.plotly_chart(fig, use_container_width=True)

with st.expander(
    f"Loss history | Total steps taken: {st.session_state.step_count} / {TOTAL_STEPS}"
):
    # Display training progress
    if st.session_state.loss_history:
        st.plotly_chart(
            visualize_loss(st.session_state.loss_history), use_container_width=True
        )


with st.expander(
    "Generate text",
    expanded=True,
):
    prompt = st.text_input("Prompt", "richard\n")
    max_new_tokens = st.number_input(
        "Max New Tokens", value=200, min_value=1, max_value=400
    )
    if st.button("Generate Text"):
        text_generator = generate_text(
            st.session_state.model, prompt, max_new_tokens, RNG(42)
        )
        st.write_stream(text_generator)
