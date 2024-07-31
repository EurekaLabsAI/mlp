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


@torch.inference_mode()
def generate_text_step_by_step(model, context, rng):
    model.eval()
    context_tensor = torch.tensor(context).unsqueeze(0).to(device)
    logits, _ = model(context_tensor)
    probs = softmax(logits[0])
    coinf = rng.random()
    next_token = sample_discrete(probs, coinf)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[token_to_char[i] for i in range(len(probs))],
            y=probs.cpu().detach().numpy(),
            marker=dict(color=probs.cpu().detach().numpy(), colorscale="Viridis"),
        )
    )
    fig.update_layout(
        title=f"""Next token probability<br>Context: {''.join(token_to_char[t] for t in context)} | Predicted token: {token_to_char[next_token]}
        """,
        xaxis_title="Token",
        yaxis_title="Probability",
    )

    return probs, next_token, fig


def visualize_logits(input_tokens, target_tokens, model):
    model.eval()
    logits, _ = model(input_tokens)
    probs = [softmax(logits[i]) for i in range(len(logits))]

    coinf = RNG(42).random()

    figs = []

    for i, prob in enumerate(probs):
        next_token = sample_discrete(prob, coinf)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[token_to_char[i] for i in range(len(prob))],
                y=prob.cpu().detach().numpy(),
                marker=dict(color=prob.cpu().detach().numpy(), colorscale="Viridis"),
            )
        )
        fig.update_layout(
            title=f"""Next token probability | Step: {st.session_state.step_count}<br>Input: {''.join(token_to_char[t.item()] for t in input_tokens[i])} | Target: {token_to_char[target_tokens[i].item()]} | Predicted token: {token_to_char[next_token]}
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


st.title("n-gram Language Model with MLP")
st.markdown(
    "This app demonstrates a simple n-gram model using a Multi-Layer Perceptron (MLP) to predict the next token in a sequence of characters. It is based on the [mlp](https://github.com/EurekaLabsAI/mlp) repository, and the paper [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)."
)
st.markdown(
    "You can train the model for a specified number of steps, visualize the predicted probabilities for the next token as the model trains, and sample text using the trained model."
)

# Parameters
st.sidebar.header("Model parameters")
context_length = st.sidebar.number_input(
    "Context Length",
    value=3,
    min_value=1,
    max_value=100,
    help="Number of characters to consider as context for predicting the next character.",
)
embedding_size = st.sidebar.number_input(
    "Embedding Size",
    value=48,
    min_value=1,
    max_value=1024,
    step=48,
    help="Size of the embedding vector for each character.",
)
hidden_size = st.sidebar.number_input(
    "Hidden Size",
    value=512,
    min_value=1,
    max_value=1024,
    step=128,
    help="Number of neurons in the hidden layer of the MLP.",
)
batch_size = st.sidebar.number_input(
    "Batch Size",
    value=128,
    min_value=1,
    max_value=128,
    step=16,
    help="Number of samples processed before the model is updated.",
)

st.sidebar.header("Information")
st.sidebar.write(f"Using device: ```{device}```")
st.sidebar.write(f"Maximum steps: {TOTAL_STEPS}")
st.sidebar.write(f"Initial learning rate: {LEARNING_RATE}")
st.sidebar.write(f"Vocabulary size: {vocab_size}")


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
training_steps = st.number_input(
    "Number of training steps",
    value=1000,
    min_value=1,
    max_value=10000,
    step=100,
    help="Number of steps to train the model.",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Train Model", type="primary"):
        with st.status(f"Training | Step: {st.session_state.step_count}") as status:
            for _ in range(training_steps):
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

with col2:
    if st.button("Reset Model"):
        init_session_state()
        st.success("Model has been reset to initial state.")

with st.expander("Model Parameters"):
    st.write(st.session_state.model)

with st.expander("Visualize batch of data"):
    st.write("#### First 8 samples from the training data")
    st.write(
        "This table shows the first 8 samples from the training data, along with the target token."
    )
    first_idx, first_targets = st.session_state.first_batch

    st.dataframe(
        {
            "Input": [
                "".join(token_to_char[t.item()] for t in tokens) for tokens in first_idx
            ],
            "Target": [token_to_char[t.item()] for t in first_targets],
        },
        use_container_width=True,
    )

    st.write("### Next token probabilities")
    st.write(
        "The following plots show the predicted probabilities for the next token in each of the first 8 samples. Note how the predictions change as the model is trained."
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
    st.write(
        "The following plot shows the training and validation loss over time. The evaluation is done every 200 steps on both the training and validation data."
    )
    if st.session_state.loss_history:
        st.plotly_chart(
            visualize_loss(st.session_state.loss_history), use_container_width=True
        )

with st.expander("Sample from the model"):
    st.write("This section allows you to generate text using the trained model.")
    prompt = st.text_input("Prompt", "richard\n", key="shared_prompt")

    if prompt == "":
        prompt = "\n" * context_length

    sample_rng = RNG(42)

    generate, step_by_step = st.columns([0.3, 0.7])

    with generate:
        st.write("### Generate text")
        max_new_tokens = st.number_input(
            "\# of new tokens",
            value=200,
            min_value=1,
            max_value=400,
            help="Number of tokens to sample from the model.",
        )

        st.write(
            "This button generates the specified number of tokens at once, sampling from the model and updating the context at each step."
        )

        if st.button("Generate Text"):
            text_generator = generate_text(
                st.session_state.model, prompt, max_new_tokens, sample_rng
            )
            st.session_state.generated_text_output = "".join(list(text_generator))

        if "generated_text_output" in st.session_state:
            st.write(st.session_state.generated_text_output)

    with step_by_step:
        st.write("### Generate text step by step")
        tokens_to_generate = st.number_input(
            "\# of tokens to generate",
            value=10,
            min_value=1,
            max_value=100,
            help="Number of tokens to generate step by step.",
        )

        st.write(
            "This button generates the specified number of tokens step by step, showing the predicted probabilities for the next token at each step, along with the generated text so far."
        )

        if (
            "prev_prompt" not in st.session_state
            or st.session_state.prev_prompt != prompt
        ):
            st.session_state.context = [char_to_token[c] for c in prompt]
            st.session_state.generated_text = []
            st.session_state.prev_prompt = prompt
            st.session_state.text_at_step = []

        st.session_state.context = st.session_state.context[-context_length:]

        if st.button("Generate next tokens", type="primary"):
            st.session_state.context = [char_to_token[c] for c in prompt]
            st.session_state.context = st.session_state.context[-context_length:]
            st.session_state.generated_text = []
            st.session_state.text_at_step = []

            figs = []
            for _ in range(tokens_to_generate):
                probs, next_token, fig = generate_text_step_by_step(
                    st.session_state.model, st.session_state.context, sample_rng
                )

                st.session_state.context = st.session_state.context[1:] + [next_token]
                st.session_state.generated_text.append(token_to_char[next_token])

                st.session_state.text_at_step.append(
                    "".join(st.session_state.generated_text)
                )

                figs.append(fig)

            st.session_state.step_by_step_figs = figs

        if "step_by_step_figs" in st.session_state:
            tabs = st.tabs(
                [f"Token {i}" for i in range(len(st.session_state.step_by_step_figs))]
            )

            for i, fig in enumerate(st.session_state.step_by_step_figs):
                with tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(
                        f"Generated text so far: {st.session_state.text_at_step[i]}"
                    )
