# mlp

This is a very exciting module because we tie a lot of things things together and train a Multi-layer Perceptron (MLP) to be an n-gram Language Model, following the paper [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) from Bengio et al. 2003.

We have multiple parallel implementations that all get to the exact same results but in very different ways:

- C version, which fully spells out all the individual operations.
- numpy version, which adds the Array abstraction. The operations are now grouped into functions that operate on arrays, but both forward and backward pass has to still be done manually.
- PyTorch version, which adds the Autograd engine. The PyTorch Tensor object looks just like an Array in numpy, but under the hood, PyTorch keeps track of the computational graph just like we saw with micrograd. The user specifies the forward pass, and then when they call `backward()` on the loss, PyTorch computes the gradients.

The major services offered by PyTorch then become clear:

- It gives you an efficient `Array` object just like that of numpy, except PyTorch calls it `Tensor` and some of the APIs are (spuriously) a bit different. Not covered in this module, PyTorch Tensors can be moved to different devices (like GPUs), greatly speeding up all the computations.
- It gives you an Autograd engine that records the computational graph of Tensors, and computes gradients for you.
- It gives you the `nn` library that packages up groups of Tensor operations into pre-built layers and loss functions that are common in deep learning.

As a result of our efforts, we will get to enjoy a much lower validation loss than we saw in the `ngram` module, and with significantly fewer parameters. However, we're also doing this at a much higher computational cost at training time (we're essentially compressing the dataset into the model parameters), and also to some extent at inference time.

TODOs:

- tune the hyperparameters so they are not terrible, I just winged it. (currently seeing val loss 2.06, recall count-based 4-gram was 2.11)
- merge the C version and make all three major versions agree (C, numpy, PyTorch)
- nice diagrams for this module

### License

MIT
