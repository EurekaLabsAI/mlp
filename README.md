# mlp

This is a very exciting module because we tie a lot of things things together and train a Multi-layer Perceptron (MLP) to be an n-gram Language Model, following the paper [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) from Bengio et al. 2003.

We have multiple parallel implementations that all get to the exact same results but in very different ways:

- micrograd, following the previous module `micrograd`. This is a highly inefficient approach but it uses our own scalar-valued gradient engine.
- numpy, where we use the array object of numpy but implement our own forward and backward pass using numpy operations.
- C, which is the same as the numpy code but it fully spells out all the individual operations in C code.
- PyTorch, where we use the pytorch library to implement the forward pass only. Just like micrograd, PyTorch will handle the backward pass for us.
- mlx/JAX? (would be nice to look into)

In this module, two critical abstractions get explored and are tied together in depth:

1. The idea of an Array (in numpy parlance) or Tensor (in PyTorch parlance): a multi-dimensional array that stores data and has operations defined on it.
2. The idea of a Module: a class that has both a `forward()` and a `backward()` method. The forward pass computes the output given the input, and the backward pass computes the gradient of the loss with respect to the input. The "autograd engine" keeps track of the computational pass that is constructed in the forward pass, and then after the forward pass iterates in the reverse order and calls `backward()` on each module, implementing backpropagation.

The services offered by PyTorch then become clear: it gives both an efficient Array/Tensor object, and it has an Autograd engine (just like micrograd) that computes gradients for you. Only burshed on in this module is a third major offering of PyTorch, the fact that PyTorch Tensors can be moved to different devices (like GPUs) in a transparent way, greatly speeding up all the computations.

As a result of our efforts, we will get to enjoy a much lower validation loss than we saw in the `ngram` module, and with significantly fewer parameters. However, we're also doing this at a much higher computational cost at training time (we're essentially compressing the dataset into the model parameters), and also to some extent at inference time.

TODOs:

- tune the hyperparameters so they are not terrible, I just winged it. (currently seeing val loss 2.06, recall count-based 4-gram was 2.11)
- implement all the other versions that match pytorch reference

### License

MIT
