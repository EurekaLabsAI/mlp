# bunch of less exciting, common utilities we'll use in multiple files
from math import log, cos, sin, pi

# -----------------------------------------------------------------------------
# random number generation

def box_muller_transform(u1, u2):
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    # This is using the Basic form of the Box-Muller transform
    # u1 and u2 are simple floats in [0, 1)
    # z1 and z2 are standard normal random variables
    z1 = (-2 * log(u1)) ** 0.5 * cos(2 * pi * u2)
    z2 = (-2 * log(u1)) ** 0.5 * sin(2 * pi * u2)
    return z1, z2

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 from Uniform(0, 1), i.e. interval [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def rand(self, n, a=0, b=1):
        # return n random float32 from Uniform(a, b), in a list
        return [self.random() * (b - a) + a for _ in range(n)]

    def randn(self, n, mu=0, sigma=1):
        # return n random float32 from Normal(0, 1), in a list
        # (note box-muller transform returns two numbers at a time)
        out = []
        for _ in range((n + 1) // 2):
            u1, u2 = self.random(), self.random()
            z1, z2 = box_muller_transform(u1, u2)
            out.extend([z1 * sigma + mu, z2 * sigma + mu])
        out = out[:n] # if n is odd crop list
        return out
