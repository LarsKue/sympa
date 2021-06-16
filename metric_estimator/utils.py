
def symmetricize(x):
    # make a tensor symmetric by copying the
    # upper triangular part into the lower one
    return x.triu() + x.triu(1).transpose(-1, -2)
