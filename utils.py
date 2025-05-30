import torch

def gram_matrix(ip):
    b, c, h, w = ip.size()
    features = ip.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)
