import torch
from torch import nn, optim
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import transforms
from utils import gram_matrix

class StyleTransferModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vgg = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.replace_pooling()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def replace_pooling(self):
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.MaxPool2d):
                self.vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2)

    def run(self, content_img, style_img, num_steps=300, style_weight=1e6, content_weight=1):
        content = self.transform(content_img).unsqueeze(0).to(self.device)
        style = self.transform(style_img).unsqueeze(0).to(self.device)
        target = content.clone().requires_grad_(True).to(self.device)

        layers = {1: 's', 2: 's', 3: 's', 4: 'sc', 5: 's'}
        conv_indices = {}
        idx = 0
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.Conv2d):
                idx += 1
                if idx in layers:
                    conv_indices[idx] = i

        optimizer = optim.Adam([target], lr=0.03)
        for step in range(1, num_steps + 1):
            target.data.clamp_(0, 1)
            optimizer.zero_grad()

            style_loss = 0
            content_loss = 0

            for k, tag in layers.items():
                idx = conv_indices[k]
                curr_model = self.vgg[:idx + 1]

                target_feat = curr_model(target)
                if 'c' in tag:
                    content_feat = curr_model(content).detach()
                    content_loss += F.mse_loss(target_feat, content_feat)
                if 's' in tag:
                    style_feat = gram_matrix(curr_model(style)).detach()
                    target_gram = gram_matrix(target_feat)
                    style_loss += F.mse_loss(target_gram, style_feat)

            total_loss = style_loss * style_weight + content_loss * content_weight
            total_loss.backward()
            optimizer.step()

        return target.squeeze().detach().cpu()
