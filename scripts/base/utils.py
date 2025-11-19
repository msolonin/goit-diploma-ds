# -*- coding: utf-8 -*-

from torchvision import transforms
from scripts.base.constants import TRANSFORM_SIZE


transform = transforms.Compose([
    transforms.Resize(TRANSFORM_SIZE),
    transforms.ToTensor(),
])
