from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.transforms as transforms
from yiku.PATH import ASSETS
from PIL import Image
import torch

pre = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def check_amp(device="0"):
    if not torch.cuda.is_available():
        return False
    DEVICE = torch.device("cuda:0")
    m = mobilenet_v3_small(weights=None)
    m.load_state_dict(torch.load(ASSETS / 'mobilenet_v3_small.pth'))
    im_path = ASSETS / 'amp-test.jpg'  # image to check
    im = pre(Image.open(im_path))
    im = torch.unsqueeze(im, 0)
    im = im.cuda()
    m.eval()
    m.to(DEVICE)
    a = m(im)
    with torch.cuda.amp.autocast(True):
        b = m(im)
    del m
    del im
    torch.cuda.empty_cache()
    return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)
