import torch
import os

# midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

_midas_repo = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
_midas_source = "local" if os.path.exists(_midas_repo) else "github"
_midas_repo = _midas_repo if _midas_source == "local" else "intel-isl/MiDaS"

midas = torch.hub.load(_midas_repo, "DPT_Hybrid", source=_midas_source, trust_repo=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
for param in midas.parameters():
    param.requires_grad = False

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transforms = torch.hub.load(_midas_repo, "transforms", source=_midas_source, trust_repo=True)
transform = midas_transforms.dpt_transform
downsampling = 1

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

