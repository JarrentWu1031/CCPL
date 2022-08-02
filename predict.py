import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import tempfile
from cog import BasePredictor, Path, Input

import net
from function import nor_mean_std, nor_mean


class Predictor(BasePredictor):
    def setup(self):

        testing_modes = ["photo-realistic", "art"]
        self.decoder_path = {
            testing_mode: f"pretrained/{testing_mode}/decoder_iter_160000.pth.tar"
            for testing_mode in testing_modes
        }
        SCT_path = {
            testing_mode: f"pretrained/{testing_mode}/sct_iter_160000.pth.tar"
            for testing_mode in testing_modes
        }
        vgg_path = "models/vgg_normalised.pth"

        self.device = "cuda:0"
        self.vgg = net.vgg

        network_art = net.Net(self.vgg, net.decoder, "art")
        network_photo = net.Net(self.vgg, net.decoder, "photo-realistic")

        self.SCT_art = network_art.SCT
        self.SCT_photo = network_photo.SCT

        self.SCT_art.eval()
        self.SCT_photo.eval()
        self.vgg.eval()

        self.vgg.load_state_dict(torch.load(vgg_path))
        self.SCT_art.load_state_dict(torch.load(SCT_path["art"]))
        self.SCT_photo.load_state_dict(torch.load(SCT_path["photo-realistic"]))

    def predict(
        self,
        content: Path = Input(
            description="Content image.",
        ),
        style: Path = Input(
            description="Sytle image.",
        ),
        mode: str = Input(
            default="photo-realistic",
            choices=[
                "photo-realistic",
                "art",
            ],
            description="Choose the style transfer mode.",
        ),
        content_size: int = Input(
            default=512,
            description="New (minimum) size for the content image, keeping the original size if set to 0.",
        ),
        style_size: int = Input(
            default=512,
            description="New (minimum) size for the style image, keeping the original size if set to 0.",
        ),
    ) -> Path:

        # do_interpolation = False
        # preserve_color = False

        vgg = (
            nn.Sequential(*list(self.vgg.children())[:31])
            if mode == "art"
            else nn.Sequential(*list(self.vgg.children())[:18])
        )
        SCT = self.SCT_art if mode == "art" else self.SCT_photo

        decoder = net.decoder
        decoder.eval()
        decoder.load_state_dict(torch.load(self.decoder_path[mode]))
        decoder = (
            decoder
            if mode == "art"
            else nn.Sequential(*list(net.decoder.children())[10:])
        )

        vgg.to(self.device)
        decoder.to(self.device)
        SCT.to(self.device)

        crop = False
        content_tf = test_transform(content_size, crop)
        style_tf = test_transform(style_size, crop)

        content = content_tf(Image.open(str(content)))
        style = style_tf(Image.open(str(style)))

        style = style.to(self.device).unsqueeze(0)
        content = content.to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = style_transfer(vgg, decoder, SCT, content, style)
        output = output.cpu()

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        save_image(output, str(out_path))
        save_image(output, "pps.png")
        return out_path


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(
    vgg, decoder, SCT, content, style, alpha=1.0, interpolation_weights=None
):
    assert 0.0 <= alpha <= 1.0
    content_f = vgg(content)
    style_f = vgg(style)

    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = SCT(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i : i + 1]
        content_f = content_f[0:1]
    else:
        feat = SCT(content_f, style_f)
    return decoder(feat)
