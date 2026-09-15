"""
Siamese U-Net for pixel-level binary change detection.

Both the before (T1) and after (T2) images pass through a shared encoder.
At each encoder scale the absolute feature difference |feat_T1 - feat_T2| is
computed and passed as a skip connection to the decoder, so the decoder only
ever sees change-relevant information. A 1×1 conv head produces a single
change-logit per pixel.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two ConvBnReLU layers + MaxPool.

    Returns the pre-pool feature map (for skip connections) and the pooled
    map (input to the next stage).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnReLU(in_ch, out_ch)
        self.conv2 = ConvBnReLU(out_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv2(self.conv1(x))
        return self.pool(feat), feat


class DecoderBlock(nn.Module):
    """Bilinear upsample → concat skip → two ConvBnReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                          align_corners=False)
        return self.conv2(self.conv1(torch.cat([x, skip], dim=1)))


class Bottleneck(nn.Module):
    """Bridge between encoder and decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnReLU(in_ch, out_ch)
        self.conv2 = ConvBnReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for satellite image change detection.

    Parameters
    ----------
    in_channels      : input image channels (3 for RGB)
    encoder_channels : output channels per encoder stage, e.g. [64, 128, 256, 512]
    decoder_channels : output channels per decoder stage, e.g. [256, 128, 64, 32]
    dropout          : dropout probability before the classification head
    """

    def __init__(self,
                 in_channels: int = 3,
                 encoder_channels: List[int] = None,
                 decoder_channels: List[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        encoder_channels = encoder_channels or [64, 128, 256, 512]
        decoder_channels = decoder_channels or [256, 128, 64, 32]

        if len(encoder_channels) != len(decoder_channels):
            raise ValueError("encoder_channels and decoder_channels must have equal length")

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        enc_in = in_channels
        self.encoder_blocks = nn.ModuleList()
        for out_ch in encoder_channels:
            self.encoder_blocks.append(EncoderBlock(enc_in, out_ch))
            enc_in = out_ch

        self.bottleneck = Bottleneck(encoder_channels[-1], decoder_channels[0])

        self.decoder_blocks = nn.ModuleList()
        dec_in = decoder_channels[0]
        for i, out_ch in enumerate(decoder_channels[1:], start=1):
            skip_ch = encoder_channels[-(i + 1)]
            self.decoder_blocks.append(DecoderBlock(dec_in, skip_ch, out_ch))
            dec_in = out_ch

        self.dropout = nn.Dropout2d(p=dropout)
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

    def _encode(self, x: torch.Tensor):
        """Return (pooled_bottleneck_feat, [skip_feats_shallow_to_deep])."""
        skips = []
        for block in self.encoder_blocks:
            x, feat = block(x)
            skips.append(feat)
        return x, skips

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img_a : (B, 3, H, W) before image
        img_b : (B, 3, H, W) after image

        Returns
        -------
        (B, 1, H, W) change logits (pre-sigmoid)
        """
        bot_a, skips_a = self._encode(img_a)
        bot_b, skips_b = self._encode(img_b)

        x = self.bottleneck(torch.abs(bot_a - bot_b))

        diff_skips = [torch.abs(sa - sb) for sa, sb in zip(skips_a, skips_b)]
        for i, dec_block in enumerate(self.decoder_blocks):
            x = dec_block(x, diff_skips[-(i + 2)])

        x = F.interpolate(x, size=img_a.shape[2:], mode="bilinear",
                          align_corners=False)
        return self.final_conv(self.dropout(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> SiameseUNet:
    """Instantiate SiameseUNet from the project config dict."""
    m = cfg.get("model", {})
    return SiameseUNet(
        in_channels=3,
        encoder_channels=m.get("encoder_channels", [64, 128, 256, 512]),
        decoder_channels=m.get("decoder_channels", [256, 128, 64, 32]),
        dropout=m.get("dropout", 0.1),
    )
