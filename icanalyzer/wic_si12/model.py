"""Mask-aware convolutional VAE for corrected WIC and SI12 fields."""

#%% Imports

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn import functional as F


#%% Partial convolution

class PartialConv2d(nn.Conv2d):
    """Convolve valid pixels only and propagate their spatial support."""

    def __init__(self, *args, return_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_mask = bool(return_mask)
        kernel = torch.ones(1, 1, *self.kernel_size)
        self.register_buffer("mask_kernel", kernel)
        self.window_size = int(np.prod(self.kernel_size))

    def forward(self, values, mask=None):
        if mask is None:
            mask = torch.ones(
                values.shape[0], 1, *values.shape[-2:],
                dtype=values.dtype, device=values.device
            )
        if mask.shape[1] != 1:
            raise ValueError("partial-convolution mask must have one channel")
        mask = mask.to(dtype=values.dtype)

        with torch.no_grad():
            support = F.conv2d(
                mask,
                self.mask_kernel.to(dtype=values.dtype),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            updated_mask = torch.clamp(support, 0.0, 1.0)
            mask_ratio = self.window_size / (support + 1e-8)
            mask_ratio = mask_ratio * updated_mask

        raw = super().forward(values * mask)
        if self.bias is None:
            output = raw * mask_ratio
        else:
            bias = self.bias.reshape(1, self.out_channels, 1, 1)
            output = (raw - bias) * mask_ratio + bias
        if self.return_mask:
            return output, updated_mask
        return output


#%% Spline model

def _spline_basis(length, n_control_points, degree):
    n_knots = n_control_points + degree + 1
    knots = np.r_[
        np.zeros(degree),
        np.linspace(0.0, 1.0, n_knots - 2 * degree),
        np.ones(degree),
    ]
    positions = np.linspace(0.0, 1.0, int(length))
    basis = BSpline(knots, np.eye(n_control_points), degree)(positions)
    return torch.as_tensor(basis, dtype=torch.float32)


class SplineWICSI12VAE(nn.Module):
    """Legacy-style partial-convolution VAE with a tensor-product spline decoder."""

    def __init__(
        self,
        image_shape=(46, 46),
        latent_dim=36,
        spline_control_points=30,
        spline_degree=3,
    ):
        super().__init__()
        self.image_shape = tuple(int(value) for value in image_shape)
        self.latent_dim = int(latent_dim)
        self.spline_control_points = int(spline_control_points)
        self.spline_degree = int(spline_degree)
        if self.spline_control_points <= self.spline_degree:
            raise ValueError("spline control-point count must exceed its degree")

        self.enc_pconv1 = PartialConv2d(
            2, 16, kernel_size=3, stride=2, padding=1, return_mask=True
        )
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_pconv2 = PartialConv2d(
            16, 32, kernel_size=3, stride=2, padding=1, return_mask=True
        )
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_pconv3 = PartialConv2d(
            32, 64, kernel_size=3, stride=2, padding=1, return_mask=True
        )
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.enc_bn1.eval()
        self.enc_bn2.eval()
        self.enc_bn3.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 2, *self.image_shape)
            mask = torch.ones(1, 1, *self.image_shape)
            encoded, _mask = self._encode_convolutions(dummy, mask)
        self.enc_bn1.train()
        self.enc_bn2.train()
        self.enc_bn3.train()
        self.encoded_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(encoded.numel())

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_dim)

        self.decoder_seed_shape = (9, 9)
        self.up1 = nn.Upsample(size=self.decoder_seed_shape, mode="nearest")
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.spline_resize = nn.AdaptiveAvgPool2d(
            (self.spline_control_points, self.spline_control_points)
        )
        self.spline_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        self.register_buffer(
            "row_basis",
            _spline_basis(
                self.image_shape[0], self.spline_control_points, self.spline_degree
            ),
        )
        self.register_buffer(
            "column_basis",
            _spline_basis(
                self.image_shape[1], self.spline_control_points, self.spline_degree
            ),
        )

    def _encode_convolutions(self, values, mask):
        values, mask = self.enc_pconv1(values, mask)
        values = F.relu(self.enc_bn1(values))
        values, mask = self.enc_pconv2(values, mask)
        values = F.relu(self.enc_bn2(values))
        values, mask = self.enc_pconv3(values, mask)
        values = F.relu(self.enc_bn3(values))
        return values, mask

    def encode(self, values, valid):
        if valid is None:
            raise ValueError("partial-convolution encoder requires a validity mask")
        common_mask = torch.all(valid.to(dtype=torch.bool), dim=1, keepdim=True)
        hidden, _mask = self._encode_convolutions(
            values, common_mask.to(dtype=values.dtype)
        )
        hidden = hidden.reshape(hidden.shape[0], -1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode_coefficients(self, latent):
        hidden = self.fc_decode(latent)
        hidden = hidden.reshape(latent.shape[0], *self.encoded_shape)
        hidden = self.up1(hidden)
        hidden = F.leaky_relu(self.dec_bn1(self.dec_conv1(hidden)))
        hidden = self.up2(hidden)
        hidden = F.leaky_relu(self.dec_bn2(self.dec_conv2(hidden)))
        hidden = self.spline_resize(hidden)
        return self.spline_conv(hidden)

    def decode(self, latent):
        coefficients = self.decode_coefficients(latent)
        return torch.einsum(
            "yi,bcij,xj->bcyx",
            self.row_basis,
            coefficients,
            self.column_basis,
        )

    def forward(self, encoder_input, valid=None):
        mu, logvar = self.encode(encoder_input, valid)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    def configuration(self):
        return {
            "model": "SplineWICSI12VAE",
            "image_shape": list(self.image_shape),
            "latent_dim": self.latent_dim,
            "spline_control_points": self.spline_control_points,
            "spline_degree": self.spline_degree,
            "encoder": "partial_convolution",
            "encoder_channels": [2, 16, 32, 64],
            "encoder_input_fields": ["wic_corrected", "si12"],
            "quality_weight_in_encoder": False,
            "decoder_seed_shape": list(self.decoder_seed_shape),
            "decoder_channels": [64, 32, 16, 2],
        }

    @classmethod
    def from_configuration(cls, configuration):
        if configuration.get("model") != "SplineWICSI12VAE":
            raise ValueError(f"unsupported model {configuration.get('model')!r}")
        return cls(
            image_shape=tuple(configuration["image_shape"]),
            latent_dim=int(configuration["latent_dim"]),
            spline_control_points=int(configuration["spline_control_points"]),
            spline_degree=int(configuration["spline_degree"]),
        )


#%% Convolutional compatibility model

class WICSI12VAE(nn.Module):
    """Two-channel VAE whose encoder also receives two validity masks."""

    def __init__(self, image_shape=(46, 46), latent_dim=32, wic_floor_transformed=None):
        super().__init__()
        self.image_shape = tuple(int(value) for value in image_shape)
        self.latent_dim = int(latent_dim)
        self.wic_floor_transformed = (
            None if wic_floor_transformed is None else float(wic_floor_transformed)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 4, *self.image_shape)
            encoded = self.encoder(dummy)
        self.encoded_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(encoded.numel())

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    def encode(self, encoder_input):
        hidden = self.encoder(encoder_input)
        hidden = hidden.reshape(hidden.shape[0], -1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent):
        hidden = self.fc_decode(latent)
        hidden = hidden.reshape(latent.shape[0], *self.encoded_shape)
        reconstructed = self.decoder(hidden)
        if reconstructed.shape[-2:] != self.image_shape:
            reconstructed = F.interpolate(
                reconstructed,
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            )
        if self.wic_floor_transformed is not None:
            wic = self.wic_floor_transformed + F.softplus(reconstructed[:, :1])
            reconstructed = torch.cat((wic, reconstructed[:, 1:]), dim=1)
        return reconstructed

    def forward(self, encoder_input, valid=None):
        if encoder_input.shape[1] == 2:
            if valid is None:
                raise ValueError("convolutional compatibility model requires masks")
            encoder_input = torch.cat(
                (encoder_input, valid.to(dtype=encoder_input.dtype)), dim=1
            )
        mu, logvar = self.encode(encoder_input)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar

    def configuration(self):
        return {
            "model": "WICSI12VAE",
            "image_shape": list(self.image_shape),
            "latent_dim": self.latent_dim,
            "wic_floor_transformed": self.wic_floor_transformed,
            "encoder_channels": [4, 32, 64, 128],
            "decoder_channels": [128, 64, 32, 16, 2],
        }

    @classmethod
    def from_configuration(cls, configuration):
        if configuration.get("model") != "WICSI12VAE":
            raise ValueError(f"unsupported model {configuration.get('model')!r}")
        return cls(
            image_shape=tuple(configuration["image_shape"]),
            latent_dim=int(configuration["latent_dim"]),
            wic_floor_transformed=configuration.get("wic_floor_transformed"),
        )


#%% Loss

def vae_loss(
    reconstructed,
    target,
    valid,
    quality,
    transformed_variance,
    mu,
    logvar,
    beta=1.0,
    precision_scale=(1.0, 1.0),
    variance_floor=1e-3,
    maximum_relative_precision=100.0,
):
    """Return uncertainty-weighted reconstruction and KL losses.

    ``precision_scale`` is fitted once on the training split for each channel.
    Reduction uses valid support rather than the weight sum, preserving
    absolute uncertainty differences between images.
    """

    variance_floor = float(variance_floor)
    maximum_relative_precision = float(maximum_relative_precision)
    if not 0 < variance_floor < float("inf"):
        raise ValueError("variance_floor must be finite and positive")
    if not 0 < maximum_relative_precision < float("inf"):
        raise ValueError("maximum_relative_precision must be finite and positive")

    valid = valid.to(dtype=torch.bool)
    quality = torch.where(
        valid & torch.isfinite(quality),
        quality.to(dtype=reconstructed.dtype),
        torch.zeros_like(quality, dtype=reconstructed.dtype),
    )
    quality = torch.clamp(quality, min=0.0)
    variance = transformed_variance.to(dtype=reconstructed.dtype)
    finite_variance = torch.isfinite(variance) & (variance >= 0)
    support = valid & (quality > 0) & finite_variance
    variance = torch.where(
        finite_variance,
        variance,
        torch.full_like(variance, float(variance_floor)),
    )

    precision = 1.0 / torch.clamp(variance, min=variance_floor)
    precision_scale = torch.as_tensor(
        precision_scale,
        dtype=reconstructed.dtype,
        device=reconstructed.device,
    )
    if precision_scale.shape != (2,):
        raise ValueError("precision_scale must contain one value per channel")
    if not torch.all(torch.isfinite(precision_scale) & (precision_scale > 0)):
        raise ValueError("precision_scale values must be finite and positive")
    relative_precision = precision / precision_scale.reshape(1, 2, 1, 1)
    relative_precision = torch.clamp(
        relative_precision, max=maximum_relative_precision
    )

    support_float = support.to(reconstructed.dtype)
    weights = support_float * quality * relative_precision
    squared_error = (reconstructed - target)**2
    weighted_error = (weights * squared_error).sum(dim=(-2, -1))
    support_count = support_float.sum(dim=(-2, -1))
    active = support_count > 0
    if not torch.any(active):
        raise ValueError("batch contains no valid weighted pixels")
    channel_loss = weighted_error / torch.clamp(support_count, min=1.0)
    reconstruction_loss = channel_loss[active].mean()

    kl_per_sample = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
    )
    kl_loss = kl_per_sample.mean()
    total = reconstruction_loss + float(beta) * kl_loss
    return {
        "total": total,
        "reconstruction": reconstruction_loss,
        "kl": kl_loss,
    }
