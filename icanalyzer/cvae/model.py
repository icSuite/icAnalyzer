"""Field-agnostic mask-aware conditional VAE with a spline decoder."""

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


class SplineConditionalVAE(nn.Module):
    """Partial-convolution conditional VAE with a tensor-product spline decoder."""

    def __init__(
        self,
        image_shape=(46, 46),
        latent_dim=36,
        spline_control_points=30,
        spline_degree=3,
        quality_weight_in_encoder=True,
        decoder_feature_size=18,
        kp_conditioning=None,
        conditioning=None,
        field_names=None
    ):
        super().__init__()
        self.image_shape = tuple(int(value) for value in image_shape)
        self.latent_dim = int(latent_dim)
        self.spline_control_points = int(spline_control_points)
        self.spline_degree = int(spline_degree)
        self.quality_weight_in_encoder = bool(quality_weight_in_encoder)
        self.decoder_feature_size = int(decoder_feature_size)
        self.kp_conditioning = None if kp_conditioning is None else dict(kp_conditioning)
        self.conditioning = None if conditioning is None else dict(conditioning)
        if self.kp_conditioning is not None and self.conditioning is not None:
            raise ValueError("Kp and named conditioning configurations are mutually exclusive")
        if field_names is None:
            raise ValueError("field_names must explicitly identify the modeled fields")
        self.field_names = tuple(str(name) for name in field_names)
        if not self.field_names or any(not name for name in self.field_names) or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must contain distinct nonempty names")
        condition_size = 0
        if self.kp_conditioning is not None:
            condition_size = int(self.kp_conditioning.get("embedding_dim", 8))
            condition_values = [self.kp_conditioning[name] for name in ("center", "scale", "training_min", "training_max")]
            if not np.isfinite(condition_values).all() or self.kp_conditioning["scale"] <= 0 or condition_size < 1:
                raise ValueError("Kp normalization must be finite with positive scale and embedding size")
            if not 0 <= self.kp_conditioning["training_min"] <= self.kp_conditioning["training_max"] <= 9:
                raise ValueError("training Kp range must lie within 0–9")
            self.kp_conditioning["embedding_dim"] = condition_size
        elif self.conditioning is not None:
            if self.conditioning.get("kind") != "named_standardization_v1":
                raise ValueError("unsupported named conditioning configuration")
            names = tuple(self.conditioning.get("feature_names", ()))
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("named conditioning requires distinct feature names")
            arrays = {
                name: np.asarray(self.conditioning.get(name), dtype=float)
                for name in ("center", "scale", "training_min", "training_max")
            }
            if any(values.shape != (len(names),) for values in arrays.values()):
                raise ValueError("named conditioning statistics must match feature names")
            if any(not np.isfinite(values).all() for values in arrays.values()):
                raise ValueError("named conditioning statistics must be finite")
            if np.any(arrays["scale"] <= 0) or np.any(
                arrays["training_min"] > arrays["training_max"]
            ):
                raise ValueError("named conditioning scale and training ranges are invalid")
            condition_size = int(self.conditioning.get("embedding_dim", 32))
            if condition_size < 1:
                raise ValueError("named conditioning embedding size must be positive")
            self.conditioning["feature_names"] = list(names)
            self.conditioning["embedding_dim"] = condition_size
        if self.decoder_feature_size not in (18, 30):
            raise ValueError("decoder feature size must be 18 or 30")
        if self.spline_control_points <= self.spline_degree:
            raise ValueError("spline control-point count must exceed its degree")

        encoder_channels = len(self.field_names) + int(self.quality_weight_in_encoder)
        self.enc_pconv1 = PartialConv2d(
            encoder_channels, 16, kernel_size=3, stride=2, padding=1,
            return_mask=True
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
            dummy = torch.zeros(1, encoder_channels, *self.image_shape)
            mask = torch.ones(1, 1, *self.image_shape)
            encoded, _mask = self._encode_convolutions(dummy, mask)
        self.enc_bn1.train()
        self.enc_bn2.train()
        self.enc_bn3.train()
        self.encoded_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(encoded.numel())

        if condition_size:
            input_size = (
                1 if self.kp_conditioning is not None
                else len(self.conditioning["feature_names"])
            )
            embedding = nn.Sequential(
                nn.Linear(input_size, condition_size), nn.SiLU(),
                nn.Linear(condition_size, condition_size), nn.SiLU()
            )
            if self.kp_conditioning is not None:
                self.kp_embedding = embedding
            else:
                self.condition_embedding = embedding
        self.fc_mu = nn.Linear(self.flatten_dim + condition_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + condition_size, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim + condition_size, self.flatten_dim)

        self.decoder_seed_shape = (9, 9)
        self.up1 = nn.Upsample(size=self.decoder_seed_shape, mode="nearest")
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        # Preserve the legacy path unless the spatial-resolution test is requested.
        if self.decoder_feature_size == 18:
            self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            self.up2 = nn.Upsample(size=(self.decoder_feature_size, self.decoder_feature_size), mode="nearest")
        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.spline_resize = nn.AdaptiveAvgPool2d(
            (self.spline_control_points, self.spline_control_points)
        )
        self.spline_conv = nn.Conv2d(16, len(self.field_names), kernel_size=3, padding=1)

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

    def _kp_features(self, kp, reference):
        """Kp is a global image condition; it is never obscured by pixel masks."""
        if self.kp_conditioning is None:
            if kp is not None:
                raise ValueError("cannot supply Kp to an unconditional model")
            return None
        if kp is None:
            raise ValueError("Kp-conditioned model requires explicit Kp")
        kp = torch.as_tensor(kp, device=reference.device, dtype=reference.dtype)
        if kp.ndim == 0:
            kp = kp.expand(reference.shape[0])
        if kp.shape == (reference.shape[0],):
            kp = kp[:, None]
        if kp.shape != (reference.shape[0], 1) or not torch.isfinite(kp).all() or torch.any((kp < 0) | (kp > 9)):
            raise ValueError("Kp must be a finite 0–9 scalar or one value per image")
        normalized = (kp - self.kp_conditioning["center"]) / self.kp_conditioning["scale"]
        return self.kp_embedding(normalized)

    def _named_features(self, conditioning, reference):
        """Normalize and embed the complete named condition vector."""

        if self.conditioning is None:
            if conditioning is not None:
                raise ValueError("cannot supply named conditions to this model")
            return None
        if conditioning is None:
            raise ValueError("named-conditioned model requires explicit conditions")
        conditioning = torch.as_tensor(
            conditioning, device=reference.device, dtype=reference.dtype
        )
        feature_count = len(self.conditioning["feature_names"])
        if conditioning.shape == (feature_count,):
            conditioning = conditioning[None].expand(reference.shape[0], -1)
        if conditioning.shape != (reference.shape[0], feature_count):
            raise ValueError(
                "named conditions must contain one complete feature row per image"
            )
        if not torch.isfinite(conditioning).all():
            raise ValueError("named conditions must be finite")
        center = torch.as_tensor(
            self.conditioning["center"],
            dtype=reference.dtype, device=reference.device
        )
        scale = torch.as_tensor(
            self.conditioning["scale"],
            dtype=reference.dtype, device=reference.device
        )
        return self.condition_embedding((conditioning - center) / scale)

    def _condition_features(self, kp, conditioning, reference):
        if self.conditioning is not None:
            if kp is not None:
                raise ValueError("named-conditioned model does not accept scalar Kp")
            return self._named_features(conditioning, reference)
        if conditioning is not None:
            raise ValueError("model does not accept named conditions")
        return self._kp_features(kp, reference)

    def encode(self, values, valid, kp=None, conditioning=None):
        if valid is None:
            raise ValueError("partial-convolution encoder requires a validity mask")
        common_mask = torch.all(valid.to(dtype=torch.bool), dim=1, keepdim=True)
        hidden, _mask = self._encode_convolutions(
            values, common_mask.to(dtype=values.dtype)
        )
        hidden = hidden.reshape(hidden.shape[0], -1)
        condition = self._condition_features(kp, conditioning, hidden)
        if condition is not None:
            hidden = torch.cat((hidden, condition), dim=1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode_coefficients(self, latent, kp=None, conditioning=None):
        condition = self._condition_features(kp, conditioning, latent)
        if condition is not None:
            latent = torch.cat((latent, condition), dim=1)
        hidden = self.fc_decode(latent)
        hidden = hidden.reshape(latent.shape[0], *self.encoded_shape)
        hidden = self.up1(hidden)
        hidden = F.leaky_relu(self.dec_bn1(self.dec_conv1(hidden)))
        hidden = self.up2(hidden)
        hidden = F.leaky_relu(self.dec_bn2(self.dec_conv2(hidden)))
        hidden = self.spline_resize(hidden)
        return self.spline_conv(hidden)

    def decode(self, latent, kp=None, conditioning=None):
        condition_arguments = {}
        if kp is not None:
            condition_arguments["kp"] = kp
        if conditioning is not None:
            condition_arguments["conditioning"] = conditioning
        coefficients = self.decode_coefficients(latent, **condition_arguments)
        return torch.einsum(
            "xi,bcij,yj->bcyx",
            self.column_basis,
            coefficients,
            self.row_basis,
        )

    def forward(self, encoder_input, valid=None, kp=None, conditioning=None):
        condition_arguments = {}
        if kp is not None:
            condition_arguments["kp"] = kp
        if conditioning is not None:
            condition_arguments["conditioning"] = conditioning
        mu, logvar = self.encode(encoder_input, valid, **condition_arguments)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent, **condition_arguments), mu, logvar

    def forward_samples(
        self, encoder_input, valid=None, kp=None, conditioning=None,
        latent_samples=1
    ):
        """Encode once and decode several reparameterized latent draws in parallel."""

        latent_samples = int(latent_samples)
        if latent_samples < 1:
            raise ValueError("latent_samples must be positive")
        if latent_samples == 1:
            return self.forward(
                encoder_input, valid, kp=kp, conditioning=conditioning
            )

        condition_arguments = {}
        if kp is not None:
            condition_arguments["kp"] = kp
        if conditioning is not None:
            condition_arguments["conditioning"] = conditioning
        mu, logvar = self.encode(encoder_input, valid, **condition_arguments)
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn(
            mu.shape[0], latent_samples, mu.shape[1],
            dtype=mu.dtype, device=mu.device
        )
        latent = mu[:, None] + noise * standard_deviation[:, None]

        repeated_condition = {}
        if kp is not None:
            kp = torch.as_tensor(kp, dtype=mu.dtype, device=mu.device)
            if kp.ndim == 0:
                repeated_condition["kp"] = kp
            elif kp.shape[0] == mu.shape[0]:
                repeated_condition["kp"] = kp.repeat_interleave(
                    latent_samples, dim=0
                )
            else:
                raise ValueError("Kp must have one value per encoded image")
        if conditioning is not None:
            conditioning = torch.as_tensor(
                conditioning, dtype=mu.dtype, device=mu.device
            )
            if conditioning.ndim == 1:
                conditioning = conditioning[None].expand(mu.shape[0], -1)
            if conditioning.shape[0] != mu.shape[0]:
                raise ValueError("conditions must have one row per encoded image")
            repeated_condition["conditioning"] = conditioning.repeat_interleave(
                latent_samples, dim=0
            )
        reconstructed = self.decode(
            latent.reshape(mu.shape[0] * latent_samples, mu.shape[1]),
            **repeated_condition
        )
        reconstructed = reconstructed.reshape(
            mu.shape[0], latent_samples, *reconstructed.shape[1:]
        )
        return reconstructed, mu, logvar

    def configuration(self):
        configuration = {
            "model": "SplineConditionalVAE",
            "image_shape": list(self.image_shape),
            "latent_dim": self.latent_dim,
            "spline_control_points": self.spline_control_points,
            "spline_degree": self.spline_degree,
            "encoder": "partial_convolution",
            "encoder_channels": [self.enc_pconv1.in_channels, 16, 32, 64],
            "encoder_input_fields": (
                [*self.field_names, "auroral_interest_weight"]
                if self.quality_weight_in_encoder
                else list(self.field_names)
            ),
            "field_names": list(self.field_names),
            "quality_weight_in_encoder": self.quality_weight_in_encoder,
            "decoder_seed_shape": list(self.decoder_seed_shape),
            "decoder_channels": [64, 32, 16, len(self.field_names)],
        }
        # Old artifacts imply size 18; retain their exact default configuration.
        if self.decoder_feature_size != 18:
            configuration["decoder_feature_size"] = self.decoder_feature_size
        if self.kp_conditioning is not None:
            configuration["kp_conditioning"] = dict(self.kp_conditioning)
        if self.conditioning is not None:
            configuration["conditioning"] = dict(self.conditioning)
        return configuration

    @classmethod
    def from_configuration(cls, configuration):
        if configuration.get("model") not in ("SplineConditionalVAE", "SplineWICSI12VAE"):
            raise ValueError(f"unsupported model {configuration.get('model')!r}")
        field_names = configuration.get("field_names")
        if field_names is None and configuration.get("model") == "SplineWICSI12VAE":
            field_names = ("wic_corrected", "si12")
        return cls(
            image_shape=tuple(configuration["image_shape"]),
            latent_dim=int(configuration["latent_dim"]),
            spline_control_points=int(configuration["spline_control_points"]),
            spline_degree=int(configuration["spline_degree"]),
            quality_weight_in_encoder=configuration.get(
                "quality_weight_in_encoder", False
            ),
            decoder_feature_size=configuration.get("decoder_feature_size", 18),
            kp_conditioning=configuration.get("kp_conditioning"),
            conditioning=configuration.get("conditioning"),
            field_names=field_names
        )


