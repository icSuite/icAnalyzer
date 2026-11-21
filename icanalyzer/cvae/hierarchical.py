"""Hierarchical mask-aware spline CVAE for the native 184x184 IMAGE grid."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import PartialConv2d, _spline_basis


#%% Shared spatial operations

def normalization_groups(channels):
    """Use the largest small GroupNorm partition that divides the channels."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    raise ValueError("unable to choose GroupNorm groups")


def resized_coordinates(coordinates, batch_size, shape):
    values = F.interpolate(
        coordinates, size=shape, mode="bilinear", align_corners=False
    )
    return values.expand(batch_size, -1, -1, -1)


class MaskedResidualDownsample(nn.Module):
    """Stride-two partial-convolution block with FiLM and fixed coordinates."""

    def __init__(self, input_channels, output_channels, condition_size):
        super().__init__()
        self.first = PartialConv2d(
            input_channels + 3, output_channels, 3,
            stride=2, padding=1, return_mask=True
        )
        self.first_norm = nn.GroupNorm(
            normalization_groups(output_channels), output_channels
        )
        self.second = PartialConv2d(
            output_channels, output_channels, 3,
            padding=1, return_mask=True
        )
        self.second_norm = nn.GroupNorm(
            normalization_groups(output_channels), output_channels
        )
        self.skip = PartialConv2d(
            input_channels, output_channels, 1,
            stride=2, bias=False
        )
        self.film = (
            nn.Linear(condition_size, 2 * output_channels)
            if condition_size else None
        )

    def forward(self, values, mask, coordinates, condition):
        coordinate_values = resized_coordinates(
            coordinates, len(values), values.shape[-2:]
        )
        main, updated_mask = self.first(
            torch.cat((values, coordinate_values), dim=1), mask
        )
        main = F.silu(self.first_norm(main))
        main, updated_mask = self.second(main, updated_mask)
        main = self.second_norm(main)
        if self.film is not None:
            scale, offset = self.film(condition).chunk(2, dim=1)
            main = (
                main * (1 + scale[:, :, None, None])
                + offset[:, :, None, None]
            )
        residual = self.skip(values, mask)
        output = F.silu(main + residual) * updated_mask
        return output, updated_mask


class ResidualUpsample(nn.Module):
    """Coordinate-aware residual block at one explicit decoder scale."""

    def __init__(self, input_channels, output_channels, condition_size):
        super().__init__()
        self.first = nn.Conv2d(input_channels + 3, output_channels, 3, padding=1)
        self.first_norm = nn.GroupNorm(
            normalization_groups(output_channels), output_channels
        )
        self.second = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.second_norm = nn.GroupNorm(
            normalization_groups(output_channels), output_channels
        )
        self.skip = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.film = (
            nn.Linear(condition_size, 2 * output_channels)
            if condition_size else None
        )

    def forward(self, values, shape, coordinates, condition):
        values = F.interpolate(
            values, size=shape, mode="bilinear", align_corners=False
        )
        coordinate_values = resized_coordinates(coordinates, len(values), shape)
        main = torch.cat((values, coordinate_values), dim=1)
        main = F.silu(self.first_norm(self.first(main)))
        main = self.second_norm(self.second(main))
        if self.film is not None:
            scale, offset = self.film(condition).chunk(2, dim=1)
            main = (
                main * (1 + scale[:, :, None, None])
                + offset[:, :, None, None]
            )
        return F.silu(main + self.skip(values))


#%% Hierarchical conditional VAE

class HierarchicalSplineConditionalVAE(nn.Module):
    """Five-scale encoder with global and spatial stochastic latent groups."""

    ENCODER_CHANNELS = (16, 24, 32, 48, 64)
    DECODER_CHANNELS = (64, 48, 32, 24)

    def __init__(
        self,
        image_shape=(184, 184),
        global_latent_dim=36,
        spatial_latent_channels=8,
        spline_control_points=92,
        spline_degree=3,
        quality_weight_channels=None,
        encoder_mask_policy="union",
        kp_conditioning=None,
        conditioning=None,
        field_names=None,
        magnetic_latitude=None,
    ):
        super().__init__()
        self.image_shape = tuple(int(value) for value in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 32:
            raise ValueError("hierarchical model requires a two-dimensional grid at least32x32")
        if field_names is None:
            raise ValueError("field_names must explicitly identify the modeled fields")
        self.field_names = tuple(str(name) for name in field_names)
        if (
            not self.field_names or any(not name for name in self.field_names)
            or len(set(self.field_names)) != len(self.field_names)
        ):
            raise ValueError("field_names must contain distinct nonempty names")
        if quality_weight_channels is None:
            quality_weight_channels = len(self.field_names)
        self.quality_weight_channels = int(quality_weight_channels)
        if self.quality_weight_channels not in (0, 1, len(self.field_names)):
            raise ValueError("quality channels must be zero, one or match fields")
        self.quality_weight_in_encoder = self.quality_weight_channels > 0
        self.encoder_mask_policy = str(encoder_mask_policy)
        if self.encoder_mask_policy not in ("intersection", "union"):
            raise ValueError("encoder mask policy must be intersection or union")

        self.global_latent_dim = int(global_latent_dim)
        self.spatial_latent_channels = int(spatial_latent_channels)
        if self.global_latent_dim < 1 or self.spatial_latent_channels < 0:
            raise ValueError("global latent size must be positive and spatial channels nonnegative")
        self.spline_control_points = int(spline_control_points)
        self.spline_degree = int(spline_degree)
        if self.spline_control_points <= self.spline_degree:
            raise ValueError("spline control-point count must exceed its degree")

        self.kp_conditioning = (
            None if kp_conditioning is None else dict(kp_conditioning)
        )
        self.conditioning = None if conditioning is None else dict(conditioning)
        if self.kp_conditioning is not None and self.conditioning is not None:
            raise ValueError("Kp and named conditioning are mutually exclusive")
        condition_size = self._initialize_conditioning()
        self.condition_size = condition_size

        stage_shapes = []
        current_shape = self.image_shape
        for _ in self.ENCODER_CHANNELS:
            current_shape = tuple((size + 1) // 2 for size in current_shape)
            stage_shapes.append(current_shape)
        self.encoder_stage_shapes = tuple(stage_shapes)
        self.bottleneck_shape = self.encoder_stage_shapes[-1]
        self.spatial_latent_shape = (
            self.spatial_latent_channels, *self.bottleneck_shape
        )
        self.spatial_latent_dim = int(np.prod(self.spatial_latent_shape))
        self.latent_dim = self.global_latent_dim + self.spatial_latent_dim
        self.latent_groups = {
            "global": (0, self.global_latent_dim),
        }
        if self.spatial_latent_channels:
            self.latent_groups["spatial"] = (
                self.global_latent_dim, self.latent_dim
            )

        coordinates = self._coordinate_maps(magnetic_latitude)
        self.register_buffer("coordinates", coordinates)

        encoder_input_channels = len(self.field_names) + self.quality_weight_channels
        blocks = []
        input_channels = encoder_input_channels
        for output_channels in self.ENCODER_CHANNELS:
            blocks.append(MaskedResidualDownsample(
                input_channels, output_channels, condition_size
            ))
            input_channels = output_channels
        self.encoder_blocks = nn.ModuleList(blocks)
        final_channels = self.ENCODER_CHANNELS[-1]
        global_input_size = final_channels + condition_size
        self.global_mu = nn.Linear(global_input_size, self.global_latent_dim)
        self.global_logvar = nn.Linear(global_input_size, self.global_latent_dim)
        if self.spatial_latent_channels:
            self.spatial_mu = nn.Conv2d(
                final_channels, self.spatial_latent_channels, 1
            )
            self.spatial_logvar = nn.Conv2d(
                final_channels, self.spatial_latent_channels, 1
            )
            self.spatial_projection = nn.Conv2d(
                self.spatial_latent_channels, final_channels, 1
            )

        decoder_input_size = self.global_latent_dim + condition_size
        self.global_decode = nn.Linear(
            decoder_input_size,
            final_channels * int(np.prod(self.bottleneck_shape))
        )
        decoder_blocks = []
        input_channels = final_channels
        for output_channels in self.DECODER_CHANNELS:
            decoder_blocks.append(ResidualUpsample(
                input_channels, output_channels, condition_size
            ))
            input_channels = output_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_shapes = tuple(reversed(self.encoder_stage_shapes[:-1]))
        self.sensor_heads = nn.ModuleList([
            nn.Conv2d(input_channels, 1, 3, padding=1)
            for _ in self.field_names
        ])

        self.register_buffer(
            "row_basis",
            _spline_basis(
                self.image_shape[0], self.spline_control_points,
                self.spline_degree
            )
        )
        self.register_buffer(
            "column_basis",
            _spline_basis(
                self.image_shape[1], self.spline_control_points,
                self.spline_degree
            )
        )

    def _initialize_conditioning(self):
        condition_size = 0
        if self.kp_conditioning is not None:
            required = ("center", "scale", "training_min", "training_max")
            values = np.asarray([self.kp_conditioning[name] for name in required])
            condition_size = int(self.kp_conditioning.get("embedding_dim", 8))
            if (
                not np.isfinite(values).all() or values[1] <= 0
                or condition_size < 1 or not 0 <= values[2] <= values[3] <= 9
            ):
                raise ValueError("invalid Kp conditioning configuration")
            self.kp_conditioning["embedding_dim"] = condition_size
            self.kp_embedding = nn.Sequential(
                nn.Linear(1, condition_size), nn.SiLU(),
                nn.Linear(condition_size, condition_size), nn.SiLU()
            )
        elif self.conditioning is not None:
            if self.conditioning.get("kind") != "named_standardization_v1":
                raise ValueError("unsupported named conditioning configuration")
            names = tuple(self.conditioning.get("feature_names", ()))
            arrays = {
                name: np.asarray(self.conditioning.get(name), dtype=float)
                for name in ("center", "scale", "training_min", "training_max")
            }
            if (
                not names or len(set(names)) != len(names)
                or any(values.shape != (len(names),) for values in arrays.values())
                or any(not np.isfinite(values).all() for values in arrays.values())
                or np.any(arrays["scale"] <= 0)
                or np.any(arrays["training_min"] > arrays["training_max"])
            ):
                raise ValueError("invalid named conditioning configuration")
            condition_size = int(self.conditioning.get("embedding_dim", 32))
            if condition_size < 1:
                raise ValueError("conditioning embedding size must be positive")
            self.conditioning["feature_names"] = list(names)
            self.conditioning["embedding_dim"] = condition_size
            self.condition_embedding = nn.Sequential(
                nn.Linear(len(names), condition_size), nn.SiLU(),
                nn.Linear(condition_size, condition_size), nn.SiLU()
            )
        return condition_size

    def _coordinate_maps(self, magnetic_latitude):
        rows = np.linspace(-1, 1, self.image_shape[0], dtype=np.float32)
        columns = np.linspace(-1, 1, self.image_shape[1], dtype=np.float32)
        row_map = np.broadcast_to(rows[:, None], self.image_shape)
        column_map = np.broadcast_to(columns[None], self.image_shape)
        if magnetic_latitude is None:
            latitude = row_map
        else:
            latitude = np.asarray(magnetic_latitude, dtype=np.float32)
            if latitude.shape != self.image_shape or not np.isfinite(latitude).all():
                raise ValueError("magnetic latitude must be finite on the image grid")
            center = float(np.mean(latitude))
            spread = float(np.max(np.abs(latitude - center)))
            latitude = (latitude - center) / max(spread, 1e-6)
        return torch.from_numpy(
            np.stack((row_map, column_map, latitude), axis=0).copy()
        )[None]

    def _condition_features(self, kp, conditioning, reference):
        if self.conditioning is not None:
            if kp is not None:
                raise ValueError("named-conditioned model does not accept scalar Kp")
            if conditioning is None:
                raise ValueError("named-conditioned model requires conditions")
            values = torch.as_tensor(
                conditioning, dtype=reference.dtype, device=reference.device
            )
            count = len(self.conditioning["feature_names"])
            if values.shape == (count,):
                values = values[None].expand(len(reference), -1)
            if values.shape != (len(reference), count) or not torch.isfinite(values).all():
                raise ValueError("named conditions require one finite row per image")
            center = torch.as_tensor(
                self.conditioning["center"], dtype=reference.dtype,
                device=reference.device
            )
            scale = torch.as_tensor(
                self.conditioning["scale"], dtype=reference.dtype,
                device=reference.device
            )
            return self.condition_embedding((values - center) / scale)
        if conditioning is not None:
            raise ValueError("model does not accept named conditions")
        if self.kp_conditioning is None:
            if kp is not None:
                raise ValueError("cannot supply Kp to an unconditional model")
            return None
        if kp is None:
            raise ValueError("Kp-conditioned model requires explicit Kp")
        values = torch.as_tensor(
            kp, dtype=reference.dtype, device=reference.device
        )
        if values.ndim == 0:
            values = values.expand(len(reference))
        if values.shape == (len(reference),):
            values = values[:, None]
        if (
            values.shape != (len(reference), 1)
            or not torch.isfinite(values).all()
            or torch.any((values < 0) | (values > 9))
        ):
            raise ValueError("Kp must be one finite 0-9 value per image")
        normalized = (
            values - self.kp_conditioning["center"]
        ) / self.kp_conditioning["scale"]
        return self.kp_embedding(normalized)

    def _common_mask(self, valid):
        valid = valid.to(dtype=torch.bool)
        if self.encoder_mask_policy == "union":
            return torch.any(valid, dim=1, keepdim=True)
        return torch.all(valid, dim=1, keepdim=True)

    def encode(self, values, valid, kp=None, conditioning=None):
        expected_channels = len(self.field_names) + self.quality_weight_channels
        if (
            values.ndim != 4 or values.shape[1] != expected_channels
            or values.shape[-2:] != self.image_shape
        ):
            raise ValueError("encoder input does not match model configuration")
        if (
            valid is None or valid.ndim != 4 or valid.shape[0] != len(values)
            or valid.shape[-2:] != self.image_shape
            or valid.shape[1] not in (1, len(self.field_names))
        ):
            raise ValueError("validity must contain one mask or one per field")
        condition = self._condition_features(kp, conditioning, values)
        mask = self._common_mask(valid).to(dtype=values.dtype)
        hidden = values
        for block in self.encoder_blocks:
            hidden, mask = block(
                hidden, mask, self.coordinates, condition
            )
        supported = hidden * mask
        pooled = supported.sum(dim=(2, 3)) / mask.sum(
            dim=(2, 3)
        ).clamp_min(1)
        global_features = (
            torch.cat((pooled, condition), dim=1)
            if condition is not None else pooled
        )
        global_mu = self.global_mu(global_features)
        global_logvar = self.global_logvar(global_features)
        if not self.spatial_latent_channels:
            return global_mu, global_logvar
        spatial_mu = self.spatial_mu(supported).flatten(1)
        spatial_logvar = self.spatial_logvar(supported).flatten(1)
        return (
            torch.cat((global_mu, spatial_mu), dim=1),
            torch.cat((global_logvar, spatial_logvar), dim=1)
        )

    @staticmethod
    def reparameterize(mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode_coefficients(self, latent, kp=None, conditioning=None):
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent array does not match hierarchical model")
        condition = self._condition_features(kp, conditioning, latent)
        global_latent = latent[:, :self.global_latent_dim]
        decoder_input = (
            torch.cat((global_latent, condition), dim=1)
            if condition is not None else global_latent
        )
        hidden = self.global_decode(decoder_input).reshape(
            len(latent), self.ENCODER_CHANNELS[-1], *self.bottleneck_shape
        )
        if self.spatial_latent_channels:
            spatial = latent[:, self.global_latent_dim:].reshape(
                len(latent), *self.spatial_latent_shape
            )
            hidden = hidden + self.spatial_projection(spatial)
        for block, shape in zip(self.decoder_blocks, self.decoder_shapes):
            hidden = block(hidden, shape, self.coordinates, condition)
        coefficient_features = F.interpolate(
            hidden,
            size=(self.spline_control_points, self.spline_control_points),
            mode="bilinear", align_corners=False
        )
        return torch.cat([
            head(coefficient_features) for head in self.sensor_heads
        ], dim=1)

    def decode(self, latent, kp=None, conditioning=None):
        coefficients = self.decode_coefficients(
            latent, kp=kp, conditioning=conditioning
        )
        return torch.einsum(
            "xi,bcij,yj->bcyx",
            self.column_basis, coefficients, self.row_basis
        )

    def forward(self, encoder_input, valid=None, kp=None, conditioning=None):
        mu, logvar = self.encode(
            encoder_input, valid, kp=kp, conditioning=conditioning
        )
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(
            latent, kp=kp, conditioning=conditioning
        )
        return reconstructed, mu, logvar

    def forward_samples(
        self, encoder_input, valid=None, kp=None, conditioning=None,
        latent_samples=1
    ):
        latent_samples = int(latent_samples)
        if latent_samples < 1:
            raise ValueError("latent_samples must be positive")
        if latent_samples == 1:
            return self.forward(
                encoder_input, valid, kp=kp, conditioning=conditioning
            )
        mu, logvar = self.encode(
            encoder_input, valid, kp=kp, conditioning=conditioning
        )
        noise = torch.randn(
            len(mu), latent_samples, self.latent_dim,
            dtype=mu.dtype, device=mu.device
        )
        latent = mu[:, None] + noise * torch.exp(0.5 * logvar)[:, None]
        repeated = {}
        if kp is not None:
            values = torch.as_tensor(kp, dtype=mu.dtype, device=mu.device)
            if values.ndim == 0:
                repeated["kp"] = values
            else:
                repeated["kp"] = values.repeat_interleave(latent_samples, dim=0)
        if conditioning is not None:
            values = torch.as_tensor(
                conditioning, dtype=mu.dtype, device=mu.device
            )
            if values.ndim == 1:
                values = values[None].expand(len(mu), -1)
            repeated["conditioning"] = values.repeat_interleave(
                latent_samples, dim=0
            )
        reconstructed = self.decode(
            latent.reshape(len(mu) * latent_samples, self.latent_dim),
            **repeated
        )
        return reconstructed.reshape(
            len(mu), latent_samples, *reconstructed.shape[1:]
        ), mu, logvar

    def configuration(self):
        configuration = {
            "model": "HierarchicalSplineConditionalVAE",
            "image_shape": list(self.image_shape),
            "latent_dim": self.latent_dim,
            "global_latent_dim": self.global_latent_dim,
            "spatial_latent_channels": self.spatial_latent_channels,
            "spatial_latent_shape": list(self.spatial_latent_shape),
            "spline_control_points": self.spline_control_points,
            "spline_degree": self.spline_degree,
            "field_names": list(self.field_names),
            "quality_weight_in_encoder": self.quality_weight_in_encoder,
            "quality_weight_channels": self.quality_weight_channels,
            "encoder_mask_policy": self.encoder_mask_policy,
            "encoder": "five_scale_masked_residual_groupnorm_silu_film",
            "encoder_stage_shapes": [list(shape) for shape in self.encoder_stage_shapes],
            "encoder_channels": list(self.ENCODER_CHANNELS),
            "decoder": "residual_upsampling_separate_spline_heads",
            "decoder_shapes": [list(shape) for shape in self.decoder_shapes],
            "decoder_channels": list(self.DECODER_CHANNELS),
            "coordinate_channels": [
                "normalized_row", "normalized_column", "normalized_mlat"
            ],
            "latent_groups": dict(self.latent_groups),
        }
        if self.kp_conditioning is not None:
            configuration["kp_conditioning"] = dict(self.kp_conditioning)
        if self.conditioning is not None:
            configuration["conditioning"] = dict(self.conditioning)
        return configuration

    @classmethod
    def from_configuration(cls, configuration):
        if configuration.get("model") != "HierarchicalSplineConditionalVAE":
            raise ValueError("configuration is not a hierarchical spline CVAE")
        model = cls(
            image_shape=tuple(configuration["image_shape"]),
            global_latent_dim=int(configuration["global_latent_dim"]),
            spatial_latent_channels=int(configuration["spatial_latent_channels"]),
            spline_control_points=int(configuration["spline_control_points"]),
            spline_degree=int(configuration["spline_degree"]),
            quality_weight_channels=int(configuration["quality_weight_channels"]),
            encoder_mask_policy=configuration["encoder_mask_policy"],
            kp_conditioning=configuration.get("kp_conditioning"),
            conditioning=configuration.get("conditioning"),
            field_names=configuration["field_names"],
            magnetic_latitude=None,
        )
        if (
            model.latent_dim != int(configuration["latent_dim"])
            or list(model.spatial_latent_shape)
            != list(configuration["spatial_latent_shape"])
        ):
            raise ValueError("hierarchical latent configuration is inconsistent")
        return model
