# Adapted from AA-CLIP and modified for PG-CLIP; see NOTICE.
import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import (
    KANAdapter,
    ResidualKANAdapter,
    SimpleAdapter,
    SimpleProj,
    initialize_linear_layers,
)

class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        image_adapter_type: str = "simple",
        image_proj_type: str = "simple",
        kan_bottleneck: int = 16,
        kan_adapter_bottleneck: int | None = None,
        kan_proj_bottleneck: int | None = None,
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        kan_seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels
        self.image_adapter_type = image_adapter_type
        self.image_proj_type = image_proj_type
        self.kan_bottleneck = kan_bottleneck
        self.kan_adapter_bottleneck = kan_adapter_bottleneck or kan_bottleneck
        self.kan_proj_bottleneck = kan_proj_bottleneck or kan_bottleneck
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_seed = kan_seed

        if image_adapter_type not in {"simple", "kan_residual"}:
            raise ValueError(f"unsupported image adapter type: {image_adapter_type}")
        if image_proj_type not in {"simple", "kan_residual"}:
            raise ValueError(f"unsupported image projection type: {image_proj_type}")

        def make_adapter(index):
            if image_adapter_type == "simple":
                return SimpleAdapter(1024, 1024)
            return ResidualKANAdapter(
                1024,
                1024,
                bottleneck=self.kan_adapter_bottleneck,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                seed=kan_seed + index,
                relu=image_adapter_type == "kan_residual",
            )

        def make_projection(index):
            if image_proj_type == "simple":
                return SimpleProj(1024, 768, relu)
            return ResidualKANAdapter(
                1024,
                768,
                bottleneck=self.kan_proj_bottleneck,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                seed=kan_seed + 100 + index,
                relu=relu,
            )

        layer_adapters = nn.ModuleList(
            [make_adapter(i) for i in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [make_projection(i) for i in range(len(levels))]
        )
        det_proj = make_projection(len(levels))
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        self._init_weights_()

    def _init_weights_(self):
        image_modules = [
            *self.image_adapter["layer_adapters"],
            *self.image_adapter["seg_proj"],
            self.image_adapter["det_proj"],
        ]
        for module in image_modules:
            if isinstance(module, ResidualKANAdapter):
                module.initialize_preserving_base_rng()
            elif isinstance(module, KANAdapter):
                module.initialize_preserving_replacement_rng()
            else:
                initialize_linear_layers(module)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]
        eot_idx = text.argmax(dim=-1)

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), eot_idx])
        # x = (
            # x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            # @ self.clipmodel.text_projection
        # )
        return x

    def text_trainable_parameters(self):
        return list(self.text_adapter.parameters())

    def image_config(self):
        return {
            "image_adapter_type": self.image_adapter_type,
            "image_proj_type": self.image_proj_type,
            "kan_bottleneck": self.kan_bottleneck,
            "kan_adapter_bottleneck": self.kan_adapter_bottleneck,
            "kan_proj_bottleneck": self.kan_proj_bottleneck,
            "kan_grid_size": self.kan_grid_size,
            "kan_spline_order": self.kan_spline_order,
            "kan_seed": self.kan_seed,
        }
