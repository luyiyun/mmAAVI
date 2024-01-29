from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

from .dataset import GMINIBATCH
from .encoders import AttVEncoder, AttGMMVEncoder, GraphEncoder
from .decoders import (
    MLPMultiModalDecoder,
    DotMultiModalDecoder,
    MixtureMultiModalDecoder,
    GraphDecoder,
)
from .discriminator import Discriminator

T = torch.Tensor
FRES = Dict[str, Any]  # forward results
LOSS = Dict[str, T]


class MMAAVINET(nn.Module):
    def __init__(
        self,
        dim_inputs: Dict[str, int],
        dim_outputs: Dict[str, int],
        nbatches: int,
        mixture_embeddings: bool = True,
        decoder_style: Literal["mlp", "glue", "mixture"] = "mixture",
        disc_alignment: bool = True,
        dim_z: int = 30,
        dim_u: int = 30,
        dim_c: int = 6,
        dim_enc_middle: int = 200,
        hiddens_enc_unshared: Sequence[int] = (256, 256),
        hiddens_enc_z: Sequence[int] = (256,),
        hiddens_enc_c: Sequence[int] = (50,),
        hiddens_enc_u: Sequence[int] = (50,),
        hiddens_dec: Optional[Sequence[int]] = (
            256,
            256,
        ),  # 可能是None在glue-style decoder时
        hiddens_prior: Sequence[int] = (),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "lrelu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = True,
        disc_condi_train: Optional[str] = None,
        disc_on_mean: bool = False,
        disc_criterion: Literal["ce", "bce", "focal"] = "ce",
        c_reparam: bool = True,
        c_prior: Optional[Sequence[float]] = None,
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = False,
        reduction: str = "sum",
        semi_supervised: bool = False,
        graph_encoder_init_zero: bool = True,
        # graph_decoder_whole: bool = False,
        temperature: float = 1.0,
        disc_gradient_weight: float = 20.,  # alpha
        label_smooth: float = 0.1,
        focal_alpha: float = 2.0,
        focal_gamma: float = 1.0,
        mix_dec_dot_weight: float = 0.9,
        loss_weight_kl_omics: float = 1.0,
        loss_weight_rec_omics: float = 1.0,
        loss_weight_kl_graph: float = 0.01,
        loss_weight_rec_graph: Tuple[float, str] = "nomics",
        loss_weight_disc: float = 1.0,
        loss_weight_sup: float = 1.0,
    ) -> None:
        assert decoder_style in ["mlp", "glue", "mixture"]
        if loss_weight_rec_graph != "nomics":
            assert isinstance(loss_weight_rec_graph, float)
        else:
            loss_weight_rec_graph = len(dim_outputs)
        super().__init__()
        self.decoder_style = decoder_style
        if disc_alignment:
            self.discriminator = Discriminator(
                inc=dim_z,
                outc=nbatches,
                hiddens=hiddens_disc,
                nclusters=dim_c,
                disc_on_mean=disc_on_mean,
                disc_condi_train=disc_condi_train,
                act=act,
                bn=disc_bn,
                dp=dp,
                criterion=disc_criterion,
                spectral_norm=spectral_norm,
                gradient_reversal=True,
                gradient_alpha=disc_gradient_weight,
                label_smooth=label_smooth,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
        else:
            self.discriminator = None
        if mixture_embeddings:
            self.encoder = AttGMMVEncoder(
                incs=dim_inputs,
                zdim=dim_z,
                cdim=dim_c,
                udim=dim_u,
                nbatch=nbatches if input_with_batch else 0,
                hiddens_unshare=hiddens_enc_unshared,
                nlatent_middle=dim_enc_middle,
                hiddens_z=hiddens_enc_z,
                hiddens_c=hiddens_enc_c,
                hiddens_u=hiddens_enc_u,
                hiddens_prior=hiddens_prior,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                omic_embed=omic_embed,
                omic_embed_train=omic_embed_train,
                c_reparam=c_reparam,
                semi_supervised=semi_supervised,
                c_prior=c_prior,
                temperature=temperature,
            )
        else:
            self.encoder = AttVEncoder(
                incs=dim_inputs,
                outc=dim_z,
                nbatch=nbatches if input_with_batch else 0,
                hiddens_unshare=hiddens_enc_unshared,
                nlatent_middle=dim_enc_middle,
                hiddens_shared=hiddens_enc_z,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                omic_embed=omic_embed,
                omic_embed_train=omic_embed_train,
            )

        if decoder_style == "mlp":
            self.decoder = MLPMultiModalDecoder(
                inc=dim_z,
                outcs=dim_outputs,
                hiddens=hiddens_dec,
                nbatch=nbatches,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )
        elif decoder_style == "glue":
            # NOTE: 这里会将hiddens_dec用于前面的非线性变换，所以正常来说，在使用
            # glue style decoder时，应该将hiddens设为None
            self.decoder = DotMultiModalDecoder(
                outcs=dim_outputs,
                nbatch=nbatches,
                inpt=dim_z,
                hiddens=hiddens_dec,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )
        elif decoder_style == "mixture":
            self.decoder = MixtureMultiModalDecoder(
                inc=dim_z,
                outcs=dim_outputs,
                hiddens=hiddens_dec,
                nbatch=nbatches,
                act=act,
                bn=bn,
                dp=dp,
                weight_dot=mix_dec_dot_weight,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )

        # glue和mixture都需要graph encoder和decoder的参与
        if decoder_style != "mlp":
            self.gencoder = GraphEncoder(
                sum(dim_outputs.values()),
                dim_z,
                reduction=reduction,
                zero_init=graph_encoder_init_zero,
            )
            self.gdecoder = GraphDecoder()

        self._loss_weight_kl_graph = loss_weight_kl_graph
        self._loss_weight_rec_graph = loss_weight_rec_graph
        self._loss_weight_kl_omics = loss_weight_kl_omics
        self._loss_weight_rec_omics = loss_weight_rec_omics
        self._loss_weight_disc = loss_weight_disc
        self._loss_weight_sup = loss_weight_sup

    def forward(self, batch: GMINIBATCH) -> FRES:
        enc_res = self.encoder(batch)
        if self.decoder_style != "mlp":
            genc_res = self.gencoder(batch)
            enc_res.update(genc_res)
        return enc_res

    def step(self, batch: GMINIBATCH) -> Tuple[FRES, FRES, FRES, LOSS]:
        all_loss = {}

        enc_res, enc_loss = self.encoder.step(batch)
        if "zsample" not in enc_res:
            enc_res["zsample"] = enc_res["z"].rsample()
        for k, v in enc_loss.items():
            all_loss[f"enc/{k}"] = v

        if self.decoder_style != "mlp":
            genc_res, genc_loss = self.gencoder.step(batch)
            enc_res.update(genc_res)
            for k, v in genc_loss.items():
                all_loss[f"genc/{k}"] = v

        dec_res, dec_loss = self.decoder.step(batch, enc_res)
        for k, v in dec_loss.items():
            all_loss[f"dec/{k}"] = v

        if self.discriminator is not None:
            disc_res, disc_loss = self.discriminator.step(
                batch, enc_res, dec_res
            )
            for k, v in disc_loss.items():
                all_loss[f"disc/{k}"] = v
        else:
            disc_res = None

        metric_loss = 0.
        for k, v in all_loss.items():
            if k.startswith("disc"):
                continue
            elif "kl_graph" in k:
                metric_loss += v * self._loss_weight_kl_graph
            elif "rec_graph" in k:
                metric_loss += v * self._loss_weight_rec_graph
            elif k.startswith("dec"):
                metric_loss += v * self._loss_weight_rec_omics
            elif k.startswith("enc") and ("sup" in k):
                metric_loss += v * self._loss_weight_sup
            elif k.startswith("enc") and ("kl" in k):
                metric_loss += v * self._loss_weight_kl_omics
            else:
                metric_loss += v
        all_loss["metric"] = metric_loss
        total_loss = metric_loss.clone()
        for k, v in all_loss.items():
            if k.startswith("disc"):
                total_loss += v * self._loss_weight_disc
        all_loss["total"] = total_loss
        return enc_res, dec_res, disc_res, all_loss
