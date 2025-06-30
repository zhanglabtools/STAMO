
import copy
import os
from itertools import chain
from math import ceil
from typing import Any, List, Mapping, Optional, Tuple, Union, NoReturn

import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
import scipy.sparse as sp

from ..graph import check_graph
from ..num import normalize_edges
from ..utils import AUTO, config, get_chained_attr, logged, freeze_running_stats, autodevice
from . import sc
from .base import Model
from .data import AnnDataset, ArrayDataset, DataLoader, GraphDataset, ParallelDataLoader

import itertools
from .base import Trainer, TrainingPlugin
from .plugins import EarlyStopping, LRScheduler, Tensorboard


#---------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)


#----------------------------- Network definition ------------------------------

class STAMO(torch.nn.Module):

    r"""
    STAMO network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.DataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator, prior: sc.Prior, atten_cross: sc.AttentionLayer,
            u2c: Optional[sc.Classifier] = None
    ) -> None:
        ## super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)
        super().__init__()
        if not set(x2u.keys()) == set(u2x.keys()) == set(idx.keys()) != set():
            raise ValueError(
                "`x2u`, `u2x`, `idx` should share the same keys "
                "and non-empty!"
            )
        self.keys = list(idx.keys())  # Keeps a specific order

        self.g2v = g2v
        self.v2g = v2g
        self.x2u = torch.nn.ModuleDict(x2u)
        self.u2x = torch.nn.ModuleDict(u2x)  ## NBDataDecoder
        for k, v in idx.items():  # Since there is no BufferList
            self.register_buffer(f"{k}_idx", v)
        self.du = du
        self.prior = prior

        self.device = autodevice()

        self.u2c = u2c.to(self.device) if u2c else None
        self.atten_cross = atten_cross.to(self.device) if atten_cross else None

    @property
    def device(self) -> torch.device:
        r"""
        Device of the module
        """
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = device
        self.to(self._device)

    def forward(self) -> NoReturn:
        r"""
        Invalidated forward operation
        """
        raise RuntimeError("STAMO does not support forward operation!")

#----------------------------- Trainer definition ------------------------------
DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to STAMOTrainer.compute_losses


@logged
class STAMOTrainer(Trainer):

    r"""
    Trainer for :class:`STAMO`

    Parameters
    ----------
    net
        :class:`STAMO` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration

    def __init__(
            self, net: STAMO, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, lam_joint_cross: float = None,
            lam_real_cross: float = None, lam_cos: float = None, lam_gate_rec: float=None,
            normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        # super().__init__(
        #     net, lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph,
        #     lam_align=lam_align, modality_weight=modality_weight,
        #     optim=optim, lr=lr, **kwargs
        # )

        required_kwargs = (
            "lam_data", "lam_kl", "lam_graph", "lam_align",
            "modality_weight", "optim", "lr"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        super().__init__(net)
        self.required_losses = ["g_nll", "g_kl", "g_elbo"]
        for k in self.net.keys:
            self.required_losses += [f"x_{k}_nll", f"x_{k}_kl", f"x_{k}_elbo"]
        self.required_losses += ["dsc_loss", "vae_loss", "gen_loss"]
        self.earlystop_loss = "vae_loss"

        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_align = lam_align
        if min(modality_weight.values()) < 0:
            raise ValueError("Modality weight must be non-negative!")
        normalizer = sum(modality_weight.values()) / len(modality_weight)
        self.modality_weight = {k: v / normalizer for k, v in modality_weight.items()}

        self.lr = lr
        self.vae_optim = getattr(torch.optim, optim)(
            itertools.chain(
                self.net.g2v.parameters(),
                self.net.v2g.parameters(),
                self.net.x2u.parameters(),
                self.net.u2x.parameters()
            ), lr=self.lr, **kwargs
        )
        self.dsc_optim = getattr(torch.optim, optim)(
            self.net.du.parameters(), lr=self.lr, **kwargs
        )

        self.align_burnin: Optional[int] = None
        self.eidx: Optional[torch.Tensor] = None  # Full graph used by the graph encoder
        self.enorm: Optional[torch.Tensor] = None  # Full graph used by the graph encoder
        self.esgn: Optional[torch.Tensor] = None  # Full graph used by the graph encoder


        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        self.freeze_u = False
        if net.u2c:
            self.required_losses.append("sup_loss")
            self.vae_optim = getattr(torch.optim, optim)(
                chain(
                    self.net.g2v.parameters(),
                    self.net.v2g.parameters(),
                    self.net.x2u.parameters(),
                    self.net.u2x.parameters(),
                    self.net.u2c.parameters()
                ), lr=self.lr, **kwargs
            )

        required_kwargs = ("lam_joint_cross", "lam_real_cross", "lam_cos")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_joint_cross = lam_joint_cross
        self.lam_real_cross = lam_real_cross
        self.lam_cos = lam_cos
        self.lam_gate_rec = lam_gate_rec
        self.required_losses += ["cos_loss", "gate_rec_loss"]  ## "joint_cross_loss"

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt,  adj, adj_raw, pmsk, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], data[5*K:6*K], data[6*K:7*K], \
            data[7*K], data[7*K+1:]  ## data[6*K] is from shuffle_pmsk in data.py -> class AnnDataset -> function __getitem__()


        self.net.batch_size = data[7*K].shape[0]
        batch_size = data[7*K].shape[0]

        adj = { ## spatial neighbor graph with neighbors
            k: adj[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        adj_raw = { ## raw spatial neighbor graph without neighbors, only self-loops
            k: adj_raw[i] #.to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }

        xbch = {
            k: xbch[i][:batch_size].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i][:batch_size].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i][:batch_size].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }

        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k][:batch_size].shape[0])
            for i, k in enumerate(keys)
        }

        pmsk = pmsk.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)

        return x, xrep, xbch, xlbl, xdwt, xflag, adj, adj_raw, pmsk, eidx, ewt, esgn

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, adj, adj_raw, pmsk, eidx, ewt, esgn = data  ##global_idx
        batch_size = self.net.batch_size

        u, usamp, l, edge_loss = {}, {}, {}, {}
        for k in net.keys:
            u[k], usamp[k], l[k], edge_loss[k] = net.x2u[k](x[k], xrep[k], adj[k], batch_size)

        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean[:batch_size] for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}

        gate_rec_loss = torch.tensor(0.0, device=self.net.device)
        for k in net.keys:
            gate_rec_loss += edge_loss[k]
        # for k in net.keys:
        #     mse_rec_loss += F.mse_loss(xrep[k][:batch_size], rec[k][:batch_size])

        # # x_rec_loss = torch.tensor(0.0, device=self.net.device)
        # x_rec_rna = usamp[net.keys[0]] @ vsamp[getattr(net, f"{net.keys[0]}_idx")].t()
        # x_rec_atac = usamp[net.keys[1]] @ vsamp[getattr(net, f"{net.keys[1]}_idx")].t()
        # x_rec_loss = F.mse_loss(x[net.keys[0]][:batch_size], x_rec_rna[:batch_size]) + \
        #              F.mse_loss(x[net.keys[1]][:batch_size], x_rec_atac[:batch_size])
        # gate_rec_loss = gate_rec_loss + x_rec_loss


        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample() ## v is D.Normal(loc, std)
        ## v contains all graph nodes, self.eidx contains all edges, and eidx only contains edges in the minibatch graph.

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        ## u2x NBDataDecoder: dot production data decoder modeling statistical characteristics
        ## of negative binomial (NB) distribution of X
        x_nll = {
            k: -net.u2x[k](
                usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
            ).log_prob(x[k][:batch_size]).mean()
            for k in net.keys   ##getattr(net, f"{k}_idx") return index of hvg features
        }

        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x[k][:batch_size].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)


        ## https://github.com/gao-lab/GLUE/issues/87  The pairing loss for pmsk
        pmsk = pmsk.T
        usamp_stack = torch.stack([usamp[k] for k in net.keys])
        pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        # if pmsk is ture for all modalities, then average; otherwise, select the modality of ture

        ## can be replaced by the between-modality attention aggregation layer (class AttentionLayer) in sc.py
        # m = pmsk[0] & pmsk[1]
        # if m.sum():
        #     u_common = usamp_stack * pmsk_stack
        #     usamp_mean, alpha_omics_1_2 = net.atten_cross(u_common[0], u_common[1])

        if self.normalize_u:  # default False
            usamp_mean = F.normalize(usamp_mean, dim=1)

        ## usamp_mean must be able to reconstruct data from two modalities simultaneously
        ## If there are no matched spots/anchors, then the spots filtered in usamp_mean[m] reconstruct themselves!
        if self.lam_cos:
            cos_loss = torch.as_tensor(0.0, device=net.device)
            for i, m in enumerate(pmsk):
                if m.sum(): #If there are no matched spots/anchors, cos_loss is 0
                    cos_loss += 1 - F.cosine_similarity(
                        usamp_stack[i, m], usamp_mean[m]
                    ).mean()
        else:
            cos_loss = torch.as_tensor(0.0, device=net.device)

        # joint_cross_loss = torch.as_tensor(0.0, device=net.device)
        # real_cross_loss = torch.as_tensor(0.0, device=net.device)
        # + self.lam_joint_cross * joint_cross_loss \
        # + self.lam_real_cross * real_cross_loss \
        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss \
            + self.lam_cos * cos_loss + self.lam_gate_rec*gate_rec_loss

        ## The discriminator and encoder are trained in an adversarial manner
        gen_loss = vae_loss - self.lam_align * dsc_loss

        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo, 'gate_rec_loss':gate_rec_loss,
            # "joint_cross_loss": joint_cross_loss,
            # "real_cross_loss": real_cross_loss,
            "cos_loss": cos_loss,
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch
        # print('epoch:',epoch)
        if self.freeze_u:
            self.net.x2u.apply(freeze_running_stats)
            self.net.du.apply(freeze_running_stats)
        else:  # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )

    @torch.no_grad()
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.eval()
        data = self.format_data(data)
        return self.compute_losses(data, engine.state.epoch)

    def fit(  # pylint: disable=arguments-renamed
            self, data: ArrayDataset,
            graph: GraphDataset, val_split: float = None,
            data_batch_size: int = None, graph_batch_size: int = None,
            align_burnin: int = None, safe_burnin: bool = True,
            max_epochs: int = None, patience: Optional[int] = None,
            reduce_lr_patience: Optional[int] = None,
            wait_n_lrs: Optional[int] = None,
            random_seed: int = None, directory: Optional[os.PathLike] = None,
            plugins: Optional[List[TrainingPlugin]] = None
    ) -> None:
        r"""
        Fit network

        Parameters
        ----------
        data
            Data dataset
        graph
            Graph dataset
        val_split
            Validation split
        data_batch_size
            Number of samples in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        random_seed
            Random seed
        directory
            Directory to store checkpoints and tensorboard logs
        plugins
            Optional list of training plugins
        """
        required_kwargs = (
            "val_split", "data_batch_size", "graph_batch_size",
            "align_burnin", "max_epochs", "random_seed"
        )
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        if patience and reduce_lr_patience and reduce_lr_patience >= patience:
            self.logger.warning(
                "Parameter `reduce_lr_patience` should be smaller than `patience`, "
                "otherwise learning rate scheduling would be ineffective."
            )

        self.enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        self.esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        self.eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        graph.getitem_size = max(1, round(graph_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        # data_train, data_val = data.random_split([1 - val_split, val_split], random_state=random_seed)
        data_train = data
        data_train.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        # data_val.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        graph.prepare_shuffle(num_workers=config.GRAPH_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        train_loader = ParallelDataLoader(
            DataLoader(
                data_train, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            DataLoader(
                graph, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(graph) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            cycle_flags=[False, True]
        )
        val_loader = None
        # val_loader = ParallelDataLoader(
        #     DataLoader(
        #         data_val, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
        #         num_workers=config.DATALOADER_NUM_WORKERS,
        #         pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
        #         generator=torch.Generator().manual_seed(random_seed),
        #         persistent_workers=False
        #     ),
        #     DataLoader(
        #         graph, batch_size=config.DATALOADER_FETCHES_PER_BATCH, shuffle=True,
        #         num_workers=config.DATALOADER_NUM_WORKERS,
        #         pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
        #         generator=torch.Generator().manual_seed(random_seed),
        #         persistent_workers=False
        #     ),
        #     cycle_flags=[False, True]
        # )

        self.align_burnin = align_burnin

        default_plugins = [Tensorboard()]
        if reduce_lr_patience:
            default_plugins.append(LRScheduler(
                self.vae_optim, self.dsc_optim,
                monitor=self.earlystop_loss, patience=reduce_lr_patience,
                burnin=self.align_burnin if safe_burnin else 0
            ))
        if patience:
            default_plugins.append(EarlyStopping(
                monitor=self.earlystop_loss, patience=patience,
                burnin=self.align_burnin if safe_burnin else 0,
                wait_n_lrs=wait_n_lrs or 0
            ))
        plugins = default_plugins + (plugins or [])
        try:
            super().fit(
                train_loader, val_loader=val_loader,
                max_epochs=max_epochs, random_seed=random_seed,
                directory=directory, plugins=plugins
            )
        finally:
            data.clean()
            data_train.clean()
            # data_val.clean()
            graph.clean()
            self.align_burnin = None
            self.eidx = None
            self.enorm = None
            self.esgn = None

    def get_losses(  # pylint: disable=arguments-differ
            self, data: ArrayDataset, graph: GraphDataset,
            data_batch_size: int = None, graph_batch_size: int = None,
            random_seed: int = None
    ) -> Mapping[str, float]:
        required_kwargs = ("data_batch_size", "graph_batch_size", "random_seed")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

        self.enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        self.esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        self.eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        data.getitem_size = data_batch_size
        graph.getitem_size = graph_batch_size
        data.prepare_shuffle(num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        graph.prepare_shuffle(num_workers=config.GRAPH_SHUFFLE_NUM_WORKERS, random_seed=random_seed)

        loader = ParallelDataLoader(
            DataLoader(
                data, batch_size=1, shuffle=True, drop_last=False,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            DataLoader(
                graph, batch_size=1, shuffle=True, drop_last=False,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False
            ),
            cycle_flags=[False, True]
        )

        try:
            losses = super().get_losses(loader)
        finally:
            data.clean()
            graph.clean()
            self.eidx = None
            self.enorm = None
            self.esgn = None

        return losses

    def state_dict(self) -> Mapping[str, Any]:
        return {
            **super().state_dict(),
            "vae_optim": self.vae_optim.state_dict(),
            "dsc_optim": self.dsc_optim.state_dict()
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.vae_optim.load_state_dict(state_dict.pop("vae_optim"))
        self.dsc_optim.load_state_dict(state_dict.pop("dsc_optim"))
        super().load_state_dict(state_dict)

#--------------------------------- Public API ----------------------------------

@logged
class STAMOModel(Model):

    r"""
    STAMO model for spatial multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    NET_TYPE = STAMO
    TRAINER_TYPE = STAMOTrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, AnnData],
            vertices: List[str], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0
    ) -> None:
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        g2v = sc.GraphEncoder(self.vertices.size, latent_dim)
        v2g = sc.GraphDecoder()
        self.modalities, idx, x2u, u2x, all_ct = {}, {}, {}, {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            # x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
            #     data_config["rep_dim"] or len(data_config["features"]), latent_dim,
            #     h_depth=h_depth, h_dim=h_dim, dropout=dropout
            # )
            x2u[k] = sc.STAGATE(hidden_dims=[data_config["rep_dim"], h_dim, latent_dim])
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                len(data_config["features"]),
                n_batches=max(data_config["batches"].size, 1)
            )
            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = sc.Discriminator(
            latent_dim, len(self.modalities), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = sc.Prior()

        atten_cross = sc.AttentionLayer(latent_dim, latent_dim)

        super().__init__(
            g2v, v2g, x2u, u2x, idx, du, prior, atten_cross,
            u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size)
        )

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "STAMOModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(  # pylint: disable=arguments-differ
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            lam_joint_cross: float = 0.02,
            lam_real_cross: float = 0.02,
            lam_cos: float = 0.02,
            lam_gate_rec: float = 0.001,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_joint_cross=lam_joint_cross, lam_real_cross=lam_real_cross,
            lam_cos=lam_cos,lam_gate_rec=lam_gate_rec,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn"
        )
        graph = GraphDataset(
            graph, self.vertices, neg_samples=neg_samples,
            weighted_sampling=True, deemphasize_loops=True
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, graph, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch

        Returns
        -------
        losses
            Loss function values
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        graph = GraphDataset(
            graph, self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True
        )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        return super().get_losses(
            data, graph, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_graph(
            self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        graph_embedding
            Graph (feature) embedding
            with shape :math:`n_{feature} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{feature} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        graph = GraphDataset(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        v = self.net.g2v(eidx, enorm, esgn)
        if n_sample:
            return torch.cat([
                v.sample((1, )).cpu() for _ in range(n_sample)
            ]).permute(1, 0, 2).numpy()
        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )

        x = torch.FloatTensor(adata.X.todense())
        xrep = torch.FloatTensor(data.extracted_data[1][0])
        import numpy as np
        edgeList = np.nonzero(adata.uns['adj'])
        edge_sp = torch.LongTensor(np.array([edgeList[0], edgeList[1]]))

        adj = torch.as_tensor(adata.uns['adj'].todense())
        u = encoder(
            x.to(self.net.device, non_blocking=True),
            xrep.to(self.net.device, non_blocking=True),
            adj.to(self.net.device, non_blocking=True),
            batch_size = x.shape[0]
        )[0]

        if n_sample:
            result = u.sample((n_sample, )).cpu().permute(1, 0, 2)
        else:
            result = u.mean.detach().cpu()

        return result.numpy()

    @torch.no_grad()
    def encode_data_minibatch(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )

        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        result = []
        for x, xrep, xbch, xlbl, xdwt, adj, adj_raw, pmsk in data_loader:
            batch_size = adj_raw.shape[0]
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                adj.to(self.net.device, non_blocking=True),
                batch_size=batch_size
            )[0]
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu()[:batch_size])
        return torch.cat(result).numpy()


    @torch.no_grad()
    def cross_predict(
            self, keys: Tuple[str, str], adata: AnnData, graph: nx.Graph,
            batch_size: int = 128,
            target_libsize: Optional[Union[float, np.ndarray]] = 1.0,
            target_batch: Optional[np.ndarray] = None,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """

        l = target_libsize #or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        self.net.eval()
        source_key, target_key = keys

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        x2u = self.net.x2u[source_key]
        u2x = self.net.u2x[target_key]


        data = AnnDataset(
            [adata], [self.modalities[source_key]],
            mode="eval", getitem_size=batch_size
        )

        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        ## in order to predict all missing features, original feature (not only hvgs) must be used
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=self.net.device)
        v = v[getattr(self.net, f"{target_key}_idx")]

        ### predict by minibatch
        result = []
        for x, xrep, xbch, xlbl, xdwt, adj, adj_raw, pmsk in data_loader:
            batch_size = adj_raw.shape[0]
            u = x2u(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                adj.to(self.net.device, non_blocking=True),
                batch_size=batch_size
            )[0]
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                # result.append(u2x(u[:batch_size], v, None).mean.detach().cpu())
                result.append(u.mean.detach().cpu()[:batch_size])
        u = torch.cat(result).numpy()
        # return torch.cat(result).numpy()


        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(self.net.device, non_blocking=True)
            b_ = b_.to(self.net.device, non_blocking=True)
            l_ = l_.to(self.net.device, non_blocking=True)
            result.append(u2x(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()


    @torch.no_grad()
    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData, graph: nx.Graph,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Decode data

        Parameters
        ----------
        source_key
            Source modality key
        target_key
            Target modality key
        adata
            Source modality data
        graph
            Guidance graph
        target_libsize
            Target modality library size, by default 1.0
        target_batch
            Target modality batch, by default batch 0
        batch_size
            Size of minibatches

        Returns
        -------
        decoded
            Decoded data

        Note
        ----
        This is EXPERIMENTAL!
        """
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, batch_size=batch_size)
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def __repr__(self) -> str:
        return (
            f"STAMO model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )

