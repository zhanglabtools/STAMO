r"""
Integration models
"""

import os
from pathlib import Path
from typing import Mapping, Optional, List

import dill
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from ..typehint import Kws
from ..utils import config, logged, FGWOT_mapping, autodevice
from .base import Model
from .dx import integration_consistency
from .stamo import STAMOModel
import datetime


@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_obs_names: bool = True
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    data_config["use_obs_names"] = use_obs_names
    adata.uns[config.ANNDATA_KEY] = data_config


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file

    Returns
    -------
    model
        Loaded model
    """
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  # pylint: disable=no-member
    model.net.device = autodevice()  # pylint: disable=no-member
    return model

@logged
def train(
        adatas: Mapping[str, AnnData], graph: nx.Graph, model: type = STAMOModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        is_finetune: bool = True, OT_pair: List[str] = [('RNA','ATAC')]
) -> STAMOModel:
    r"""
    Fit STAMO model to integrate spatial multi-omics data

    Parameters
    ----------
    adatas
        Spatial datasets (indexed by modality name)
    graph
        Guidance graph
    model
        Model class, must be one of
        {:class:`STAMO.models.STAMO.STAMOModel`}
    init_kws
        Model initialization keyword arguments
        (see the constructor of the ``model`` class,
        :class:`STAMO.models.STAMO.STAMOModel`)
    compile_kws
        Model compile keyword arguments
        (see the ``compile`` method of the ``model`` class,
        meth:`STAMO.models.STAMO.STAMOModel.compile`)
    fit_kws
        Model fitting keyword arguments
        (see :meth:`STAMO.models.STAMO.STAMOModel.fit`)

    Returns
    -------
    model
        Fitted model object
    """
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}

    ## train.logger.info("Pretraining STAMO model..."+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False}) ## batch correction is not performed during pretrain
    pretrain_fit_kws = fit_kws.copy()
    pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    ## burnin is 0 during pretrain, large learning rate used
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")

    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)

    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    if is_finetune:
        for k, adata in adatas.items():
            adata.obsm["X_STAMO_pretrain"] = pretrain.encode_data_minibatch(k, adata)

        ### use FGWOT to cross slice anchors, then assign same cellnames to the anchors
        for src, dst in OT_pair:
            train.logger.info('FGW OT mapping from source '+src+' to '+dst)

            adata_src_sub = adatas[src].copy()
            adata_dst_sub = adatas[dst].copy()

            sample_size = 5000
            if adatas[src].shape[0] > sample_size: ### downsample to avoid memory issue
                train.logger.info('downsample source data to 5000...')
                adata_src_sub = adatas[src][adatas[src].obs.sample(n=sample_size, random_state=42).index]
            if adatas[dst].shape[0] > sample_size:
                train.logger.info('downsample target data to 5000...')
                adata_dst_sub = adatas[dst][adatas[dst].obs.sample(n=sample_size, random_state=42).index]

            pi = FGWOT_mapping(adata_src_sub, adata_dst_sub)
            pi.to_csv(os.path.join(pretrain_fit_kws["directory"], src+'_'+dst+'_OT_mapping_matrix.csv'))

            ##### Rename the src and dst data to be the re-paired data
            max_indices = np.argmax(pi.values, axis=1)
            filter_cells = np.array(
                ~pd.Series(max_indices).duplicated())  ## remove duplicated target cells in the mapping matrix
            max_indices_filter = np.argmax(pi.values, axis=1)[filter_cells]
            matching_filter = np.array([np.arange(pi.shape[0])[filter_cells], max_indices_filter]) ## local index for subsample data

            id_cell_map_s = pd.DataFrame(np.arange(adatas[src].shape[0]), index=adatas[src].obs_names)
            id_cell_map_t = pd.DataFrame(np.arange(adatas[dst].shape[0]), index=adatas[dst].obs_names)
            ## recover to global index
            matching_global = np.array([id_cell_map_s.loc[pi.index[matching_filter[0, :]], 0].values,
                                        id_cell_map_t.loc[pi.columns[matching_filter[1, :]], 0].values])

            new_cellname = adatas[dst].obs_names.values.copy()
            new_cellname[matching_global[1]] = adatas[src].obs_names.values[matching_global[0]]
            adatas[dst].obs_names = new_cellname

        # train.logger.info("Fine-tuning STAMO model..."+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        finetune_fit_kws = fit_kws.copy()
        if "directory" in finetune_fit_kws:
            finetune_fit_kws["directory"] = \
                os.path.join(finetune_fit_kws["directory"], "fine-tune")

        finetune = model(adatas, sorted(graph.nodes), **init_kws)
        finetune.adopt_pretrained_model(pretrain)
        finetune.compile(**compile_kws)

        train.logger.debug("Increasing random seed by 1 to prevent idential data order...")
        finetune.random_seed += 1
        finetune.fit(adatas, graph, **finetune_fit_kws)
        if "directory" in finetune_fit_kws:
            finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))
        # train.logger.info("End of training..." + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        return finetune

    else:
        return pretrain

