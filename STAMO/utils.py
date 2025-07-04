r"""
Miscellaneous utilities
"""

import os
import logging
import signal
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Process
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from pybedtools.helpers import set_bedtools_path

from .typehint import RandomState, T

AUTO = "AUTO"  # Flag for using automatically determined hyperparameters


#------------------------------ Global containers ------------------------------

processes: Mapping[int, Mapping[int, Process]] = defaultdict(dict)  # id -> pid -> process


#-------------------------------- Meta classes ---------------------------------

class SingletonMeta(type):

    r"""
    Ensure singletons via a meta class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


#--------------------------------- Log manager ---------------------------------

class _CriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


class _NonCriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class LogManager(metaclass=SingletonMeta):

    r"""
    Manage loggers used in the package
    """

    def __init__(self) -> None:
        self._loggers = {}
        self._log_file = None
        self._console_log_level = logging.INFO
        self._file_log_level = logging.DEBUG
        self._file_fmt = \
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
        self._console_fmt = \
            "[%(levelname)s] %(name)s: %(message)s"
        self._date_fmt = "%Y-%m-%d %H:%M:%S"

    @property
    def log_file(self) -> str:
        r"""
        Configure log file
        """
        return self._log_file

    @property
    def file_log_level(self) -> int:
        r"""
        Configure logging level in the log file
        """
        return self._file_log_level

    @property
    def console_log_level(self) -> int:
        r"""
        Configure logging level printed in the console
        """
        return self._console_log_level

    def _create_file_handler(self) -> logging.FileHandler:
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.file_log_level)
        file_handler.setFormatter(logging.Formatter(
            fmt=self._file_fmt, datefmt=self._date_fmt))
        return file_handler

    def _create_console_handler(self, critical: bool) -> logging.StreamHandler:
        if critical:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.addFilter(_CriticalFilter())
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.addFilter(_NonCriticalFilter())
        console_handler.setLevel(self.console_log_level)
        console_handler.setFormatter(logging.Formatter(fmt=self._console_fmt))
        return console_handler

    def get_logger(self, name: str) -> logging.Logger:
        r"""
        Get a logger by name
        """
        if name in self._loggers:
            return self._loggers[name]
        new_logger = logging.getLogger(name)
        new_logger.setLevel(logging.DEBUG)  # lowest level
        new_logger.addHandler(self._create_console_handler(True))
        new_logger.addHandler(self._create_console_handler(False))
        if self.log_file:
            new_logger.addHandler(self._create_file_handler())
        self._loggers[name] = new_logger
        return new_logger

    @log_file.setter
    def log_file(self, file_name: os.PathLike) -> None:
        self._log_file = file_name
        for logger in self._loggers.values():
            for idx, handler in enumerate(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.handlers[idx].close()
                    if self.log_file:
                        logger.handlers[idx] = self._create_file_handler()
                    else:
                        del logger.handlers[idx]
                    break
            else:
                if file_name:
                    logger.addHandler(self._create_file_handler())

    @file_log_level.setter
    def file_log_level(self, log_level: int) -> None:
        self._file_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(self.file_log_level)
                    break

    @console_log_level.setter
    def console_log_level(self, log_level: int) -> None:
        self._console_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler:  # pylint: disable=unidiomatic-typecheck
                    handler.setLevel(self.console_log_level)


log = LogManager()


def logged(obj: T) -> T:
    r"""
    Add logger as an attribute
    """
    obj.logger = log.get_logger(obj.__name__)
    return obj


#---------------------------- Configuration Manager ----------------------------

@logged
class ConfigManager(metaclass=SingletonMeta):

    r"""
    Global configurations
    """

    def __init__(self) -> None:
        self.TMP_PREFIX = "STAMOTMP"
        self.ANNDATA_KEY = "__STAMO__"
        self.CPU_ONLY = False
        self.CUDNN_MODE = "repeatability"
        self.MASKED_GPUS = []
        self.ARRAY_SHUFFLE_NUM_WORKERS = 0
        self.GRAPH_SHUFFLE_NUM_WORKERS = 0  #default = 1
        self.FORCE_TERMINATE_WORKER_PATIENCE = 60
        self.DATALOADER_NUM_WORKERS = 0
        self.DATALOADER_FETCHES_PER_WORKER = 1  #default = 4 will report error
        self.DATALOADER_PIN_MEMORY = True
        self.CHECKPOINT_SAVE_INTERVAL = 100
        self.CHECKPOINT_SAVE_NUMBERS = 1
        self.PRINT_LOSS_INTERVAL = 10
        self.TENSORBOARD_FLUSH_SECS = 5
        self.ALLOW_TRAINING_INTERRUPTION = True
        self.BEDTOOLS_PATH = ""

    @property
    def TMP_PREFIX(self) -> str:
        r"""
        Prefix of temporary files and directories created.
        Default values is ``"STAMOTMP"``.
        """
        return self._TMP_PREFIX

    @TMP_PREFIX.setter
    def TMP_PREFIX(self, tmp_prefix: str) -> None:
        self._TMP_PREFIX = tmp_prefix

    @property
    def ANNDATA_KEY(self) -> str:
        r"""
        Key in ``adata.uns`` for storing dataset configurations.
        Default value is ``"__STAMO__"``
        """
        return self._ANNDATA_KEY

    @ANNDATA_KEY.setter
    def ANNDATA_KEY(self, anndata_key: str) -> None:
        self._ANNDATA_KEY = anndata_key

    @property
    def CPU_ONLY(self) -> bool:
        r"""
        Whether computation should use only CPUs.
        Default value is ``False``.
        """
        return self._CPU_ONLY

    @CPU_ONLY.setter
    def CPU_ONLY(self, cpu_only: bool) -> None:
        self._CPU_ONLY = cpu_only
        if self._CPU_ONLY and self._DATALOADER_NUM_WORKERS:
            self.logger.warning(
                "It is recommended to set `DATALOADER_NUM_WORKERS` to 0 "
                "when using CPU_ONLY mode. Otherwise, deadlocks may happen "
                "occationally."
            )

    @property
    def CUDNN_MODE(self) -> str:
        r"""
        CuDNN computation mode, should be one of {"repeatability", "performance"}.
        Default value is ``"repeatability"``.

        Note
        ----
        As of now, due to the use of :meth:`torch.Tensor.scatter_add_`
        operation, the results are not completely reproducible even when
        ``CUDNN_MODE`` is set to ``"repeatability"``, if GPU is used as
        computation device. Exact repeatability can only be achieved on CPU.
        The situtation might change with new releases of :mod:`torch`.
        """
        return self._CUDNN_MODE

    @CUDNN_MODE.setter
    def CUDNN_MODE(self, cudnn_mode: str) -> None:
        if cudnn_mode not in ("repeatability", "performance"):
            raise ValueError("Invalid mode!")
        self._CUDNN_MODE = cudnn_mode
        torch.backends.cudnn.deterministic = self._CUDNN_MODE == "repeatability"
        torch.backends.cudnn.benchmark = self._CUDNN_MODE == "performance"

    @property
    def MASKED_GPUS(self) -> List[int]:
        r"""
        A list of GPUs that should not be used when selecting computation device.
        This must be set before initializing any model, otherwise would be ineffective.
        Default value is ``[]``.
        """
        return self._MASKED_GPUS

    @MASKED_GPUS.setter
    def MASKED_GPUS(self, masked_gpus: List[int]) -> None:
        if masked_gpus:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for item in masked_gpus:
                if item >= device_count:
                    raise ValueError(f"GPU device \"{item}\" is non-existent!")
        self._MASKED_GPUS = masked_gpus

    @property
    def ARRAY_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for array data shuffling.
        Default value is ``0``.
        """
        return self._ARRAY_SHUFFLE_NUM_WORKERS

    @ARRAY_SHUFFLE_NUM_WORKERS.setter
    def ARRAY_SHUFFLE_NUM_WORKERS(self, array_shuffle_num_workers: int) -> None:
        self._ARRAY_SHUFFLE_NUM_WORKERS = array_shuffle_num_workers

    @property
    def GRAPH_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for graph data shuffling.
        Default value is ``1``.
        """
        return self._GRAPH_SHUFFLE_NUM_WORKERS

    @GRAPH_SHUFFLE_NUM_WORKERS.setter
    def GRAPH_SHUFFLE_NUM_WORKERS(self, graph_shuffle_num_workers: int) -> None:
        self._GRAPH_SHUFFLE_NUM_WORKERS = graph_shuffle_num_workers

    @property
    def FORCE_TERMINATE_WORKER_PATIENCE(self) -> int:
        r"""
        Seconds to wait before force terminating unresponsive workers.
        Default value is ``60``.
        """
        return self._FORCE_TERMINATE_WORKER_PATIENCE

    @FORCE_TERMINATE_WORKER_PATIENCE.setter
    def FORCE_TERMINATE_WORKER_PATIENCE(self, force_terminate_worker_patience: int) -> None:
        self._FORCE_TERMINATE_WORKER_PATIENCE = force_terminate_worker_patience

    @property
    def DATALOADER_NUM_WORKERS(self) -> int:
        r"""
        Number of worker processes to use in data loader.
        Default value is ``0``.
        """
        return self._DATALOADER_NUM_WORKERS

    @DATALOADER_NUM_WORKERS.setter
    def DATALOADER_NUM_WORKERS(self, dataloader_num_workers: int) -> None:
        if dataloader_num_workers > 8:
            self.logger.warning(
                "Worker number 1-8 is generally sufficient, "
                "too many workers might have negative impact on speed."
            )
        self._DATALOADER_NUM_WORKERS = dataloader_num_workers

    @property
    def DATALOADER_FETCHES_PER_WORKER(self) -> int:
        r"""
        Number of fetches per worker per batch to use in data loader.
        Default value is ``4``.
        """
        return self._DATALOADER_FETCHES_PER_WORKER

    @DATALOADER_FETCHES_PER_WORKER.setter
    def DATALOADER_FETCHES_PER_WORKER(self, dataloader_fetches_per_worker: int) -> None:
        self._DATALOADER_FETCHES_PER_WORKER = dataloader_fetches_per_worker

    @property
    def DATALOADER_FETCHES_PER_BATCH(self) -> int:
        r"""
        Number of fetches per batch in data loader (read-only).
        """
        return max(1, self.DATALOADER_NUM_WORKERS) * self.DATALOADER_FETCHES_PER_WORKER

    @property
    def DATALOADER_PIN_MEMORY(self) -> bool:
        r"""
        Whether to use pin memory in data loader.
        Default value is ``True``.
        """
        return self._DATALOADER_PIN_MEMORY

    @DATALOADER_PIN_MEMORY.setter
    def DATALOADER_PIN_MEMORY(self, dataloader_pin_memory: bool):
        self._DATALOADER_PIN_MEMORY = dataloader_pin_memory

    @property
    def CHECKPOINT_SAVE_INTERVAL(self) -> int:
        r"""
        Automatically save checkpoints every n epochs.
        Default value is ``10``.
        """
        return self._CHECKPOINT_SAVE_INTERVAL

    @CHECKPOINT_SAVE_INTERVAL.setter
    def CHECKPOINT_SAVE_INTERVAL(self, checkpoint_save_interval: int) -> None:
        self._CHECKPOINT_SAVE_INTERVAL = checkpoint_save_interval

    @property
    def CHECKPOINT_SAVE_NUMBERS(self) -> int:
        r"""
        Maximal number of checkpoints to preserve at any point.
        Default value is ``3``.
        """
        return self._CHECKPOINT_SAVE_NUMBERS

    @CHECKPOINT_SAVE_NUMBERS.setter
    def CHECKPOINT_SAVE_NUMBERS(self, checkpoint_save_numbers: int) -> None:
        self._CHECKPOINT_SAVE_NUMBERS = checkpoint_save_numbers

    @property
    def PRINT_LOSS_INTERVAL(self) -> int:
        r"""
        Print loss values every n epochs.
        Default value is ``10``.
        """
        return self._PRINT_LOSS_INTERVAL

    @PRINT_LOSS_INTERVAL.setter
    def PRINT_LOSS_INTERVAL(self, print_loss_interval: int) -> None:
        self._PRINT_LOSS_INTERVAL = print_loss_interval

    @property
    def TENSORBOARD_FLUSH_SECS(self) -> int:
        r"""
        Flush tensorboard logs to file every n seconds.
        Default values is ``5``.
        """
        return self._TENSORBOARD_FLUSH_SECS

    @TENSORBOARD_FLUSH_SECS.setter
    def TENSORBOARD_FLUSH_SECS(self, tensorboard_flush_secs: int) -> None:
        self._TENSORBOARD_FLUSH_SECS = tensorboard_flush_secs

    @property
    def ALLOW_TRAINING_INTERRUPTION(self) -> bool:
        r"""
        Allow interruption before model training converges.
        Default values is ``True``.
        """
        return self._ALLOW_TRAINING_INTERRUPTION

    @ALLOW_TRAINING_INTERRUPTION.setter
    def ALLOW_TRAINING_INTERRUPTION(self, allow_training_interruption: bool) -> None:
        self._ALLOW_TRAINING_INTERRUPTION = allow_training_interruption

    @property
    def BEDTOOLS_PATH(self) -> str:
        r"""
        Path to bedtools executable.
        Default value is ``bedtools``.
        """
        return self._BEDTOOLS_PATH

    @BEDTOOLS_PATH.setter
    def BEDTOOLS_PATH(self, bedtools_path: str) -> None:
        self._BEDTOOLS_PATH = bedtools_path
        set_bedtools_path(bedtools_path)


config = ConfigManager()


#---------------------------- Interruption handling ----------------------------

@logged
class DelayedKeyboardInterrupt:  # pragma: no cover

    r"""
    Shield a code block from keyboard interruptions, delaying handling
    till the block is finished (adapted from
    `https://stackoverflow.com/a/21919644
    <https://stackoverflow.com/a/21919644>`__).
    """

    def __init__(self):
        self.signal_received = None
        self.old_handler = None

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self._handler)

    def _handler(self, sig, frame):
        self.signal_received = (sig, frame)
        self.logger.debug("SIGINT received, delaying KeyboardInterrupt...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


#--------------------------- Constrained data frame ----------------------------

@logged
class ConstrainedDataFrame(pd.DataFrame):

    r"""
    Data frame with certain format constraints

    Note
    ----
    Format constraints are checked and maintained automatically.
    """

    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.verify(self)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Rectify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be rectified

        Returns
        -------
        rectified_df
            Rectified data frame
        """
        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        r"""
        Verify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be verified
        """

    @property
    def df(self) -> pd.DataFrame:
        r"""
        Convert to regular data frame
        """
        return pd.DataFrame(self)

    def __repr__(self) -> str:
        r"""
        Note
        ----
        We need to explicitly call :func:`repr` on the regular data frame
        to bypass integrity verification, because when the terminal is
        too narrow, :mod:`pandas` would split the data frame internally,
        causing format verification to fail.
        """
        return repr(self.df)


#--------------------------- Other utility functions ---------------------------

def get_chained_attr(x: Any, attr: str) -> Any:
    r"""
    Get attribute from an object, with support for chained attribute names.

    Parameters
    ----------
    x
        Object to get attribute from
    attr
        Attribute name

    Returns
    -------
    attr_value
        Attribute value
    """
    for k in attr.split("."):
        if not hasattr(x, k):
            raise AttributeError(f"{attr} not found!")
        x = getattr(x, k)
    return x


def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random


@logged
def run_command(
        command: str,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        log_command: bool = True, print_output: bool = True,
        err_message: Optional[Mapping[int, str]] = None, **kwargs
) -> Optional[List[str]]:
    r"""
    Run an external command and get realtime output

    Parameters
    ----------
    command
        A string containing the command to be executed
    stdout
        Where to redirect stdout
    stderr
        Where to redirect stderr
    echo_command
        Whether to log the command being printed (log level is INFO)
    print_output
        Whether to print stdout of the command.
        If ``stdout`` is PIPE and ``print_output`` is set to False,
        the output will be returned as a list of output lines.
    err_message
        Look up dict of error message (indexed by error code)
    **kwargs
        Other keyword arguments to be passed to :class:`subprocess.Popen`

    Returns
    -------
    output_lines
        A list of output lines (only returned if ``stdout`` is PIPE
        and ``print_output`` is False)
    """
    if log_command:
        run_command.logger.info("Executing external command: %s", command)
    executable = command.split(" ")[0]
    with subprocess.Popen(command, stdout=stdout, stderr=stderr,
                          shell=True, **kwargs) as p:
        if stdout == subprocess.PIPE:
            prompt = f"{executable} ({p.pid}): "
            output_lines = []

            def _handle(line):
                line = line.strip().decode()
                if print_output:
                    print(prompt + line)
                else:
                    output_lines.append(line)

            while True:
                _handle(p.stdout.readline())
                ret = p.poll()
                if ret is not None:
                    # Handle output between last readlines and successful poll
                    for line in p.stdout.readlines():
                        _handle(line)
                    break
        else:
            output_lines = None
            ret = p.wait()
    if ret != 0:
        err_message = err_message or {}
        if ret in err_message:
            err_message = " " + err_message[ret]
        elif "__default__" in err_message:
            err_message = " " + err_message["__default__"]
        else:
            err_message = ""
        raise RuntimeError(
            f"{executable} exited with error code: {ret}.{err_message}")
    if stdout == subprocess.PIPE and not print_output:
        return output_lines


import functools
import pynvml
from torch.nn.modules.batchnorm import _NormBase

#----------------------------- Utility functions -------------------------------
def freeze_running_stats(m: torch.nn.Module) -> None:
    r"""
    Selectively stops normalization layers from updating running stats

    Parameters
    ----------
    m
        Network module
    """
    if isinstance(m, _NormBase):
        m.eval()


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    based on GPU availability and memory usage

    Returns
    -------
    device
        Computation device
    """
    used_device = -1
    if not config.CPU_ONLY:
        try:
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free for i in range(pynvml.nvmlDeviceGetCount())
            ])
            if free_mems.size:
                for item in config.MASKED_GPUS:
                    free_mems[item] = -1
                best_devices = np.where(free_mems == free_mems.max())[0]
                used_device = np.random.choice(best_devices, 1)[0]
                if free_mems[used_device] < 0:
                    used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda:"+str(used_device))
    # return torch.device("cuda")

#----------------------------- new utility functions added by STAMO -------------------------------

from typing import Optional
import sklearn
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
# from harmony import harmonize

# def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=666):
#     """\
#     Clustering using the mclust algorithm.
#     The parameters are the same as those in the R package mclust.
#     """

#     np.random.seed(random_seed)
#     import rpy2.robjects as robjects
#     robjects.r.library("mclust")

#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
#     r_random_seed = robjects.r['set.seed']
#     r_random_seed(random_seed)
#     rmclust = robjects.r['Mclust']

#     res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
#     mclust_res = np.array(res[-2])

#     adata.obs['mclust'] = mclust_res
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata

import scipy.sparse as sp
def refine_clusters(result, adj, p=0.5):
    """
    Reassigning Cluster Labels Using Spatial Domain Information.

    Parameters
    ----------
    result
        Clustering result to refine.
    adj
        Adjcency matrix.
    k_neighbors or min_distance
        Different way to calculate adj.
    p : float (default: 0.5)
        Rate of label changes in terms of neighbors

    Returns
    -------
    Check post_processed cluster label.
    """
    if sp.issparse(adj):
        adj = adj.A

    pred_after = []
    for i in range(result.shape[0]):
        temp = list(adj[i])
        temp_list = []
        for index, value in enumerate(temp):
            if value > 0:
                temp_list.append(index)
        self_pred = result[i]
        neighbour_pred = []
        for j in temp_list:
            neighbour_pred.append(result[j])
        if (neighbour_pred.count(self_pred) < (len(neighbour_pred)) * p) and (
                neighbour_pred.count(max(set(neighbour_pred), key=neighbour_pred.count)) > (len(neighbour_pred)) * p):
            pred_after.append(np.argmax(np.bincount(np.array(neighbour_pred))))
        else:
            pred_after.append(self_pred)
    return np.array(pred_after)

def cluster_post_process(adata, clutser_result, key_added="refine_clustering", p=0.5, run_times=3):
    """
    Post_processing tool for cluster label that integrates neighborhood information.

    Parameters
    ----------
    adata : Anndata
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    clutser_result: int array
        Array of cluster labels
    key_added : str (default: 'pp_clustering')
        `adata.obs` key under which to add the cluster labels.
    p : float (default: 0.5)
        Rate of label changes in terms of neighbors.
    run_times : int (default: 3)
        Number of post-process runs. If the label does not change in two consecutive
        processes, the run is also terminated.

    Returns
    -------
    adata.obs[key_added]
        Array of dim (number of samples) that stores the post-processed cluster
        label for each cell.
    """

    print("\nPost-processing for clustering result ...")
    result_final = pd.DataFrame(np.zeros(clutser_result.shape[0]))
    i = 1
    while True:
        clutser_result = refine_clusters(clutser_result, adata.uns["adj"], p)  # renew labels in each loop
        print("Refining clusters, run times: {}/{}".format(i, run_times))
        result_final.loc[:, i] = clutser_result
        if result_final.loc[:, i].equals(result_final.loc[:, i - 1]) or i == run_times:
            adata.obs[key_added] = np.array(result_final.loc[:, i])
            adata.obs[key_added] = adata.obs[key_added].astype('category')
            return adata
        i += 1

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    # G_df.loc['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    # G_df.loc['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G
    adata.uns['edgeList'] = np.nonzero(adata.uns['adj'])


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.show()


def split_adata_ob(ads, ad_ref, ob='obs', key='emb'):
    len_ads = [_.n_obs for _ in ads]
    if ob == 'obsm':
        split_obsms = np.split(ad_ref.obsm[key], np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obsms):
            ad.obsm[key] = v
    else:
        split_obs = np.split(ad_ref.obs[key].to_list(), np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obs):
            ad.obs[key] = v


class tfidfTransformer:
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / (1e-8 + X.sum(axis=0))
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / (1e-8 + X.sum(axis=1)))
            return tf.multiply(self.idf)
        else:
            tf = X / (1e-8 + X.sum(axis=1, keepdims=True))
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# optional, other reasonable preprocessing steps also ok
class lsiTransformer:
    def __init__(
            self, n_components: int = 20, drop_first=True, use_highly_variable=None, log=True, norm=True, z_score=True,
            tfidf=True, svd=True, use_counts=False, pcaAlgo='arpack'
    ):

        self.drop_first = drop_first
        self.n_components = n_components + drop_first
        self.use_highly_variable = use_highly_variable

        self.log = log
        self.norm = norm
        self.z_score = z_score
        self.svd = svd
        self.tfidf = tfidf
        self.use_counts = use_counts

        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(
            n_components=self.n_components, random_state=777, algorithm=pcaAlgo
        )
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        if self.use_highly_variable is None:
            self.use_highly_variable = "highly_variable" in adata.var
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X = adata_use.layers['counts']
        else:
            X = adata_use.X
        if self.tfidf:
            X = self.tfidfTransformer.fit_transform(X)
        if scipy.sparse.issparse(X):
            X = X.A.astype("float32")
        if self.norm:
            X = self.normalizer.fit_transform(X)
        if self.log:
            X = np.log1p(X * 1e4)  # L1-norm and target_sum=1e4 and log1p
        self.pcaTransformer.fit(X)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError("Transformer was not fitted on any data")
        adata_use = (
            adata[:, adata.var["highly_variable"]]
            if self.use_highly_variable
            else adata
        )
        if self.use_counts:
            X_pp = adata_use.layers['counts']
        else:
            X_pp = adata_use.X
        if self.tfidf:
            X_pp = self.tfidfTransformer.transform(X_pp)
        if scipy.sparse.issparse(X_pp):
            X_pp = X_pp.A.astype("float32")
        if self.norm:
            X_pp = self.normalizer.transform(X_pp)
        if self.log:
            X_pp = np.log1p(X_pp * 1e4)
        if self.svd:
            X_pp = self.pcaTransformer.transform(X_pp)
        if self.z_score:
            X_pp -= X_pp.mean(axis=1, keepdims=True)
            X_pp /= (1e-8 + X_pp.std(axis=1, ddof=1, keepdims=True))
        pp_df = pd.DataFrame(X_pp, index=adata_use.obs_names).iloc[
                :, int(self.drop_first):
                ]
        return pp_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)


# CLR-normalization
def clr_normalize(adata):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    # sc.pp.pca(adata, n_comps=min(50, adata.n_vars-1))
    return adata


# def harmony(latent, batch_labels, use_gpu=True):
#     df_batches = pd.DataFrame(np.reshape(batch_labels, (-1, 1)), columns=['batch'])
#     bc_latent = harmonize(
#         latent, df_batches, batch_key="batch", use_gpu=use_gpu, verbose=True
#     )
#     return bc_latent


def RNA_preprocess(rna_ad, n_hvg=5000, key='X_pca'):
    sc.pp.highly_variable_genes(rna_ad, n_top_genes=n_hvg, flavor="seurat_v3") #, batch_key=batch_key
    sc.pp.normalize_total(rna_ad, target_sum=1e4)
    sc.pp.log1p(rna_ad)
    sc.pp.scale(rna_ad)
    ## ad_concat = ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()].copy()
    sc.pp.pca(rna_ad, n_comps=min(50, rna_ad.n_vars - 1), use_highly_variable=True)

    return rna_ad.obsm[key]

# def RNA_preprocess(rna_ads, batch_corr=False, n_hvg=5000, lognorm=True, scale=False, batch_key='Batch',
#                    key='X_pca'):
#     measured_ads = [ad for ad in rna_ads if ad is not None]
#     ad_concat = sc.concat(measured_ads)
#     if lognorm:
#         sc.pp.normalize_total(ad_concat, target_sum=1e4)
#         sc.pp.log1p(ad_concat)
#     if n_hvg:
#         sc.pp.highly_variable_genes(ad_concat, n_top_genes=n_hvg) #, batch_key=batch_key
#         # ad_concat = ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()].copy()
#     if scale:
#         sc.pp.scale(ad_concat)
#     sc.pp.pca(ad_concat, n_comps=min(50, ad_concat.n_vars - 1), use_highly_variable=True)
#
#     if len(measured_ads) > 1 and batch_corr: ## correct batch effects between slices in the same modality
#         pass
#         # ad_concat.obsm[key] = harmony(
#         #     ad_concat.obsm['X_pca'],
#         #     ad_concat.obs[batch_key].to_list(),
#         #     use_gpu=True
#         # )
#     else:
#         ad_concat.obsm[key] = ad_concat.obsm['X_pca']
#     # split_adata_ob([ad for ad in rna_ads if ad is not None], ad_concat, ob='obsm', key=key)
#     return ad_concat

#
# def ADT_preprocess(adt_ads, batch_corr=False, favor='clr', lognorm=True, scale=False, batch_key='Batch', key='X_pca'):
#     measured_ads = [ad for ad in adt_ads if ad is not None]
#     ad_concat = sc.concat(measured_ads)
#     if favor == 'clr':
#         ad_concat = clr_normalize(ad_concat)
#         # if scale: sc.pp.scale(ad_concat)
#     else:
#         if lognorm:
#             sc.pp.normalize_total(ad_concat, target_sum=1e4)
#             sc.pp.log1p(ad_concat)
#         if scale: sc.pp.scale(ad_concat)
#
#     sc.pp.pca(ad_concat, n_comps=min(50, ad_concat.n_vars - 1))
#
#     if len(measured_ads) > 1 and batch_corr:
#         ad_concat.obsm[key] = harmony(ad_concat.obsm['X_pca'], ad_concat.obs[batch_key].to_list(), use_gpu=True)
#     else:
#         ad_concat.obsm[key] = ad_concat.obsm['X_pca']
#     # split_adata_ob([ad for ad in adt_ads if ad is not None], ad_concat, ob='obsm', key=key)
#     return ad_concat

def Epigenome_preprocess(epi_ads, n_peak=100000, key='X_lsi'):
    measured_ads = [ad for ad in epi_ads if ad is not None]
    ad_concat = sc.concat(measured_ads)
    sc.pp.highly_variable_genes(ad_concat, flavor='seurat_v3', n_top_genes=n_peak)  ## , batch_key=batch_key

    transformer = lsiTransformer(n_components=50, drop_first=True, log=True, norm=True, z_score=True, tfidf=True,
                                 svd=True, pcaAlgo='arpack')
    ad_concat.obsm[key] = transformer.fit_transform(
        ad_concat[:, ad_concat.var.query('highly_variable').index.to_numpy()]).values

    # if len(measured_ads) > 1 and batch_corr:  ## correct batch effects between slices in the same modality
    #     # ad_concat.obsm[key] = harmony(ad_concat.obsm['X_lsi'], ad_concat.obs[batch_key].to_list(), use_gpu=True)
    #     pass
    # else:
    #     ad_concat.obsm[key] = ad_concat.obsm['X_lsi']

    # split_adata_ob([ad for ad in epi_ads if ad is not None], ad_concat, ob='obsm', key=key)

    return ad_concat.obsm['X_lsi']


from scipy.spatial import distance
import ot
def FGWOT_mapping(adata1, adata2, alpha=0.5):
    embed1 = adata1.obsm["X_STAMO_pretrain"]
    embed2 = adata2.obsm["X_STAMO_pretrain"]
    coord1 = adata1.obsm['spatial']
    coord2 = adata2.obsm['spatial']

    ### spatial coordinate distance between spots in each slice
    nx = ot.backend.NumpyBackend()
    a = np.float64(nx.from_numpy(coord1))
    b = np.float64(nx.from_numpy(coord2))
    D1 = ot.dist(a, a, metric='euclidean')
    D2 = ot.dist(b, b, metric='euclidean')
    # if self.norm:
    #     D1 /= nx.min(D1[D1 > 0])
    #     D2 /= nx.min(D2[D2 > 0])

    #### Euclidean distance of latent vectors
    X1, X2 = nx.from_numpy(embed1), nx.from_numpy(embed2)

    dissimilarity = 'euclidean'  ## 'cosine'
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'cosine':
        M = ot.dist(X1, X2, metric=dissimilarity)
    else:
        M = distance.cdist(X1, X2, metric='cosine')

    D1 /= D1[D1 > 0].max()
    D1 *= M.max()
    D2 /= D2[D2 > 0].max()
    D2 *= M.max()

    ####  uniform distribution for spots
    d1 = nx.ones((embed1.shape[0],)) / embed1.shape[0]
    d2 = nx.ones((embed2.shape[0],)) / embed2.shape[0]

    G0 = d1[:, None] * d2[None, :]

    C1, C2, p, q, G_init = D1, D2, d1, d2, G0
    loss_fun = 'square_loss'  # 'kl_loss' #'square_loss' #
    armijo = False

    ## armijo (bool, optional) – If True the step of the line-search is found via an armijo research.
    ## Else closed form is used. If there are convergence issues use False.
    # if armijo:
    #     def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
    #         return ot.optim.line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs)
    # else:
    #     def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
    #         return solve_gromov_linesearch(G, deltaG, cost_G, C1, C2, M=0., reg=1., nx=nx, **kwargs)

    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    res, logw = ot.gromov.semirelaxed_fused_gromov_wasserstein(M, C1, C2, p=p, loss_fun=loss_fun, alpha=alpha, log=True,
                                                               verbose=False)
    pi = pd.DataFrame(res, index=adata1.obs_names, columns=adata2.obs_names)


    ###
    if 'annotation' in adata1.obs.columns and 'annotation' in adata2.obs.columns:
        max_indices = np.argmax(pi.values, axis=1)
        matching = np.array([range(res.shape[0]), max_indices])
        overall_score = sum(
            [adata1.obs['annotation'].values[ii] == adata2.obs['annotation'][matching.T[:, 1]].values[ii] for ii in
             range(adata1.shape[0])]) / adata1.shape[0]
        print("How many source profiles were paired to an destination profile of the same cluster type:",overall_score)
    return pi


from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
## ds1 query pool; ds2 candicate pool
def nn_approx(ds1, ds2, norm=True, knn=15, metric='manhattan', n_trees=10, include_distances=False):
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2)

        # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind, dist = [], []
    for i in range(ds1.shape[0]):
        i_ind, i_dist = a.get_nns_by_vector(ds1[i, :], knn, search_k=-1, include_distances=True)
        ind.append(i_ind)
        dist.append(i_dist)
    ind = np.array(ind)

    if include_distances:
        return ind, np.array(dist)
    else:
        return ind

def cross_predict(input_adata, ref_adata, model, guidance, OT_pair=("H3K27ac", "RNA")):
    """
    Cross-omics generation: generate missing omics data from input_adata, using ref_adata as reference

    Args:
        input_adata (anndata.AnnData): Measured omics data as model input for generation.
        ref_adata (anndata.AnnData): Reference data for corresponding missing omics
        model (STAMOModel): Fitted model object
        guidance: Guidance graph
    Returns:
        Generated adata
    """

    knn_ind = nn_approx(input_adata.obsm["X_STAMO"], ref_adata.obsm["X_STAMO"], knn=15)
    target_pool = ref_adata.X.toarray()
    p_q = target_pool[knn_ind.ravel()].reshape(*(knn_ind.shape), ref_adata.X.shape[1])  # p_q.shape = (9548, 10, 19588)
    imps = np.mean(p_q, axis=1)

    predict_X = model.cross_predict(OT_pair, input_adata, guidance, target_libsize=imps.sum(axis=1))

    predict_adata = sc.AnnData(predict_X)
    predict_adata.X = sp.csr_matrix(predict_adata.X)
    predict_adata.var_names = ref_adata.var_names
    predict_adata.obsm['spatial'] = ref_adata.obsm['spatial']

    return predict_adata
