"""Regression tests for graph_conformal's train/valid/calib/test and tune/qscore splits."""

import os

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl  # noqa: E402
import numpy as np  # noqa: E402
import pyrallis.cfgparsing as pyr_c  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import graph_conformal.utils as gutils  # noqa: E402  (must precede other repo imports)
from graph_conformal.config import BaseExptConfig, DatasetSplitConfig  # noqa: E402
from graph_conformal.constants import (  # noqa: E402
    FEATURE_FIELD,
    LABEL_FIELD,
    PREDDEF_FIELD,
    PreDefSplit,
    Stage,
)
from graph_conformal.data_module import ClassificationDataset  # noqa: E402

N = 1000
STAGES = (Stage.TRAIN, Stage.VALIDATION, Stage.CALIBRATION, Stage.TEST)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _graph(predefined=False):
    g = dgl.graph((torch.arange(N), (torch.arange(N) + 1) % N))
    g.ndata[LABEL_FIELD] = torch.arange(N) % 3
    g.ndata[FEATURE_FIELD] = torch.randn(N, 4)
    if predefined:  # first 30% train, next 10% valid, rest test/calib
        split = torch.full((N,), float(PreDefSplit.TESTCALIB))
        split[: int(0.3 * N)] = PreDefSplit.TRAIN
        split[int(0.3 * N): int(0.4 * N)] = PreDefSplit.VALIDATION
        g.ndata[PREDDEF_FIELD] = split
    return g


def _dataset(name="Cora", predefined=False, seed=0, nspc=None):
    style = "split" if nspc is None else "n_samples_per_class"
    args = BaseExptConfig(dataset=name, seed=seed, dataset_loading_style=style, dataset_n_samples_per_class=nspc,
                          dataset_split_fractions=DatasetSplitConfig(train=0.2, valid=0.1, calib=0.35))
    gutils.set_seed_and_precision(seed)  # as train_base_gnn.py / conformal_trials.py do
    return ClassificationDataset(name, graph=_graph(predefined), args=args)


def _masks(ds):
    return {s: ds.graph.ndata[s.mask_dstr].clone() for s in STAGES}


def test_resplit_keeps_training_split():
    ds = _dataset()
    first = _masks(ds)
    for trial in range(3):
        ds.resplit_calib_test(trial, False)
        m = _masks(ds)
        assert torch.equal(m[Stage.TRAIN], first[Stage.TRAIN])
        assert torch.equal(m[Stage.VALIDATION], first[Stage.VALIDATION])
        used = first[Stage.TRAIN] | first[Stage.VALIDATION]
        assert not (m[Stage.CALIBRATION] & used).any()
        assert not (m[Stage.TEST] & used).any()
        assert torch.equal(m[Stage.CALIBRATION] | m[Stage.TEST], ~used)


def test_resplit_changes_calibration_set():
    ds = _dataset()
    ds.resplit_calib_test(0, False)
    c0 = _masks(ds)[Stage.CALIBRATION]
    ds.resplit_calib_test(1, False)
    assert not torch.equal(c0, _masks(ds)[Stage.CALIBRATION])


def test_training_split_matches_training_run():
    # the order drawn after seeding must match what a fresh training run draws
    a, b = _dataset(seed=3), _dataset(seed=3)
    b.resplit_calib_test(7, False)
    assert torch.equal(_masks(a)[Stage.TRAIN], _masks(b)[Stage.TRAIN])


@pytest.mark.parametrize("resplit", [None, 0, 5])
def test_tune_qscore_split_is_random_partition(resplit):
    ds = _dataset()
    if resplit is not None:
        ds.resplit_calib_test(resplit, False)
    ds.split_graph_calib_tune_qscore(0.5)
    calib = ds.graph.ndata[Stage.CALIBRATION.mask_dstr]
    tune = ds.graph.ndata[Stage.CALIBRATION_TUNE.mask_dstr]
    qs = ds.graph.ndata[Stage.CALIBRATION_QSCORE.mask_dstr]
    assert not (tune & qs).any()
    assert torch.equal(tune | qs, calib)
    assert tune.sum() == int(0.5 * calib.sum())
    # not the old rule (lowest-index half): both halves of the index range appear in each part
    idx = calib.nonzero().ravel()
    median = idx[len(idx) // 2]
    for part in (tune, qs):
        p = part.nonzero().ravel()
        assert 0.3 < (p < median).float().mean() < 0.7


def test_tune_qscore_split_is_reproducible():
    a, b = _dataset(), _dataset()
    for ds in (a, b):
        ds.resplit_calib_test(2, False)
        ds.split_graph_calib_tune_qscore(0.5)
    assert torch.equal(a.graph.ndata[Stage.CALIBRATION_TUNE.mask_dstr],
                       b.graph.ndata[Stage.CALIBRATION_TUNE.mask_dstr])


def test_predefined_split_calib_not_index_ordered():
    ds = _dataset(name="Flickr", predefined=True)
    calib = _masks(ds)[Stage.CALIBRATION].nonzero().ravel()
    pool = (ds.graph.ndata[PREDDEF_FIELD] == PreDefSplit.TESTCALIB).nonzero().ravel()
    # old behavior: calibration = first len(calib) pool nodes by index
    assert not torch.equal(calib, pool[: len(calib)])


def test_saved_split_configs_still_load():
    path = os.path.join(REPO, "configs/custom_configs/best_base_configs/Cora/split/0.2_0.1/basegnn_config.yaml")
    with open(path) as f:
        cfg = pyr_c.load(BaseExptConfig, f)
    assert cfg.dataset_loading_style == "split" and cfg.dataset_n_samples_per_class is None


# --- n_samples_per_class -----------------------------------------------------
NSPC = 10


@pytest.mark.parametrize("predefined", [False, True])
def test_nspc_resplit_keeps_training_split(predefined):
    name = "Flickr" if predefined else "Cora"
    ds = _dataset(name=name, predefined=predefined, nspc=NSPC)
    first = _masks(ds)
    assert first[Stage.TRAIN].sum() == 3 * NSPC
    for trial in range(3):
        ds.resplit_calib_test(trial, False)
        m = _masks(ds)
        assert torch.equal(m[Stage.TRAIN], first[Stage.TRAIN])
        assert torch.equal(m[Stage.VALIDATION], first[Stage.VALIDATION])
        used = first[Stage.TRAIN] | first[Stage.VALIDATION]
        assert not ((m[Stage.CALIBRATION] | m[Stage.TEST]) & used).any()
        assert not (m[Stage.CALIBRATION] & m[Stage.TEST]).any()


def test_nspc_resplit_uses_all_remaining_nodes():
    # with a resplit seed, calib + test must cover every non-train/valid node
    # (the old slicing dropped 2 * n_samples_per_class nodes per class)
    ds = _dataset(nspc=NSPC)
    first = _masks(ds)
    remaining = ~(first[Stage.TRAIN] | first[Stage.VALIDATION])
    assert torch.equal(first[Stage.CALIBRATION] | first[Stage.TEST], remaining)
    ds.resplit_calib_test(4, False)
    m = _masks(ds)
    assert torch.equal(m[Stage.CALIBRATION] | m[Stage.TEST], remaining)
    labels = ds.graph.ndata[LABEL_FIELD]
    for c in range(3):
        assert (m[Stage.CALIBRATION] & (labels == c)).sum() == NSPC


def test_nspc_training_split_matches_training_run():
    a, b = _dataset(seed=3, nspc=NSPC), _dataset(seed=3, nspc=NSPC)
    b.resplit_calib_test(7, False)
    assert torch.equal(_masks(a)[Stage.TRAIN], _masks(b)[Stage.TRAIN])


@pytest.mark.parametrize("resplit", [None, 2])
def test_nspc_tune_qscore_uses_calibration_nodes(resplit):
    ds = _dataset(nspc=NSPC)
    if resplit is not None:
        ds.resplit_calib_test(resplit, False)
    ds.split_graph_calib_tune_qscore(0.5)
    calib = ds.graph.ndata[Stage.CALIBRATION.mask_dstr]
    tune = ds.graph.ndata[Stage.CALIBRATION_TUNE.mask_dstr]
    qs = ds.graph.ndata[Stage.CALIBRATION_QSCORE.mask_dstr]
    assert not (tune & qs).any()
    assert torch.equal(tune | qs, calib)  # old code marked nodes 0, 1, 2, ... instead
    labels = ds.graph.ndata[LABEL_FIELD]
    for c in range(3):
        n_c = (calib & (labels == c)).sum().item()
        assert (tune & (labels == c)).sum().item() == int(0.5 * n_c)
