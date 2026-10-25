"""Homogeneous baselines: committee composition, federated XGBoost bagging, profiles.

Slow tests run the real runner (Flower KD, Ray) on SYNTHETIC, NON-PUBLISHABLE data.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from heterofl.experiments.common import committee_composition
from heterofl.experiments.runner import main
from heterofl.federation.xgb_bagging import FedXgbBagging, logit, xgb_native_params
from heterofl.utils.config import ConfigError, config_diff, load_config, semantic_config
from heterofl.utils.io import load_npz, read_json
from heterofl.utils.reproducibility import source_patch

from .conftest import ROOT, TINY_MODELS

FAMILIES = ("xgboost", "random_forest", "mlp")
PROFILES = {"xgboost": "homog_xgboost", "random_forest": "homog_random_forest",
            "mlp": "homog_mlp"}
XGB = {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3, "subsample": 1.0,
       "colsample_bytree": 1.0, "tree_method": "hist"}


# --------------------------------------------------------------------------- labels
def test_committee_composition_labels():
    het = [{"bank": "A", "model": "xgboost"}, {"bank": "B", "model": "random_forest"},
           {"bank": "C", "model": "mlp"}]
    assert committee_composition(het) == "Heterogeneous"
    homog = [{"bank": b, "model": "random_forest"} for b in "ABC"]
    assert committee_composition(homog) == "Homogeneous 3xRandom Forest"


# --------------------------------------------------------------------------- profiles
@pytest.mark.parametrize("family", FAMILIES)
def test_homogeneous_profiles_differ_from_paper_only_in_clients(family):
    paper = load_config(ROOT / "configs" / "paper.yaml")
    cfg = load_config(ROOT / "configs" / f"{PROFILES[family]}.yaml")
    assert [c["model"] for c in cfg["clients"]] == [family] * 3
    assert [c["bank"] for c in cfg["clients"]] == [c["bank"] for c in paper["clients"]]
    # data, splits, partitioning, models, rounds, KD and evaluation are inherited unchanged
    diff = set(config_diff(semantic_config(paper), semantic_config(cfg)))
    assert diff <= {"clients", "profile", "fedxgb"}, diff
    assert cfg["results_dir"] != paper["results_dir"]
    assert set(cfg["scenarios"]) == set(paper["scenarios"])


def test_fedavg_is_an_allowed_ablation_method():
    cfg = load_config(ROOT / "configs" / "homog_mlp.yaml")
    assert cfg["ablations"]["runs"]["fedavg_20_epochs"]["method"] == "fedavg"
    assert set(cfg["ablations"]["runs"]) == {"kd_lambda_0.0", "fedavg_20_epochs"}


def test_fedxgb_requires_trees_per_round():
    with pytest.raises(ConfigError, match="fedxgb.trees_per_round"):
        load_config(ROOT / "configs" / "homog_xgboost.yaml", ["fedxgb.trees_per_round=0"])


# --------------------------------------------------------------------------- bagging
def _banks(toy_xy):
    X, y = toy_xy
    return {"A": (X[:500], y[:500]), "B": (X[500:900], y[500:900]), "C": (X[900:], y[900:])}


def test_bagging_with_one_bank_equals_plain_boosting(toy_xy):
    X, y = toy_xy
    fx = FedXgbBagging({"A": (X, y)}, XGB, seed=0, n_jobs=1, num_rounds=4, trees_per_round=1)
    *_, (_, p, _) = fx.rounds({"x": X[:300]})
    b0 = logit(y.mean())
    ref = xgb.train(xgb_native_params(XGB, 0, 1),
                    xgb.DMatrix(X, label=y, base_margin=np.full(len(y), b0)), num_boost_round=4)
    expect = ref.predict(xgb.DMatrix(X[:300], base_margin=np.full(300, b0)))
    np.testing.assert_allclose(p["x"], expect, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fx.predict_proba(X[:300]), expect, rtol=1e-5, atol=1e-6)


def test_bagging_appends_trees_and_accounts_communication(toy_xy, tmp_path):
    fx = FedXgbBagging(_banks(toy_xy), XGB, seed=1, n_jobs=1, num_rounds=3, trees_per_round=2)
    X = toy_xy[0][:200]
    out = list(fx.rounds({"x": X}))
    assert [info["n_trees"] for *_, info in out] == [6, 12, 18]
    assert all(len(bst.get_dump()) == 2 for rnd in fx.boosters for bst in rnd)
    # the global model of round r is the incremental score after round r
    for r, p, _ in out:
        np.testing.assert_allclose(fx.predict_proba(X, upto_round=r), p["x"], atol=1e-9)
    ups = [info["bytes_up"] for *_, info in out]
    downs = [info["bytes_down"] for *_, info in out]
    assert downs[0] == 0 and downs[1] == 3 * ups[0] and downs[2] == 3 * (ups[0] + ups[1])
    meta = read_json(fx.save(tmp_path / "m", upto_round=2) / "meta.json")
    assert meta["rounds"] == 2 and len(meta["boosters_in_order"]) == 6
    assert meta["b0"] == pytest.approx(fx.b0)


def test_bagging_bank_update_uses_only_its_own_rows(toy_xy):
    """Changing bank C's rows cannot change the trees banks A and B fit in round 1."""
    banks = _banks(toy_xy)
    a = FedXgbBagging(banks, XGB, seed=2, n_jobs=1, num_rounds=1, trees_per_round=2)
    list(a.rounds())
    Xc, yc = banks["C"]
    other = {**banks, "C": (Xc[::-1].copy() * 1.5, yc[::-1].copy())}
    b2 = FedXgbBagging(other, XGB, seed=2, n_jobs=1, num_rounds=1, trees_per_round=2)
    b2.b0 = a.b0  # isolate the per-bank update from the pooled-rate intercept
    list(b2.rounds())
    for i in (0, 1):
        assert a.boosters[0][i].get_dump() == b2.boosters[0][i].get_dump()
    assert a.boosters[0][2].get_dump() != b2.boosters[0][2].get_dump()


# --------------------------------------------------------------------------- provenance
def test_source_patch_includes_untracked_files(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "tracked.py").write_text("x = 1\n")
    git("add", "tracked.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (tmp_path / "tracked.py").write_text("x = 2\n")
    (tmp_path / "new_module.py").write_text("y = 3\n")
    patch = source_patch(tmp_path).decode()
    assert "+x = 2" in patch and "new_module.py" in patch and "+y = 3" in patch


# --------------------------------------------------------------------------- end to end
def _run_args(tmp_path, csv, results, family, methods, extra=()):
    clients = ",".join(f"{{bank: {b}, model: {family}}}" for b in "ABC")
    sets = [f"data.path={csv}", f"data.processed_dir={tmp_path / 'processed'}",
            f"data.partitions_dir={tmp_path / 'partitions'}", f"results_dir={results}",
            "partitioning.min_fraud_per_client=5", "partitioning.min_legit_per_client=100",
            "federation.client_num_cpus=1", "federation.job_timeout=900",
            "evaluation.bootstrap.n_resamples=50", "fedavg.local_epochs=1",
            "ablations.enabled=false", f"clients=[{clients}]", *TINY_MODELS, *extra]
    args = ["--config", str(ROOT / "configs" / "smoke_synthetic.yaml"), "--methods", *methods]
    for s in sets:
        args += ["--set", s]
    return args


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_homogeneous_committee_end_to_end(tmp_path, synthetic_csv, family):
    methods = ["local", "ensemble", "heterogeneous_kd"]
    extra = []
    if family == "xgboost":
        methods.append("fedxgb_bagging")
        extra = ["methods=[local, ensemble, heterogeneous_kd, fedxgb_bagging]",
                 "fedxgb.trees_per_round=3"]
    res = tmp_path / "results"
    assert main(_run_args(tmp_path, synthetic_csv, res, family, methods, extra)
                + ["--no-report"]) == 0
    sdir = res / "seed_42" / "moderate_non_iid"
    metrics = pd.concat(pd.read_csv(sdir / m / "metrics.csv") for m in methods)
    assert metrics["partition_sha256"].nunique() == 1

    # homogeneous members: same family, but different seeds and data -> different models
    local = {b: load_npz(sdir / "local" / "predictions" / f"bank_{b}.npz")["p_val"] for b in "ABC"}
    assert not np.allclose(local["A"], local["B"]) and not np.allclose(local["B"], local["C"])
    assert set(metrics.loc[metrics["system"].str.startswith("bank_"), "model_type"]) == {family}
    label = metrics.loc[metrics["system"] == "kd_committee", "system_label"].item()
    assert label.startswith("Homogeneous 3x")
    assert metrics.loc[metrics["system"] == "ensemble_equal", "system_label"].item().startswith(
        "Homogeneous 3x")

    kd = sdir / "heterogeneous_kd"
    diag = read_json(kd / "diagnostics.json")
    for b in "ABC":  # KD round 0 = the local-only (E) members
        assert diag["round0_vs_local_only"][b]["max_abs_diff_val_probs_vs_local"] < 1e-6
    # proxy predictions: one per proxy row, ordered; teachers are leave-one-out means of
    # the previous round's (float32-on-the-wire) predictions; students changed
    clients = kd / "federation" / "clients"
    n_proxy = read_json(kd / "federation" / "server" / "rounds.json")["n_proxy"]
    for r in (1, 2):
        prev = {b: np.load(clients / f"bank_{b}" / f"round_{r - 1:02d}" / "proxy_probs.npy")
                for b in "ABC"}
        for b in "ABC":
            t = np.load(clients / f"bank_{b}" / f"round_{r:02d}" / "teacher.npy")
            assert t.shape == (n_proxy,)
            others = [prev[o] for o in "ABC" if o != b]
            np.testing.assert_allclose(t, np.mean(others, axis=0), atol=1e-6)
    p0 = np.load(clients / "bank_C" / "round_00" / "proxy_probs.npy")
    p2 = np.load(clients / "bank_C" / "round_02" / "proxy_probs.npy")
    assert p0.shape == (n_proxy,) and not np.allclose(p0, p2)

    if family == "xgboost":
        fx = read_json(sdir / "fedxgb_bagging" / "runtime.json")["fedxgb_bagging"]
        assert fx["n_trees_global"] == 2 * 3 * 3   # 2 smoke rounds x 3 banks x 3 trees
        assert (sdir / "fedxgb_bagging" / "models" / "fedxgb_global" / "meta.json").is_file()
        assert (sdir / "fedxgb_bagging" / "rounds.csv").is_file()

    # the per-profile report works for homogeneous profiles
    from heterofl.evaluation.report import generate_report

    rdir = generate_report(res)  # reads the configuration the runner stored
    assert "homogeneous 3x" in (res / "tables" / "table5_per_bank.tex").read_text()
    assert (rdir / "summary.md").is_file()


# --------------------------------------------------------------------------- comparison
@pytest.mark.slow
def test_cross_profile_comparison_pairs_on_identical_partitions(tmp_path, synthetic_csv):
    from heterofl.evaluation.committee_comparison import ComparisonError, compare

    def run(name, family, methods, extra=()):
        res = tmp_path / name
        if family == "hetero":
            args = _run_args(tmp_path, synthetic_csv, res, "xgboost", methods, extra)
            i = args.index("clients=[{bank: A, model: xgboost},{bank: B, model: xgboost},"
                           "{bank: C, model: xgboost}]")
            args[i] = ("clients=[{bank: A, model: xgboost},{bank: B, model: random_forest},"
                       "{bank: C, model: mlp}]")
        else:
            args = _run_args(tmp_path, synthetic_csv, res, family, methods, extra)
        assert main(args + ["--no-report"]) == 0
        return res

    het = run("het", "hetero", ["local", "ensemble", "fedavg", "heterogeneous_kd"])
    xgb_dir = run("xgb", "xgboost", ["local", "ensemble", "heterogeneous_kd", "fedxgb_bagging"],
                  ["methods=[local, ensemble, heterogeneous_kd, fedxgb_bagging]",
                   "fedxgb.trees_per_round=3"])
    summary = compare({"hetero": het, "xgboost": xgb_dir}, tmp_path / "cmp")
    text = summary.read_text()
    assert "Identical partitions and test labels" in text
    pairs = pd.read_csv(tmp_path / "cmp" / "reports" / "paired_differences.csv")
    got = set(zip(pairs["a"], pairs["b"], strict=True))
    assert {("kd_het", "fedavg"), ("kd_het", "kd_xgb"), ("kd_het", "fedxgb"),
            ("ens_het", "ens_xgb"), ("kd_xgb", "ens_xgb"), ("kd_het", "ens_het")} <= got
    per_seed = pd.read_csv(tmp_path / "cmp" / "reports" / "per_seed_metrics.csv")
    a = per_seed[per_seed["key"] == "kd_het"]["test_average_precision"].item()
    b = per_seed[per_seed["key"] == "kd_xgb"]["test_average_precision"].item()
    row = pairs[(pairs["a"] == "kd_het") & (pairs["b"] == "kd_xgb")]
    assert row["mean_diff"].item() == pytest.approx(a - b)
    for name in ("comparison_main", "comparison_paired", "comparison_mechanism",
                 "comparison_cost", "comparison_diversity", "comparison_per_bank"):
        assert (tmp_path / "cmp" / "tables" / f"{name}.tex").is_file(), name
    assert (tmp_path / "cmp" / "figures" / "fig_c1_ap_by_system.pdf").is_file()
    assert (tmp_path / "cmp" / "figures" / "fig_c2_paired_differences.pdf").is_file()

    # a profile that differs in anything but the client families is refused
    other = run("other", "xgboost", ["local", "ensemble"], ["kd.lambda_kd=0.25"])
    with pytest.raises(ComparisonError, match="kd.lambda_kd"):
        compare({"hetero": het, "xgboost": other}, tmp_path / "cmp2")
