"""Flower federation: messages, client state, FedAvg and heterogeneous KD.

Tests marked ``slow`` launch real Flower simulations (Ray) on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from heterofl.experiments.common import RunContext
from heterofl.experiments.federated_common import build_job
from heterofl.federation import messages as M
from heterofl.federation.simulation import launch_subprocess, space_free_python
from heterofl.federation.state import ClientStateError, ClientStateStore
from heterofl.models.model_factory import create_model
from heterofl.models.mlp import MLPModel
from heterofl.utils.io import read_json

from .test_models import MODELS


# --------------------------------------------------------------------------- unit
def test_probability_payload_roundtrip():
    p = np.random.default_rng(0).random(1000)
    rec = M.probs_to_record(p)
    np.testing.assert_allclose(M.record_to_probs(rec), p, atol=1e-7)  # float32 on the wire
    assert M.payload_bytes(RecordDict({"x": rec})) >= 4000


def test_fedavg_aggregation_is_example_weighted():
    def reply(v, n):
        return RecordDict({"arrays": ArrayRecord({"w": Array(np.full(3, v, dtype=np.float32))}),
                           "metrics": MetricRecord({"num-examples": n})})
    agg = aggregate_arrayrecords([reply(1.0, 100), reply(4.0, 300)], "num-examples")
    np.testing.assert_allclose(agg["w"].numpy(), np.full(3, (1 * 100 + 4 * 300) / 400))


def _ctx() -> Context:
    return Context(run_id=1, node_id=7, node_config={"partition-id": 0}, state=RecordDict(),
                   run_config={})


def test_client_state_persists_and_is_verified(tmp_path, toy_xy):
    X, y = toy_xy
    m = create_model("random_forest", MODELS, 1)
    m.fit(X, y)
    ctx = _ctx()
    store = ClientStateStore(tmp_path, "B")
    path, digest = store.save_round(m, 0, val_probs=m.predict_proba(X), proxy_probs=np.zeros(3),
                                    teacher=None, info={})
    ClientStateStore.record(ctx, bank="B", rnd=0, path=path, digest=digest, model_type="random_forest")
    loaded = store.load_previous(ctx, expected_round=0, model_type="random_forest",
                                 models_cfg=MODELS, seed=1, n_jobs=1)
    np.testing.assert_array_equal(loaded.predict_proba(X), m.predict_proba(X))
    with pytest.raises(ClientStateError, match="round"):
        store.load_previous(ctx, expected_round=1, model_type="random_forest", models_cfg=MODELS,
                            seed=1, n_jobs=1)
    with pytest.raises(ClientStateError, match="no client state"):
        store.load_previous(_ctx(), expected_round=0, model_type="random_forest",
                            models_cfg=MODELS, seed=1, n_jobs=1)
    with pytest.raises(ClientStateError, match="belongs to bank"):
        ClientStateStore(tmp_path, "A").load_previous(ctx, expected_round=0, model_type="random_forest",
                                                      models_cfg=MODELS, seed=1, n_jobs=1)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ClientStateError, match="hash"):
        store.load_previous(ctx, expected_round=0, model_type="random_forest", models_cfg=MODELS,
                            seed=1, n_jobs=1)


def test_space_free_interpreter_path_is_usable():
    import subprocess

    exe = space_free_python()
    assert not any(c.isspace() for c in exe)
    out = subprocess.run([exe, "-c", "import heterofl, flwr; print('ok')"], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "ok"


# --------------------------------------------------------------------------- simulations
def _run_ctx(prepared, tmp_path, method, extra_cfg=None) -> RunContext:
    cfg, _, data = prepared
    cfg = {**cfg, **(extra_cfg or {})}
    out = tmp_path / method
    out.mkdir()
    return RunContext(cfg=cfg, seed=42, scenario="moderate_non_iid", method=method, data=data,
                      out_dir=out, run_id="test")


@pytest.mark.slow
def test_fedavg_simulation_aggregates_parameters(prepared, tmp_path):
    ctx = _run_ctx(prepared, tmp_path, "fedavg")
    job = build_job(ctx, "fedavg")
    launch_subprocess(job.path, ctx.out_dir / "federation.log", timeout=600)
    server = ctx.out_dir / "federation" / "server"
    status = read_json(server / "status.json")
    assert status["status"] == "completed"
    assert sorted(status["node_to_bank"].values()) == ["A", "B", "C"]  # three distinct clients
    summ = read_json(server / "rounds.json")
    rounds = {r["round"]: r for r in summ["rounds"]}
    assert set(rounds) == {0, 1, 2}
    for r in (1, 2):
        cm = rounds[r]["client_metrics"]
        assert set(cm) == {"A", "B", "C"}
        n = {b: cm[b]["num-examples"] for b in cm}
        for b in cm:  # weights are the example-count proportions
            assert rounds[r]["weights"][b] == pytest.approx(n[b] / sum(n.values()))
        assert rounds[r]["bytes_up"] > 0 and rounds[r]["bytes_down"] > 0
        _, y = ctx.data.bank("A")
        assert n["A"] == len(y)
    # the global model actually changes every round
    states = [MLPModel.load(server / f"global_round_{r:02d}" / "model.pt").get_state_dict()
              for r in (0, 1, 2)]
    for a, b in zip(states, states[1:], strict=False):
        assert any(not np.allclose(a[k], b[k]) for k in a)


@pytest.mark.slow
def test_kd_simulation_exchanges_predictions_with_correct_teachers(prepared, tmp_path):
    ctx = _run_ctx(prepared, tmp_path, "heterogeneous_kd")
    job = build_job(ctx, "kd")
    launch_subprocess(job.path, ctx.out_dir / "federation.log", timeout=600)
    jd = ctx.out_dir / "federation"
    assert read_json(jd / "server" / "status.json")["status"] == "completed"
    summ = read_json(jd / "server" / "rounds.json")
    n_proxy = summ["n_proxy"]
    assert n_proxy == len(ctx.data.proxy_features()[1])
    assert summ["aggregation"].startswith("per-example proxy probabilities")
    types = {}
    for bank in "ABC":
        for r in range(job.num_rounds + 1):
            meta = read_json(jd / "clients" / f"bank_{bank}" / f"round_{r:02d}" / "meta.json")
            assert meta["round"] == r and meta["bank"] == bank
            types[bank] = meta["model_type"]
    assert types == {"A": "xgboost", "B": "random_forest", "C": "mlp"}
    # every teacher a client received equals the leave-one-out mean of the OTHER clients'
    # proxy predictions from the previous round (float32 on the wire)
    for r in range(1, job.num_rounds + 1):
        prev = {b: np.load(jd / "clients" / f"bank_{b}" / f"round_{r - 1:02d}" / "proxy_probs.npy")
                for b in "ABC"}
        for b in "ABC":
            teacher = np.load(jd / "clients" / f"bank_{b}" / f"round_{r:02d}" / "teacher.npy")
            expected = np.mean([prev[o] for o in "ABC" if o != b], axis=0)
            np.testing.assert_allclose(teacher, expected, atol=1e-6)
            assert teacher.shape == (n_proxy,)
    # communication = prediction vectors only: float32 * n_proxy per client and direction
    r1 = summ["rounds"][1]
    assert 3 * 4 * n_proxy <= r1["bytes_up"] < 3 * 4 * n_proxy + 3 * 1024
    assert 3 * 4 * n_proxy <= r1["bytes_down"] < 3 * 4 * n_proxy + 3 * 1024
    assert summ["rounds"][0]["bytes_down"] == 0


@pytest.mark.slow
def test_round_failure_is_detected(prepared, tmp_path):
    ctx = _run_ctx(prepared, tmp_path, "heterogeneous_kd",
                   {"debug": {"fail": {"partition_id": 1, "round": 1}}})
    job = build_job(ctx, "kd")
    with pytest.raises(RuntimeError, match="failed"):
        launch_subprocess(job.path, ctx.out_dir / "federation.log", timeout=600)
    status = read_json(ctx.out_dir / "federation" / "server" / "status.json")
    assert status["status"] == "failed"
    assert "bank B failed" in status["error"]


# --------------------------------------------------------------------------- identification
def _job(tmp_path, n_clients=3):
    from heterofl.federation.job import ClientSpec, FederatedJob

    job = FederatedJob(mode="kd", job_dir=str(tmp_path), view_dir=str(tmp_path), seed=1,
                       num_rounds=1, models={"mlp": {"torch_threads": 1}}, fedavg={}, kd={},
                       expected_proxy_sha256="x",
                       clients=[ClientSpec(i, b, m) for i, (b, m) in enumerate(
                           [("A", "xgboost"), ("B", "random_forest"), ("C", "mlp")][:n_clients])],
                       node_wait_timeout=1.0, round_timeout=1.0)
    job.save()
    return job


class _Reply:
    def __init__(self, node, info=None, error=None):
        from types import SimpleNamespace

        self.metadata = SimpleNamespace(src_node_id=node)
        self.content = {M.INFO: info} if info is not None else {}
        self.error = error

    def has_error(self):
        return self.error is not None


@pytest.fixture()
def stub_message(monkeypatch):
    """flwr.app.Message needs a live Flower run context; the strategy only sends it."""
    import heterofl.federation.heterogeneous_strategy as strategy

    monkeypatch.setattr(strategy, "Message", lambda **kw: kw)


class _Grid:
    def __init__(self, replies):
        self.replies = replies

    def get_node_ids(self):
        return [r.metadata.src_node_id for r in self.replies]

    def send_and_receive(self, msgs, timeout=None):
        assert len(msgs) == len(self.replies)
        return self.replies


def _info(job, pid, node):
    from heterofl.federation.client_app import identify_info

    return identify_info(job, pid, node)


def test_identify_reply_marks_unassigned_nodes(tmp_path):
    job = _job(tmp_path)
    assert _info(job, 1, 11) == {"assigned": True, "partition-id": 1, "bank": "B",
                                 "model-type": "random_forest", "mode": "kd", "node-id": 11}
    assert _info(job, 7, 17) == {"assigned": False, "partition-id": 7, "mode": "kd",
                                 "node-id": 17}


def test_identify_clients_ignores_only_explicitly_unassigned_nodes(tmp_path, stub_message):
    from heterofl.federation.heterogeneous_strategy import identify_clients

    job = _job(tmp_path)
    replies = [_Reply(10 + p, _info(job, p, 10 + p)) for p in range(4)]   # node 13: no bank
    mapping = identify_clients(_Grid(replies), job)
    assert {n: s.bank for n, s in mapping.items()} == {10: "A", 11: "B", 12: "C"}


def test_identify_errors_are_never_swallowed(tmp_path, stub_message):
    """The old code skipped any error whose text contained 'No client is assigned'."""
    from types import SimpleNamespace

    from heterofl.federation.heterogeneous_strategy import identify_clients
    from heterofl.federation.job import FederationError

    job = _job(tmp_path)
    good = [_Reply(10 + p, _info(job, p, 10 + p)) for p in range(3)]
    err = SimpleNamespace(code=2, reason="FederationError: No client is assigned to partition-id 3")
    with pytest.raises(FederationError, match="identify failed on node 13"):
        identify_clients(_Grid([*good, _Reply(13, error=err)]), job)
    legacy = {k: v for k, v in _info(job, 0, 10).items() if k != "assigned"}
    with pytest.raises(FederationError, match="no 'assigned' field"):
        identify_clients(_Grid([_Reply(10, legacy), *good[1:]]), job)
    with pytest.raises(FederationError, match="Two nodes claim bank A"):
        identify_clients(_Grid([*good, _Reply(19, _info(job, 0, 19))]), job)
    lying = {**_info(job, 9, 10), "partition-id": 0}
    with pytest.raises(FederationError, match="unassigned, but the job assigns it"):
        identify_clients(_Grid([_Reply(10, lying), *good[1:]]), job)
    with pytest.raises(FederationError, match="Identified banks"):
        identify_clients(_Grid(good[:2] + [_Reply(13, _info(job, 5, 13))]), job)


# --------------------------------------------------------------------------- trusted paths
def test_job_path_is_validated(tmp_path, monkeypatch):
    from heterofl.federation.job import FederationError, check_job_path

    job = _job(tmp_path)
    assert check_job_path(str(job.path)) == job.path.resolve()
    with pytest.raises(FederationError, match="absolute"):
        check_job_path("relative/job.json")
    with pytest.raises(FederationError, match="absolute"):
        check_job_path(str(tmp_path / "other.json"))
    with pytest.raises(FederationError, match="does not exist"):
        check_job_path(str(tmp_path / "missing" / "job.json"))
    monkeypatch.setenv("HETEROFL_JOB_ROOT", str(tmp_path / "elsewhere"))
    with pytest.raises(FederationError, match="outside HETEROFL_JOB_ROOT"):
        check_job_path(str(job.path))


def test_client_state_refuses_checkpoints_outside_its_directory(tmp_path, toy_xy):
    X, y = toy_xy
    m = create_model("xgboost", MODELS, 1)
    m.fit(X, y)
    foreign = m.save(tmp_path / "elsewhere" / "model")
    ctx = _ctx()
    ClientStateStore.record(ctx, bank="A", rnd=0, path=foreign,
                            digest=__import__("heterofl.utils.reproducibility",
                                              fromlist=["x"]).sha256_file(foreign),
                            model_type="xgboost")
    with pytest.raises(ClientStateError, match="outside this bank's state directory"):
        ClientStateStore(tmp_path / "job", "A").load_previous(
            ctx, expected_round=0, model_type="xgboost", models_cfg=MODELS, seed=1, n_jobs=1)


@pytest.mark.slow
def test_kd_simulation_with_an_extra_unassigned_node(prepared, tmp_path):
    ctx = _run_ctx(prepared, tmp_path, "heterogeneous_kd")
    job = build_job(ctx, "kd")
    launch_subprocess(job.path, ctx.out_dir / "federation.log", timeout=600, num_supernodes=4)
    status = read_json(ctx.out_dir / "federation" / "server" / "status.json")
    assert status["status"] == "completed"
    assert sorted(status["node_to_bank"].values()) == ["A", "B", "C"]
    assert "Ignoring 1 node(s) without a bank" in (ctx.out_dir / "federation.log").read_text()
