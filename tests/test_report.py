import json

from chronospect import analyze
from chronospect.synthetic import aging_process, multi_timescale, single_timescale


def test_multi_verdict_multi_stationary():
    r = analyze(multi_timescale(T=2048, timescales=(5.0, 100.0), n_traj=8, seed=7))
    assert "multi-timescale" in r.verdict
    assert "aging" not in r.verdict  # stationary
    assert r.n_dominant_timescales >= 2


def test_single_verdict_single_stationary():
    r = analyze(single_timescale(T=2048, timescale=20.0, n_traj=8, seed=7))
    assert "single-speed" in r.verdict
    assert "aging" not in r.verdict


def test_aging_verdict():
    r = analyze(aging_process(T=2048, n_traj=8, seed=7))
    assert "aging" in r.verdict
    assert r.aging_index >= 0.3


def test_report_is_json_serializable():
    r = analyze(single_timescale(T=1024, timescale=12.0, n_traj=4, seed=1))
    s = json.dumps(r.to_dict())
    assert "verdict" in json.loads(s)
