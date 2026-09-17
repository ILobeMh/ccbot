"""Tests for utils.process_tree_rss (ps output parsing)."""

import subprocess

import ccbot.utils as u


def test_sums_descendants(monkeypatch):
    ps = "1 0 100\n10 1 200\n11 10 300\n12 10 400\n20 1 500\n"
    monkeypatch.setattr(
        u.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=ps, stderr=""),
    )
    totals = u.process_tree_rss([10, 20, 99])
    assert totals == {10: (200 + 300 + 400) * 1024, 20: 500 * 1024, 99: 0}
