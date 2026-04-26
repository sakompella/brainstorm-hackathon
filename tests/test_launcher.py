"""Tests for scripts/launcher.py service launcher."""

from __future__ import annotations

from scripts.launcher import parse_args


def test_parse_args_defaults_to_hard_dataset() -> None:
    args = parse_args(())

    assert args.data_dir == "data/hard"


def test_parse_args_preserves_explicit_dataset_override() -> None:
    args = parse_args(("data/super_easy",))

    assert args.data_dir == "data/super_easy"
