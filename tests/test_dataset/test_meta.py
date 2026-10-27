"""Tests for scalar graph-level target interop."""

from __future__ import annotations

import numpy as np
import pytest
from molpy import Frame

from molhub.dataset.meta import Targets


class TestTargetsWrite:
    def test_roundtrip(self):
        f = Frame()
        Targets(f).write({"energy": -40.5})
        assert Targets(f).read()["energy"] == pytest.approx(-40.5)

    def test_mixed_types_in_one_call(self):
        f = Frame()
        values = {"e": -1.5, "n": 2, "ok": False, "s": "x"}
        Targets(f).write(values)
        assert Targets(f).read() == values

    def test_bool_is_not_widened_to_int(self):
        f = Frame()
        Targets(f).write({"flag": True})
        assert Targets(f)["flag"] is True
        assert f.meta.dtype("flag") == "bool"

    def test_string_infers_string_dtype(self):
        f = Frame()
        Targets(f).write({"tag": "gdb9"})
        assert f.meta.dtype("tag") == "string"

    def test_numpy_scalars_are_reduced(self):
        f = Frame()
        Targets(f).write({"f": np.float64(2.5), "i": np.int64(3), "b": np.bool_(True)})
        assert Targets(f).read() == {"f": 2.5, "i": 3, "b": True}

    def test_replaces_rather_than_merges(self):
        f = Frame()
        Targets(f).write({"a": 1.0, "b": 2.0})
        Targets(f).write({"c": 3.0})
        assert Targets(f).read() == {"c": 3.0}

    def test_clearing_via_empty_mapping(self):
        f = Frame()
        Targets(f).write({"a": 1.0})
        Targets(f).write({})
        assert Targets(f).read() == {}

    def test_live_meta_mutations_are_visible(self):
        f = Frame()
        targets = Targets(f)
        targets.write({"a": 1.0})
        live = f.meta
        live["b"] = 2.0
        assert targets.read() == {"a": 1.0, "b": 2.0}
        live.update({"a": 3.0})
        assert targets["a"] == 3.0
        live.clear()
        assert targets.read() == {}

    def test_frames_do_not_share_metadata(self):
        f, g = Frame(), Frame()
        Targets(f).write({"a": 1.0})
        Targets(g).write({"a": 2.0})
        assert Targets(f).read()["a"] == pytest.approx(1.0)
        assert Targets(g).read()["a"] == pytest.approx(2.0)

    def test_copy_isolates_metadata(self):
        f = Frame()
        Targets(f).write({"a": 1.0})
        clone = f.copy()
        Targets(clone).write({"a": 9.0})
        assert Targets(f).read()["a"] == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [None, [1, 2], {"a": 1}, (1, 2), object()])
    def test_rejects_unsupported_value(self, bad):
        with pytest.raises(TypeError, match="unsupported type"):
            Targets(Frame()).write({"bad": bad})

    def test_error_names_the_offending_key(self):
        with pytest.raises(TypeError, match="'culprit'"):
            Targets(Frame()).write({"fine": 1.0, "culprit": None})


class TestTargetsMapping:
    @pytest.fixture
    def frame(self):
        f = Frame()
        Targets(f).write({"energy": -40.5, "n": 3})
        return f

    def test_getitem(self, frame):
        assert Targets(frame)["energy"] == pytest.approx(-40.5)

    def test_getitem_missing_key_raises(self, frame):
        with pytest.raises(KeyError):
            Targets(frame)["nope"]

    def test_contains(self, frame):
        t = Targets(frame)
        assert "energy" in t
        assert "nope" not in t

    def test_len(self, frame):
        assert len(Targets(frame)) == 2

    def test_iter_and_keys_agree(self, frame):
        t = Targets(frame)
        assert sorted(t) == sorted(t.keys()) == ["energy", "n"]

    def test_empty_frame(self):
        t = Targets(Frame())
        assert len(t) == 0
        assert t.read() == {}
        assert t.keys() == []
