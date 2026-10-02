"""Transport/TAR mathematics on synthetic fixtures; no network or real data."""
import gzip
import io
import tarfile

import pytest

from cognix.adapters.carla.train_acquisition import (
    AcquisitionError, GzipTrainStream, TrainTarStream, train_member_path,
    validate_range_response,
)


def archive(entries, fmt=tarfile.USTAR_FORMAT):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=fmt) as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return out.getvalue()


def parser(tmp_path, target=2, reuse=None):
    boundaries = []
    def boundary(record, evidence):
        boundaries.append((record["scenario_id"], evidence))
    return TrainTarStream(tmp_path, reuse or {}, boundary, target), boundaries


@pytest.mark.parametrize("chunk_size", [1, 7, 511, 777, 1 << 20])
def test_single_decoder_survives_partial_gzip_tar_states(tmp_path, chunk_size):
    raw = archive([(f"train/Town01/scenario-{s}/rgb-front/000000.jpg", bytes([s]) * 1301)
                   for s in [1, 10, 11]])
    compressed = gzip.compress(raw)
    p, boundaries = parser(tmp_path)
    d = GzipTrainStream(p)
    # Same decoder before/after a hypothetical staged-prefix boundary.
    split = len(compressed) // 2
    for block in [compressed[:split], compressed[split:]]:
        for i in range(0, len(block), chunk_size):
            d.feed(block[i:i+chunk_size])
            if p.done:
                break
        if p.done:
            break
    assert p.done
    assert [s[0] for s in boundaries] == ["Town01/scenario-1", "Town01/scenario-10"]
    assert p.trailing_boundary["next_scenario_id"] == "Town01/scenario-11"
    assert not (tmp_path / "train/Town01/scenario-11").exists()
    assert (tmp_path / "train/Town01/scenario-10/rgb-front/000000.jpg").read_bytes() == bytes([10]) * 1301


def test_partial_payload_is_never_counted_complete(tmp_path):
    raw = archive([("train/Town01/scenario-1/rgb-front/000000.jpg", b"x" * 1000)])
    p, boundaries = parser(tmp_path)
    p.feed(raw[:1000])
    assert not boundaries and not p.current["source_files"]
    assert not p.done and p.state == "payload"
    p.close()


def test_complete_payload_requires_next_scenario_or_valid_gzip_end(tmp_path):
    raw = archive([("train/Town01/scenario-1/rgb-front/000000.jpg", b"a")])
    p, boundaries = parser(tmp_path, target=1)
    p.feed(raw[:1024])
    assert not boundaries and not p.done
    d = GzipTrainStream(p)
    # Start a separate parser/decoder for the full termination case.
    p.close()
    p, boundaries = parser(tmp_path, target=1)
    GzipTrainStream(p).feed(gzip.compress(raw))
    assert p.done and boundaries[0][1]["kind"] == "valid_gzip_and_tar_termination"


@pytest.mark.parametrize("path", ["test/normal/Town01/scenario-1/a", "../train/a",
                                  "/train/a", "train/Town01/../escape", "train\\bad"])
def test_nontrain_and_traversal_paths_rejected(path):
    with pytest.raises(AcquisitionError):
        train_member_path(path)


def test_archive_links_and_checksum_errors_rejected(tmp_path):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        info = tarfile.TarInfo("train/Town01/scenario-1/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "outside"
        tar.addfile(info)
    p, _ = parser(tmp_path)
    with pytest.raises(AcquisitionError, match="links"):
        p.feed(out.getvalue())
    p, _ = parser(tmp_path)
    raw = bytearray(archive([("train/Town01/scenario-1/a", b"data")]))
    raw[0] ^= 1
    with pytest.raises(AcquisitionError, match="Bad TAR header"):
        p.feed(raw)


def test_reused_scenario_is_verified_without_writes(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    f = original / "a"
    f.write_bytes(b"preserved")
    before = f.stat().st_mtime_ns
    p, _ = parser(tmp_path / "new", target=1, reuse={"Town01/scenario-1": original})
    GzipTrainStream(p).feed(gzip.compress(archive([("train/Town01/scenario-1/a", b"preserved")])))
    assert p.done and f.stat().st_mtime_ns == before
    assert f.read_bytes() == b"preserved"
    assert not (tmp_path / "new/train/Town01/scenario-1").exists()


def test_existing_staged_mismatch_fails_without_overwrite(tmp_path):
    f = tmp_path / "train/Town01/scenario-1/a"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"wrong")
    p, _ = parser(tmp_path, target=1)
    with pytest.raises(AcquisitionError, match="content mismatch"):
        p.feed(archive([("train/Town01/scenario-1/a", b"right")]))
    assert f.read_bytes() == b"wrong"


@pytest.mark.parametrize("fmt", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_long_names_are_parsed_before_safe_extraction(tmp_path, fmt):
    name = "train/Town01/scenario-1/" + "x" * 130
    p, _ = parser(tmp_path, target=1)
    GzipTrainStream(p).feed(gzip.compress(archive([(name, b"extension")], fmt)))
    assert p.done and (tmp_path / name).read_bytes() == b"extension"


@pytest.mark.parametrize("status,content_range", [(200,"bytes 12-19/20"),
    (206,"bytes 0-19/20"), (206,"bytes 12-20/20"), (206,"invalid")])
def test_bad_range_response_never_authorizes_append(status, content_range):
    with pytest.raises(AcquisitionError):
        validate_range_response(status, {"Content-Range": content_range}, 12)


def test_checked_range_acceptance_and_size_identity():
    assert validate_range_response(206, {"Content-Range": "bytes 12-19/20", "Content-Length": "8"},12) == (19,20)
    with pytest.raises(AcquisitionError, match="total changed"):
        validate_range_response(206,{"Content-Range":"bytes 12-19/20"},12,21)
    with pytest.raises(AcquisitionError, match="encoding"):
        validate_range_response(206,{"Content-Range":"bytes 12-19/20","Content-Encoding":"gzip"},12)


def test_corrupt_gzip_trailer_is_not_valid_archive_termination(tmp_path):
    data = bytearray(gzip.compress(archive([("train/Town01/scenario-1/a", b"x")])))
    data[-8] ^= 1
    p, boundaries = parser(tmp_path, target=1)
    with pytest.raises(AcquisitionError, match="Gzip stream corrupt"):
        GzipTrainStream(p).feed(data)
    assert not boundaries and not p.done
