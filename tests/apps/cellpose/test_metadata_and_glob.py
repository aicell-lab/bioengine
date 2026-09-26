from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

# apps/cellpose-finetuning uses a hyphen so it can't be imported directly
_main_path = Path(__file__).parents[3] / "apps" / "cellpose-finetuning" / "main.py"
_spec = importlib.util.spec_from_file_location("cellpose_finetuning_main", _main_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
list_matching_artifact_paths = _mod.list_matching_artifact_paths
make_training_pairs_from_metadata = _mod.make_training_pairs_from_metadata
match_image_annotation_pairs = _mod.match_image_annotation_pairs


def test_match_image_annotation_pairs_with_nested_globs() -> None:
    image_files = [
        "images/a/t0001.tif",
        "images/b/t0002.tif",
        "images/c/extra.tif",
    ]
    annotation_files = [
        "annotations/a/t0001_mask.ome.tif",
        "annotations/b/t0002_mask.ome.tif",
        "annotations/x/other_mask.ome.tif",
    ]

    pairs = match_image_annotation_pairs(
        image_files,
        annotation_files,
        "images/*/*.tif",
        "annotations/*/*_mask.ome.tif",
    )

    assert pairs == [
        ("images/a/t0001.tif", ["annotations/a/t0001_mask.ome.tif"]),
        ("images/b/t0002.tif", ["annotations/b/t0002_mask.ome.tif"]),
    ]


def test_match_image_annotation_pairs_with_mixed_ome_suffix_convention() -> None:
    image_files = [
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
        "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
    ]
    annotation_files = [
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0000_mask.ome.tif",
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0001_mask.ome.tif",
    ]

    pairs = match_image_annotation_pairs(
        image_files,
        annotation_files,
        "images/108bb69d-2e52-4382-8100-e96173db24ee/*.tif",
        "annotations/108bb69d-2e52-4382-8100-e96173db24ee/*_mask.ome.tif",
    )

    assert pairs == [
        (
            "images/108bb69d-2e52-4382-8100-e96173db24ee/t0000.ome.tif",
            ["annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0000_mask.ome.tif"],
        ),
        (
            "images/108bb69d-2e52-4382-8100-e96173db24ee/t0001.ome.tif",
            ["annotations/108bb69d-2e52-4382-8100-e96173db24ee/t0001_mask.ome.tif"],
        ),
    ]


class _FakeArtifact:
    async def ls(self, folder_path: str, detail: bool = False):
        await asyncio.sleep(0)
        folder = folder_path.rstrip("/") + "/"
        if folder == "metadata/":
            entries = [{"path": "metadata/sample.json", "type": "file"}]
        elif folder == "images/train/":
            entries = [
                {"path": "images/train/t0001.tif", "type": "file"},
                {"path": "images/train/t0002.tif", "type": "file"},
            ]
        else:
            entries = []
        return entries if detail else [e["path"].rsplit("/", 1)[-1] for e in entries]

    async def get(self, remote_paths, local_paths, on_error="ignore"):
        await asyncio.sleep(0)
        for remote, local in zip(remote_paths, local_paths):
            local_path = Path(local)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if str(remote).endswith("metadata/sample.json"):
                local_path.write_text(
                    """
[
  {"image_path": "images/train/t0001.tif", "mask_path": "annotations/train/t0001_mask.ome.tif", "split": "train"},
  {"image_path": "images/test/t0002.tif", "mask_path": "annotations/test/t0002_mask.ome.tif", "split": "test"}
]
                    """.strip(),
                    encoding="utf-8",
                )
            else:
                local_path.write_bytes(b"dummy")


class _FakeArtifactNestedMetadata:
        async def ls(self, folder_path: str, detail: bool = False):
                await asyncio.sleep(0)
                folder = folder_path.rstrip("/") + "/"
                if folder == "metadata/":
                        entries = [{"path": "metadata/records.json", "type": "file"}]
                else:
                        entries = []
                return entries if detail else [
                        e["path"].rsplit("/", 1)[-1] for e in entries
                ]

        async def get(self, remote_paths, local_paths, on_error="ignore"):
                await asyncio.sleep(0)
                for remote, local in zip(remote_paths, local_paths):
                        local_path = Path(local)
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        if str(remote).endswith("metadata/records.json"):
                                local_path.write_text(
                                        """
{
    "payload": {
        "items": [
            {
                "imagePath": "images/train/t0010.tif",
                "maskPath": "annotations/train/t0010_mask.ome.tif",
                "split": "train"
            },
            {
                "source": {"path": "images/test/t0011.tif"},
                "target": {"path": "annotations/test/t0011_mask.ome.tif"},
                "subset": "validation"
            }
        ]
    }
}
                                        """.strip(),
                                        encoding="utf-8",
                                )
                        else:
                                local_path.write_bytes(b"dummy")


def test_make_training_pairs_from_metadata(tmp_path: Path) -> None:
    artifact = _FakeArtifact()
    train_pairs, test_pairs = asyncio.run(
        make_training_pairs_from_metadata(
            artifact=artifact,
            metadata_dir="metadata/",
            save_path=tmp_path,
            n_samples=None,
        )
    )

    assert len(train_pairs) == 1
    assert len(test_pairs) == 1
    assert train_pairs[0]["image"].exists()
    assert train_pairs[0]["annotation"].exists()
    assert test_pairs[0]["image"].exists()
    assert test_pairs[0]["annotation"].exists()


def test_list_matching_artifact_paths_directory_without_trailing_slash() -> None:
    artifact = _FakeArtifact()
    matches = asyncio.run(list_matching_artifact_paths(artifact, "images/train"))
    assert matches == [
        "images/train/t0001.tif",
        "images/train/t0002.tif",
    ]


def test_make_training_pairs_from_nested_metadata_formats(tmp_path: Path) -> None:
    artifact = _FakeArtifactNestedMetadata()
    train_pairs, test_pairs = asyncio.run(
        make_training_pairs_from_metadata(
            artifact=artifact,
            metadata_dir="metadata/",
            save_path=tmp_path,
            n_samples=None,
        )
    )

    assert len(train_pairs) == 1
    assert len(test_pairs) == 1
    assert train_pairs[0]["image"].as_posix().endswith("images/train/t0010.tif")
    assert train_pairs[0]["annotation"].as_posix().endswith("annotations/train/t0010_mask.ome.tif")
    assert test_pairs[0]["image"].as_posix().endswith("images/test/t0011.tif")
    assert test_pairs[0]["annotation"].as_posix().endswith("annotations/test/t0011_mask.ome.tif")
