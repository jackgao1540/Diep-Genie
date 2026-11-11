import pytest
from scripts.validate_public_boundary import validate_paths


def test_public_boundary_permits_source_and_exact_root_readme():
    assert validate_paths(["README.md", "genie_diep/model.py", "docs/assets/sample-01-original.webp", "assets/comparison.png"])["public_markdown"] == ["README.md"]


@pytest.mark.parametrize("path", ["HISTORY.md", "history.MD", "notes.Md", "web/README.md", "readme.md", "runs/log.json", "Private-Archive/original.txt", "weights/model.PT", "browser-screenshot.png", ".env", "feedback.jsonl", "backup.bundle"])
def test_public_boundary_rejects_private_paths_and_case_variants(path):
    with pytest.raises(ValueError, match="Private paths"):
        validate_paths(["README.md", path])
