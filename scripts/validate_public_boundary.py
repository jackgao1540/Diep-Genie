"""Enforce the public source boundary against Git's tracked path list."""
from pathlib import Path, PurePosixPath
import subprocess


PRIVATE_DIRECTORIES = {"data", "runs", "research", "exports", "private-archive", ".venv", ".agents", ".codex", "__pycache__", ".pytest_cache"}
PRIVATE_SUFFIXES = {".pt", ".pth", ".ckpt", ".zip", ".mp4", ".webm", ".pdf", ".pem", ".key", ".bundle"}


def validate_paths(paths):
    violations = []
    for name in paths:
        path = PurePosixPath(name)
        parts = {part.lower() for part in path.parts}
        lower = name.lower()
        if ((path.suffix.lower() == ".md" and name != "README.md")
                or parts & PRIVATE_DIRECTORIES
                or path.suffix.lower() in PRIVATE_SUFFIXES
                or "screenshot" in lower
                or path.name.lower().startswith(".env")
                or path.name.lower() == "feedback.jsonl"
                or path.name.lower().endswith("checkpoint.jpg")):
            violations.append(name)
    if violations:
        raise ValueError("Private paths tracked in public source: " + ", ".join(violations))
    return {"validation": "passed", "tracked_files": len(paths), "public_markdown": [p for p in paths if p.lower().endswith(".md")]}


def validate(root):
    root = Path(root).resolve()
    result = subprocess.run(["git", f"--git-dir={root / '.git'}", f"--work-tree={root}", "ls-files", "-z"], check=True, capture_output=True)
    return validate_paths([p for p in result.stdout.decode().split("\0") if p])


if __name__ == "__main__":
    import json
    print(json.dumps(validate(Path(__file__).resolve().parents[1])))
