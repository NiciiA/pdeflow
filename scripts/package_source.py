"""Package portable source for the Colab archive workflow; no data or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def source_file(path: Path, root: Path) -> bool:
    """Exclude generated metadata, hidden files, caches and binary run data."""
    return (
        path.is_file()
        and path.name != "colab-legacy.ipynb"
        and not path.is_symlink()
        and not any(part.startswith(".") or part.endswith(".egg-info") or part in {
            "__pycache__", "node_modules", "cache", "caches", "dist", "build"
        } for part in path.relative_to(root).parts)
        and path.suffix not in {".pyc", ".pt", ".pth", ".npz", ".npy", ".pem", ".key"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [root / p for p in ("pyproject.toml", "requirements-tested.txt", "README.md", "SOURCE_PACKAGE.md", "AGENTS.md", "MANIFEST.in", ".gitignore", ".nvmrc", "setup_structure.sh")]
    for folder in ("src", "configs", "docs", "tests", "scripts", "notebooks"):
        files.extend(p for p in (root / folder).rglob("*") if source_file(p, root))
    frontend = root / "frontend"
    for name in ("package.json", "package-lock.json", "check-node.mjs", "tsconfig.json", "vite.config.ts", "components.json", "index.html", "eslint.config.js", "playwright.config.ts", "THIRD_PARTY.md"):
        if (frontend / name).is_file(): files.append(frontend / name)
    for folder in ("src", "e2e"):
        files.extend(p for p in (frontend / folder).rglob("*") if source_file(p, root))
    manifest = {
        "format": 1,
        "root": "pdeflow/",
        "instructions": "SOURCE_PACKAGE.md",
        "excluded": ["outputs", ".git", ".venv", "node_modules", "caches", "build metadata", "datasets", "checkpoints", "references"],
        "files": {},
    }
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            archive.writestr("pdeflow/" + name, data)
            manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        archive.writestr("pdeflow/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
