"""Project packaging, archive creation, and kernel metadata generation."""

import json
import logging
import zipfile
from pathlib import Path

from kagglex.config import RunConfig
from kagglex.ignore import IgnoreFilter
from kagglex.workspace import ProjectType, detect_project_type

logger = logging.getLogger(__name__)

# Inline code payload limit for Kaggle kernels (5 MB).
MAX_PAYLOAD_BYTES = 5 * 1024 * 1024
MAX_KERNEL_UPLOAD_BYTES = MAX_PAYLOAD_BYTES


def package_project(
    project_dir: Path,
    output_zip: Path,
    target_file: Path | None = None,
    ignore_filter: IgnoreFilter | None = None,
) -> Path:
    """Package project files into a zip archive based on detected project layout.

    Args:
        project_dir: Root directory of project to package.
        output_zip: Destination zip file path.
        target_file: Optional standalone script file.
        ignore_filter: Optional IgnoreFilter instance.

    Returns:
        Path to generated zip archive.
    """
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    filt = ignore_filter or IgnoreFilter(base_dir=project_dir)

    proj_type, canonical_root = detect_project_type(
        target_file if target_file else project_dir
    )

    file_count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if target_file and proj_type == ProjectType.STANDALONE_SCRIPT:
            # Standalone script mode: package script and accompanying files if present
            zf.write(target_file, arcname=target_file.name)
            file_count += 1

            for companion in ["requirements.txt", "README.md", ".env"]:
                comp_path = canonical_root / companion
                if comp_path.is_file() and not filt.should_ignore(comp_path):
                    zf.write(comp_path, arcname=companion)
                    file_count += 1
        else:
            # Package directory structure respecting ignore filter
            for file_path in canonical_root.rglob("*"):
                if not file_path.is_file():
                    continue

                if filt.should_ignore(file_path):
                    continue

                arcname = file_path.relative_to(canonical_root)
                zf.write(file_path, arcname=str(arcname))
                file_count += 1

    size_bytes = output_zip.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Bundled project size ({size_mb:.2f} MB) exceeds maximum limit "
            f"({MAX_PAYLOAD_BYTES / (1024 * 1024):.0f} MB). "
            "Please add large files to .gitignore or .kaggleignore, or push them as a Kaggle Dataset."
        )

    logger.info(
        "Packaged project '%s' (%d files, %.2f MB) into %s",
        canonical_root.name,
        file_count,
        size_mb,
        output_zip.name,
    )
    return output_zip


def package_local_data(
    data_paths: list[Path],
    base_dir: Path,
    output_zip: Path,
    ignore_filter: IgnoreFilter | None = None,
) -> Path | None:
    """Package small local data files or manifests into a data zip archive.

    Args:
        data_paths: List of file or folder paths.
        base_dir: Base directory for resolving relative paths.
        output_zip: Target zip archive path.
        ignore_filter: Optional ignore filter.

    Returns:
        Path to zip archive if files were bundled, None otherwise.
    """
    if not data_paths:
        return None

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    filt = ignore_filter or IgnoreFilter(
        base_dir=base_dir, use_gitignore=False, use_kaggleignore=False
    )
    file_count = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in data_paths:
            if p.is_absolute():
                resolved_p = p.resolve()
            elif (Path.cwd() / p).resolve().exists():
                resolved_p = (Path.cwd() / p).resolve()
            else:
                resolved_p = (base_dir / p).resolve()

            if not resolved_p.exists():
                logger.warning("Data path does not exist: %s", resolved_p)
                continue

            ref_dir = base_dir.resolve()
            try:
                resolved_p.relative_to(ref_dir)
            except ValueError:
                ref_dir = resolved_p.parent

            if resolved_p.is_file():
                if not filt.should_ignore(resolved_p):
                    try:
                        arc_name = str(resolved_p.relative_to(ref_dir))
                    except ValueError:
                        arc_name = resolved_p.name
                    zf.write(resolved_p, arcname=arc_name)
                    file_count += 1
            elif resolved_p.is_dir():
                for f in resolved_p.rglob("*"):
                    if f.is_file() and not filt.should_ignore(f):
                        try:
                            arc_name = str(f.relative_to(ref_dir))
                        except ValueError:
                            arc_name = str(f.relative_to(resolved_p.parent))
                        zf.write(f, arcname=arc_name)
                        file_count += 1

    if file_count == 0:
        if output_zip.exists():
            output_zip.unlink()
        return None

    size_bytes = output_zip.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Bundled data size ({size_mb:.2f} MB) exceeds maximum upload limit "
            f"({MAX_PAYLOAD_BYTES / (1024 * 1024):.0f} MB). "
            "Please upload large datasets using `kagglex dataset push`."
        )

    logger.info(
        "Packaged %d local data files (%.2f MB) into %s",
        file_count,
        size_mb,
        output_zip.name,
    )
    return output_zip


def create_kernel_metadata(
    config: RunConfig,
    kaggle_username: str,
    staging_dir: Path,
    code_file: str = "kaggle_bootstrap.py",
) -> Path:
    """Generate kernel-metadata.json file required by Kaggle.

    Args:
        config: Run configuration.
        kaggle_username: Authenticated Kaggle username.
        staging_dir: Directory where bundle is staged.
        code_file: Remote entrypoint filename.

    Returns:
        Path to generated kernel-metadata.json.
    """
    enable_gpu = config.gpu_type in {"t4-2x", "p100"}
    enable_tpu = config.enable_tpu or config.gpu_type == "v3-8"

    # Normalize dataset sources to ensure username prefix
    dataset_sources = []
    for ds in config.dataset_slugs:
        ds = ds.strip()
        if "/" not in ds:
            ds = f"{kaggle_username}/{ds}"
        dataset_sources.append(ds)

    # Normalize parent kernel sources (for output chaining)
    kernel_sources = []
    for ks in config.parent_kernels:
        ks = ks.strip()
        if "/" not in ks:
            ks = f"{kaggle_username}/{ks}"
        kernel_sources.append(ks)

    metadata = {
        "id": f"{kaggle_username}/{config.slug}",
        "title": config.title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "true" if enable_tpu else "false",
        "enable_internet": "true" if config.enable_internet else "false",
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
        "competition_sources": [],
    }

    metadata_path = staging_dir / "kernel-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.debug("Generated kernel metadata at %s", metadata_path)
    return metadata_path
