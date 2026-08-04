"""
Convert the on-disk TACO dataset to a YOLO-format dataset for YOLOv8 training.

TACO (Trash Annotations in Context) ships a COCO-format annotations file but
not the images themselves. This script:

1. Reads ``TACO-master/TACO-master/data/annotations.json``.
2. Downloads each image from the URL recorded in the annotations
   (``flickr_640_url`` -> ``flickr_url`` -> ``coco_url``). Skips already
   downloaded files and silently drops 404 / timeout errors. Many TACO
   Flickr URLs are dead since the dataset was archived; the script keeps
   what it can.
3. Maps TACO's 60 fine-grained categories onto the project's six classes
   (``cardboard``, ``glass``, ``metal``, ``paper``, ``plastic``, ``trash``)
   using a curated, defensible mapping table.
4. Performs a stratified 70/15/15 train/val/test split on the dominant
   project class of each image, with ``random_state=42``.
5. Writes YOLO-format ``.txt`` label files (one line per object,
   ``class_id cx cy w h`` with all coordinates normalised to [0, 1]) and
   copies images into ``data/yolo/images/{train,val,test}/``.
6. Writes ``data/yolo/dataset.yaml`` in the Ultralytics schema.
7. Writes a Markdown summary to ``reports/taco_conversion_report.md``.

Usage::

    python scripts/convert_taco_to_yolo.py                 # full conversion
    python scripts/convert_taco_to_yolo.py --limit 10      # smoke test (10 images)
    python scripts/convert_taco_to_yolo.py --skip-download # reuse existing images
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Project taxonomy and TACO category mapping
# ---------------------------------------------------------------------------

# The project uses exactly six class names. They must match the keys of
# MATERIAL_FEATURE_LIBRARY in src/multimodal_inference.py exactly so the
# YOLO detector's output feeds cleanly into the Random Forest.
PROJECT_CLASSES: List[str] = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(PROJECT_CLASSES)}

# Map every TACO category_id to a project class. Any id not in this table
# is dropped (not silently re-mapped) so the user can audit the mapping.
# Annotations dropped include only the annotation, not the image: the image
# is still emitted if at least one of its annotations maps to a project class.
TACO_CATEGORY_TO_PROJECT: Dict[int, str] = {
    # cardboard: cartons, drink/meal/egg cartons, pizza box, toilet tube
    13: "cardboard", 14: "cardboard", 15: "cardboard", 16: "cardboard",
    17: "cardboard", 18: "cardboard", 19: "cardboard",
    # glass: glass bottle, broken glass, glass cup, glass jar
    6: "glass", 9: "glass", 23: "glass", 26: "glass",
    # metal: cans, foils, lids, scrap metal, pop tab, rope, shoe, squeezable tube
    0: "metal", 8: "metal", 10: "metal", 11: "metal", 12: "metal", 28: "metal",
    50: "metal", 51: "metal", 52: "metal", 53: "metal", 54: "metal",
    # paper: only "Normal paper" — the rest are coated or contaminated and go to trash
    33: "paper",
    # plastic: blister packs, plastic bottles, caps, plastic cups, plastic film,
    # wrappers, straws (incl. plastic-coated paper straw), styrofoam, "other plastic"
    2: "plastic", 3: "plastic", 4: "plastic", 5: "plastic", 7: "plastic",
    21: "plastic", 24: "plastic", 29: "plastic", 36: "plastic", 39: "plastic",
    55: "plastic", 56: "plastic", 57: "plastic",
    # trash: everything else combustible or non-combustible we don't want as RDF
    1: "trash", 20: "trash", 22: "trash", 25: "trash", 27: "trash",
    30: "trash", 31: "trash", 32: "trash", 34: "trash", 35: "trash",
    37: "trash", 38: "trash", 40: "trash", 41: "trash", 42: "trash",
    43: "trash", 44: "trash", 45: "trash", 46: "trash", 47: "trash",
    48: "trash", 49: "trash", 58: "trash", 59: "trash",
}

# Human-readable notes for the dissertation documentation. Loaded lazily.
TACO_CATEGORY_NAMES: Dict[int, str] = {}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TACOAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox_xywh: Tuple[float, float, float, float]  # COCO format: top-left x,y + w,h


@dataclass
class TACOImage:
    image_id: int
    width: int
    height: int
    file_name: str
    flickr_640_url: Optional[str]
    flickr_url: Optional[str]
    coco_url: Optional[str]
    annotations: List[TACOAnnotation] = field(default_factory=list)

    def candidate_urls(self) -> List[str]:
        """Try smaller (640px) first, then full Flickr, then COCO URL."""
        urls: List[str] = []
        for candidate in (self.flickr_640_url, self.flickr_url, self.coco_url):
            if candidate:
                urls.append(candidate)
        return urls


@dataclass
class ProjectAnnotation:
    class_index: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_file: Path) -> logging.Logger:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("taco_to_yolo")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, UnicodeDecodeError):
        pass

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# TACO loading
# ---------------------------------------------------------------------------


def load_taco(annotations_path: Path) -> Tuple[Dict[int, TACOImage], Dict[int, str]]:
    """Read TACO's COCO-format annotations and return (images, category_names)."""
    with annotations_path.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    cat_names: Dict[int, str] = {cat["id"]: cat.get("name", "?") for cat in coco.get("categories", [])}

    images: Dict[int, TACOImage] = {}
    for im in coco.get("images", []):
        images[im["id"]] = TACOImage(
            image_id=int(im["id"]),
            width=int(im.get("width") or 0),
            height=int(im.get("height") or 0),
            file_name=str(im.get("file_name", "")),
            flickr_640_url=im.get("flickr_640_url"),
            flickr_url=im.get("flickr_url"),
            coco_url=im.get("coco_url"),
        )

    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        image_id = int(ann["image_id"])
        if image_id not in images:
            continue
        images[image_id].annotations.append(
            TACOAnnotation(
                annotation_id=int(ann["id"]),
                image_id=image_id,
                category_id=int(ann["category_id"]),
                bbox_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
        )
    return images, cat_names


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 15.0) -> Optional[bytes]:
    """Download a single URL with a short timeout. Returns None on any failure."""
    try:
        import requests  # local import so the script still imports without it
    except ImportError:
        return None
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "taco-yolo-converter/1.0 (+academic use)"},
        )
        if response.status_code != 200 or len(response.content) == 0:
            return None
        return response.content
    except Exception:  # noqa: BLE001 - requests raises many subclasses
        return None


def download_image(image: TACOImage, dest: Path, logger: logging.Logger) -> bool:
    """Download a single TACO image. Skips if already present. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for url in image.candidate_urls():
        content = _http_get(url)
        if content is None:
            continue
        try:
            dest.write_bytes(content)
            return True
        except OSError as exc:
            logger.debug("Failed to write %s: %s", dest, exc)
            return False
    return False


def download_all(
    images: Iterable[TACOImage],
    dest_dir: Path,
    logger: logging.Logger,
    limit: Optional[int] = None,
) -> Tuple[int, int, List[TACOImage]]:
    """Download every TACO image into dest_dir. Returns (succeeded, attempted, failed_list)."""
    import requests  # fail fast if missing

    image_list = list(images)
    if limit is not None and limit > 0:
        image_list = image_list[:limit]

    dest_dir.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed: List[TACOImage] = []
    for image in tqdm(image_list, desc="Downloading TACO images", unit="img"):
        dest = dest_dir / f"{image.image_id:06d}.jpg"
        if download_image(image, dest, logger):
            succeeded += 1
        else:
            failed.append(image)
    return succeeded, len(image_list), failed


# ---------------------------------------------------------------------------
# Annotation conversion
# ---------------------------------------------------------------------------


def coco_to_yolo(
    bbox_xywh: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert COCO [x,y,w,h] (top-left, pixels) to YOLO [cx,cy,w,h] (centre, normalised)."""
    x, y, w, h = bbox_xywh
    if img_w <= 0 or img_h <= 0:
        return None
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        # Clip and accept; this happens at image borders.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 1e-6), 1.0)
        nh = min(max(nh, 1e-6), 1.0)
    return cx, cy, nw, nh


def project_annotations_for_image(image: TACOImage) -> List[ProjectAnnotation]:
    """Convert every TACO annotation on this image into a ProjectAnnotation."""
    out: List[ProjectAnnotation] = []
    for ann in image.annotations:
        project_class = TACO_CATEGORY_TO_PROJECT.get(ann.category_id)
        if project_class is None:
            continue
        converted = coco_to_yolo(ann.bbox_xywh, image.width, image.height)
        if converted is None:
            continue
        cx, cy, w, h = converted
        out.append(
            ProjectAnnotation(
                class_index=CLASS_TO_INDEX[project_class],
                class_name=project_class,
                cx=cx, cy=cy, w=w, h=h,
            )
        )
    return out


def dominant_class(annotations: List[ProjectAnnotation]) -> Optional[str]:
    """Pick the image's dominant project class (most annotations; ties broken alphabetically)."""
    if not annotations:
        return None
    counts = Counter(a.class_name for a in annotations)
    top_count = max(counts.values())
    candidates = sorted(name for name, n in counts.items() if n == top_count)
    return candidates[0]


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    items_with_class: List[Tuple[int, str]], seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """70/15/15 stratified split by class. Returns (train_ids, val_ids, test_ids)."""
    from sklearn.model_selection import train_test_split

    if not items_with_class:
        return [], [], []
    ids = [t[0] for t in items_with_class]
    classes = [t[1] for t in items_with_class]

    # First split: 70% train, 30% temp
    train_ids, temp_ids, train_cls, temp_cls = train_test_split(
        ids, classes, test_size=0.30, random_state=seed, stratify=classes
    )
    # Second split: split the 30% into 15% val and 15% test
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_cls, test_size=0.50, random_state=seed, stratify=temp_cls
    )
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_yolo_label(label_path: Path, annotations: List[ProjectAnnotation]) -> None:
    """Write a single YOLO label file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as fh:
        for ann in annotations:
            fh.write(f"{ann.class_index} {ann.cx:.6f} {ann.cy:.6f} {ann.w:.6f} {ann.h:.6f}\n")


def copy_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    shutil.copy2(src, dest)


def write_dataset_yaml(
    yaml_path: Path,
    project_root_for_yaml: Path,
) -> None:
    """Write the Ultralytics dataset.yaml in the schema the trainer expects."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(project_root_for_yaml).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(PROJECT_CLASSES)},
    }
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_summary(
    report_path: Path,
    attempted: int,
    succeeded: int,
    split_counts: Dict[str, int],
    class_image_counts: Dict[str, int],
    class_annotation_counts: Dict[str, int],
    failed_image_ids: List[int],
    duration_seconds: float,
) -> None:
    """Write a Markdown summary of the conversion."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    download_rate = (succeeded / attempted) if attempted else 0.0
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# TACO -> YOLO Conversion Report\n\n")
        fh.write(f"**Attempted downloads:** {attempted}  \n")
        fh.write(f"**Successful downloads:** {succeeded} ({download_rate:.1%})  \n")
        fh.write(f"**Conversion duration:** {duration_seconds:.1f}s\n\n")
        fh.write("---\n\n")
        fh.write("## Split Counts\n\n")
        fh.write("| Split | Images |\n|---|---:|\n")
        for split in ("train", "val", "test"):
            fh.write(f"| {split} | {split_counts.get(split, 0)} |\n")
        fh.write("\n## Per-Class Annotation Counts\n\n")
        fh.write("| Class | Annotations | Images |\n|---|---:|---:|\n")
        for cls in PROJECT_CLASSES:
            fh.write(
                f"| {cls} | {class_annotation_counts.get(cls, 0)} | "
                f"{class_image_counts.get(cls, 0)} |\n"
            )
        if failed_image_ids:
            fh.write("\n## Failed Image IDs (first 50)\n\n")
            fh.write("These images could not be downloaded. They are excluded from training.\n\n")
            sample = failed_image_ids[:50]
            fh.write(", ".join(f"`{iid}`" for iid in sample))
            if len(failed_image_ids) > 50:
                fh.write(f"\n\n*(+{len(failed_image_ids) - 50} more not shown)*\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TACO annotations to a YOLO dataset.")
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(project_root / "TACO-master" / "TACO-master" / "data" / "annotations.json"),
        help="Path to TACO annotations.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "yolo"),
        help="Output directory (must contain images/, labels/, dataset.yaml).",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(project_root / "data" / "taco_images"),
        help="Where to cache downloaded TACO images.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(project_root / "reports" / "taco_conversion_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(project_root / "taco_conversion.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images to download (smoke test). 0 = no limit.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse any images already in --download-dir; do not re-download failures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified split.",
    )
    return parser.parse_args()


def main() -> None:
    import time

    args = parse_args()
    logger = setup_logging(Path(args.log))
    logger.info("=" * 80)
    logger.info("TACO -> YOLO conversion")
    logger.info("=" * 80)

    annotations_path = Path(args.annotations)
    output_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    report_path = Path(args.report)

    if not annotations_path.exists():
        logger.error("TACO annotations not found at %s", annotations_path)
        return

    started = time.time()

    logger.info("Loading TACO annotations from %s", annotations_path)
    images, cat_names = load_taco(annotations_path)
    TACO_CATEGORY_NAMES.update(cat_names)
    logger.info("Loaded %d images and %d categories", len(images), len(cat_names))

    # Download (or reuse) the images
    attempted = len(images)
    succeeded = 0
    failed: List[TACOImage] = []
    if args.skip_download:
        logger.info("Skipping download, reusing files in %s", download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)
        for image in tqdm(images.values(), desc="Reusing cached images", unit="img"):
            dest = download_dir / f"{image.image_id:06d}.jpg"
            if dest.exists() and dest.stat().st_size > 0:
                succeeded += 1
            else:
                failed.append(image)
        attempted = len(images)
    else:
        logger.info("Downloading images to %s", download_dir)
        succeeded, attempted, failed = download_all(
            images.values(),
            download_dir,
            logger,
            limit=args.limit if args.limit > 0 else None,
        )

    if succeeded == 0:
        logger.error(
            "No images were available. TACO Flickr URLs may all be dead. "
            "Drop a YOLO-format dataset at %s instead (e.g. from Roboflow).",
            output_dir,
        )
        return

    download_rate = succeeded / attempted if attempted else 0.0
    logger.info("Downloaded %d / %d images (%.1f%%)", succeeded, attempted, 100 * download_rate)

    # Build per-image project annotations and assign dominant class
    images_with_class: List[Tuple[int, str]] = []
    project_anns_by_image: Dict[int, List[ProjectAnnotation]] = {}
    for image in images.values():
        cached = download_dir / f"{image.image_id:06d}.jpg"
        if not cached.exists() or cached.stat().st_size == 0:
            continue
        project_anns = project_annotations_for_image(image)
        if not project_anns:
            continue
        dom = dominant_class(project_anns)
        if dom is None:
            continue
        project_anns_by_image[image.image_id] = project_anns
        images_with_class.append((image.image_id, dom))

    logger.info(
        "%d images have at least one project-class annotation", len(images_with_class)
    )

    # Stratified split
    train_ids, val_ids, test_ids = stratified_split(images_with_class, seed=args.seed)
    id_to_split: Dict[int, str] = {}
    for iid in train_ids:
        id_to_split[iid] = "train"
    for iid in val_ids:
        id_to_split[iid] = "val"
    for iid in test_ids:
        id_to_split[iid] = "test"

    # Reset output dirs (clean slate for labels and images, but keep dataset.yaml)
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    for sub in ("train", "val", "test"):
        for d in (images_root / sub, labels_root / sub):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    split_counts: Dict[str, int] = Counter()
    class_image_counts: Dict[str, int] = Counter()
    class_annotation_counts: Dict[str, int] = Counter()

    for image_id, split in id_to_split.items():
        image = images[image_id]
        cached = download_dir / f"{image.image_id:06d}.jpg"
        if not cached.exists():
            continue
        target_image = images_root / split / f"{image_id:06d}.jpg"
        copy_image(cached, target_image)

        annotations = project_anns_by_image[image_id]
        label_path = labels_root / split / f"{image_id:06d}.txt"
        write_yolo_label(label_path, annotations)

        split_counts[split] += 1
        for ann in annotations:
            class_annotation_counts[ann.class_name] += 1
        # dominant class counted once per image
        dom = dominant_class(annotations)
        if dom:
            class_image_counts[dom] += 1

    # dataset.yaml
    write_dataset_yaml(output_dir / "dataset.yaml", Path("data/yolo"))

    # Markdown summary
    duration = time.time() - started
    write_summary(
        report_path=report_path,
        attempted=attempted,
        succeeded=succeeded,
        split_counts=dict(split_counts),
        class_image_counts=dict(class_image_counts),
        class_annotation_counts=dict(class_annotation_counts),
        failed_image_ids=[im.image_id for im in failed],
        duration_seconds=duration,
    )

    logger.info("Wrote dataset.yaml to %s", output_dir / "dataset.yaml")
    logger.info("Wrote summary to %s", report_path)
    logger.info("Done in %.1fs", duration)
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("taco_to_yolo").error("Conversion failed: %s", exc, exc_info=True)
        raise