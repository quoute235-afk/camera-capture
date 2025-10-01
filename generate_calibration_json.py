"""Command line tool to compute stereo calibration and export JSON."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import cv2

from stereo_reconstruction import compute_stereo_calibration, save_calibration

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def _iter_image_files(directory: Path) -> Iterator[Path]:
    """Yield image files from *directory* sorted by name."""

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            yield path


def _load_image_pairs(left_dir: Path, right_dir: Path) -> List[Tuple[cv2.Mat, cv2.Mat]]:
    """Load image pairs from the provided directories."""

    left_images = list(_iter_image_files(left_dir))
    right_images = list(_iter_image_files(right_dir))

    if not left_images:
        raise FileNotFoundError(f"Keine Bilder in {left_dir} gefunden.")
    if not right_images:
        raise FileNotFoundError(f"Keine Bilder in {right_dir} gefunden.")

    if len(left_images) != len(right_images):
        print(
            "Warnung: Unterschiedliche Anzahl an Dateien – es werden nur die ersten "
            f"{min(len(left_images), len(right_images))} Paare verwendet."
        )

    pairs: List[Tuple[cv2.Mat, cv2.Mat]] = []
    for left_path, right_path in zip(left_images, right_images):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None:
            raise ValueError(f"Konnte Bild nicht lesen: {left_path}")
        if right is None:
            raise ValueError(f"Konnte Bild nicht lesen: {right_path}")
        pairs.append((left, right))
    return pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet eine Stereo-Kalibrierung aus linken und rechten Bildern und "
            "speichert das Ergebnis als JSON."
        )
    )
    parser.add_argument("--left", required=True, help="Ordner mit Bildern der linken Kamera.")
    parser.add_argument("--right", required=True, help="Ordner mit Bildern der rechten Kamera.")
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Kantenlänge eines Schachbrett-Quadrats in gewünschten Einheiten (Standard: 1.0).",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=6,
        help="Anzahl innerer Ecken pro Zeile des Schachbretts (Standard: 6).",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=6,
        help="Anzahl innerer Ecken pro Spalte des Schachbretts (Standard: 6).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stereo_calibration.json"),
        help="Dateiname der zu erzeugenden JSON-Datei.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    left_dir = Path(args.left)
    right_dir = Path(args.right)

    if not left_dir.is_dir():
        raise NotADirectoryError(f"Ordner nicht gefunden: {left_dir}")
    if not right_dir.is_dir():
        raise NotADirectoryError(f"Ordner nicht gefunden: {right_dir}")

    print(f"Lade Bildpaare aus {left_dir} und {right_dir} …")
    image_pairs = _load_image_pairs(left_dir, right_dir)

    print(
        f"Starte Kalibrierung mit {len(image_pairs)} Paar(en), "
        f"Schachbrett {args.pattern_cols}x{args.pattern_rows}, Quadratgröße {args.square_size}."
    )

    calibration = compute_stereo_calibration(
        image_pairs,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size=args.square_size,
    )

    save_calibration(args.output, calibration)
    print(f"Kalibrierung gespeichert in {args.output.resolve()}")


if __name__ == "__main__":
    main()
