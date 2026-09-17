from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from svd import SVD


DEFAULT_URLS = ("https://picsum.photos/512?grayscale",)
DEFAULT_RANKS = (1, 5, 10, 25, 50)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def load_urls(urls: Iterable[str], url_file: str | None) -> list[str]:
    all_urls = [url.strip() for url in urls if url.strip()]

    if url_file:
        with open(url_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_urls.append(line)

    return all_urls or list(DEFAULT_URLS)


def safe_image_name(url: str, index: int) -> str:
    parsed = urlparse(url)
    path_name = Path(parsed.path).stem
    base = path_name or parsed.netloc or f"image_{index}"
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{index:02d}_{base}_{url_hash}"


def download_grayscale_image(url: str, timeout: int) -> Image.Image:
    request = Request(url, headers={"User-Agent": "svd-image-pipeline/1.0"})

    with urlopen(request, timeout=timeout) as response:
        image_bytes = response.read()

    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.convert("L")


def resize_for_svd(image: Image.Image, max_size: int | None) -> Image.Image:
    if max_size is None:
        return image

    image = image.copy()
    image.thumbnail((max_size, max_size), LANCZOS)
    return image


def rank_approximation(svd: SVD, image_array: np.ndarray, rank: int) -> np.ndarray:
    matrix = image_array.astype(np.float64)
    approximation = svd.forward(matrix, rank_approx=rank)

    if hasattr(approximation, "detach"):
        approximation = approximation.detach().cpu().numpy()

    return np.clip(approximation, 0, 255).astype(np.uint8)


def save_comparison_grid(
    original: Image.Image,
    approximations: dict[int, Image.Image],
    output_path: Path,
) -> None:
    panels = [("original", original), *[(f"rank {rank}", image) for rank, image in approximations.items()]]
    label_height = 26
    gap = 8
    width = sum(image.width for _, image in panels) + gap * (len(panels) - 1)
    height = max(image.height for _, image in panels) + label_height

    grid = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(grid)

    x = 0
    for label, image in panels:
        draw.text((x + 4, 6), label, fill=0)
        grid.paste(image, (x, label_height))
        x += image.width + gap

    grid.save(output_path)


def process_url(
    url: str,
    index: int,
    ranks: Iterable[int],
    output_dir: Path,
    max_size: int | None,
    timeout: int,
) -> list[Path]:
    image_name = safe_image_name(url, index)
    image_dir = output_dir / image_name
    image_dir.mkdir(parents=True, exist_ok=True)

    grayscale = download_grayscale_image(url, timeout=timeout)
    grayscale = resize_for_svd(grayscale, max_size=max_size)
    grayscale_path = image_dir / "original_grayscale.png"
    grayscale.save(grayscale_path)

    image_array = np.asarray(grayscale)
    saved_paths = [grayscale_path]
    approximations: dict[int, Image.Image] = {}
    svd = SVD()

    for rank in ranks:
        approx_array = rank_approximation(svd, image_array, rank)
        approx_image = Image.fromarray(approx_array)
        approximations[rank] = approx_image

        approx_path = image_dir / f"rank_{rank}.png"
        approx_image.save(approx_path)
        saved_paths.append(approx_path)

    grid_path = image_dir / "rank_comparison.png"
    save_comparison_grid(grayscale, approximations, grid_path)
    saved_paths.append(grid_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download grayscale images and save low-rank SVD approximations."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="Image URLs to download. If omitted, a default grayscale internet image is used.",
    )
    parser.add_argument(
        "--url-file",
        help="Optional text file of image URLs, one URL per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where output image folders are written.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANKS),
        help="SVD ranks to save.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Largest width or height before SVD. Use 0 to keep the original size.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Download timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = sorted({rank for rank in args.ranks if rank > 0})
    max_size = None if args.max_size == 0 else args.max_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = load_urls(args.urls, args.url_file)

    for index, url in enumerate(urls, start=1):
        try:
            saved_paths = process_url(
                url=url,
                index=index,
                ranks=ranks,
                output_dir=output_dir,
                max_size=max_size,
                timeout=args.timeout,
            )
        except (HTTPError, URLError, OSError, ValueError) as error:
            print(f"Failed to process {url}: {error}")
            continue

        print(f"Processed {url}")
        for path in saved_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
