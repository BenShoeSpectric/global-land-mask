"""Build compact polygon data from official Natural Earth land shapefiles."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def read_polygon_rings(filename):
    """Yield rings from an ESRI polygon shapefile using only the standard library."""
    with filename.open("rb") as stream:
        header = stream.read(100)
        if len(header) != 100 or struct.unpack("<i", header[32:36])[0] != 5:
            raise ValueError(f"{filename} is not a polygon shapefile")
        while True:
            record_header = stream.read(8)
            if not record_header:
                return
            if len(record_header) != 8:
                raise ValueError(f"truncated record header in {filename}")
            _, words = struct.unpack(">2i", record_header)
            record = stream.read(words * 2)
            if len(record) != words * 2:
                raise ValueError(f"truncated record in {filename}")
            shape_type = struct.unpack("<i", record[:4])[0]
            if shape_type == 0:
                continue
            if shape_type != 5:
                raise ValueError(f"unexpected shape type {shape_type} in {filename}")
            part_count, point_count = struct.unpack("<2i", record[36:44])
            parts = list(struct.unpack(f"<{part_count}i", record[44 : 44 + 4 * part_count]))
            parts.append(point_count)
            start = 44 + 4 * part_count
            coordinates = np.frombuffer(
                record, dtype="<f8", count=point_count * 2, offset=start
            ).reshape(-1, 2)
            for first, last in zip(parts, parts[1:]):
                ring = coordinates[first:last]
                if len(ring) >= 4:
                    yield ring.copy()


def signed_area(ring):
    return 0.5 * np.sum(
        ring[:, 0] * np.roll(ring[:, 1], -1)
        - np.roll(ring[:, 0], -1) * ring[:, 1]
    )


def build(source, output, resolution, source_commit):
    rings = list(read_polygon_rings(source))
    points = np.concatenate(rings).astype("<f4")
    offsets = np.concatenate(([0], np.cumsum([len(ring) for ring in rings]))).astype("<i4")
    bounds = np.asarray(
        [
            [ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max()]
            for ring in rings
        ],
        dtype="<f4",
    )
    # ESRI polygon exteriors are clockwise; counter-clockwise rings are holes.
    is_hole = np.asarray([signed_area(ring) > 0 for ring in rings], dtype=bool)
    np.savez_compressed(
        output,
        points=points,
        ring_offsets=offsets,
        bounds=bounds,
        is_hole=is_hole,
    )
    metadata = {
        "dataset": "Natural Earth Land",
        "resolution": resolution,
        "scale": f"1:{int(resolution[:-1]) * 1_000_000:,}",
        "source": "https://github.com/nvkelso/natural-earth-vector",
        "source_commit": source_commit,
        "source_file": source.name,
        "natural_earth_version": source.with_suffix(".VERSION.txt").read_text().strip(),
        "license": "Public domain; see https://www.naturalearthdata.com/about/terms-of-use/",
        "ring_count": len(rings),
        "point_count": len(points),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{resolution}: {len(rings)} rings, {len(points)} points, "
        f"{is_hole.sum()} holes -> {output}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for resolution in ("10m", "50m"):
        source = args.source_root / f"{resolution}_physical" / f"ne_{resolution}_land.shp"
        output = args.output_dir / f"natural-earth-{resolution}-land.npz"
        build(source, output, resolution, args.source_commit)


if __name__ == "__main__":
    main()
