#!/usr/bin/env python3
"""Map anchor-relative FoR relations to image locations."""

from __future__ import annotations

import argparse
import json
from typing import Literal

CameraView = Literal["eye_level", "top_down"]
AnchorDirection = Literal[
    "facing_viewer",
    "facing_away_from_viewer",
    "facing_image_left",
    "facing_image_right",
    "facing_image_top",
    "facing_image_bottom",
]
Relation = Literal[
    "anchor_front",
    "anchor_back",
    "anchor_left",
    "anchor_right",
    "anchor_up",
    "anchor_down",
]
ImagePosition = Literal[
    "image_left",
    "image_right",
    "image_top",
    "image_bottom",
    "foreground",
    "background",
    "Unknown",
]

EYE_LEVEL_MAPPING: dict[AnchorDirection, dict[Relation, ImagePosition]] = {
    "facing_viewer": {
        "anchor_front": "foreground",
        "anchor_back": "background",
        "anchor_left": "image_right",
        "anchor_right": "image_left",
        "anchor_up": "image_top",
        "anchor_down": "image_bottom",
    },
    "facing_away_from_viewer": {
        "anchor_front": "background",
        "anchor_back": "foreground",
        "anchor_left": "image_left",
        "anchor_right": "image_right",
        "anchor_up": "image_top",
        "anchor_down": "image_bottom",
    },
    "facing_image_left": {
        "anchor_front": "image_left",
        "anchor_back": "image_right",
        "anchor_left": "foreground",
        "anchor_right": "background",
        "anchor_up": "image_top",
        "anchor_down": "image_bottom",
    },
    "facing_image_right": {
        "anchor_front": "image_right",
        "anchor_back": "image_left",
        "anchor_left": "background",
        "anchor_right": "foreground",
        "anchor_up": "image_top",
        "anchor_down": "image_bottom",
    },
    "facing_image_top": {
        "anchor_front": "image_top",
        "anchor_back": "image_bottom",
        "anchor_left": "Unknown",
        "anchor_right": "Unknown",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_image_bottom": {
        "anchor_front": "image_bottom",
        "anchor_back": "image_top",
        "anchor_left": "Unknown",
        "anchor_right": "Unknown",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
}

TOP_DOWN_MAPPING: dict[AnchorDirection, dict[Relation, ImagePosition]] = {
    "facing_image_top": {
        "anchor_front": "image_top",
        "anchor_back": "image_bottom",
        "anchor_left": "image_left",
        "anchor_right": "image_right",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_image_bottom": {
        "anchor_front": "image_bottom",
        "anchor_back": "image_top",
        "anchor_left": "image_right",
        "anchor_right": "image_left",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_image_left": {
        "anchor_front": "image_left",
        "anchor_back": "image_right",
        "anchor_left": "image_bottom",
        "anchor_right": "image_top",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_image_right": {
        "anchor_front": "image_right",
        "anchor_back": "image_left",
        "anchor_left": "image_top",
        "anchor_right": "image_bottom",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_viewer": {
        "anchor_front": "Unknown",
        "anchor_back": "Unknown",
        "anchor_left": "Unknown",
        "anchor_right": "Unknown",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
    "facing_away_from_viewer": {
        "anchor_front": "Unknown",
        "anchor_back": "Unknown",
        "anchor_left": "Unknown",
        "anchor_right": "Unknown",
        "anchor_up": "Unknown",
        "anchor_down": "Unknown",
    },
}

MAPPING: dict[CameraView, dict[AnchorDirection, dict[Relation, ImagePosition]]] = {
    "eye_level": EYE_LEVEL_MAPPING,
    "top_down": TOP_DOWN_MAPPING,
}


def get_image_position(
    camera_view: CameraView,
    anchor_direction: AnchorDirection,
    relation: Relation,
) -> ImagePosition:
    """Return expected image location for a camera view, anchor direction, and FoR relation."""
    try:
        return MAPPING[camera_view][anchor_direction][relation]
    except KeyError as exc:
        raise ValueError(
            f"unsupported combination: camera_view={camera_view!r}, "
            f"anchor_direction={anchor_direction!r}, relation={relation!r}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Map FoR relation to expected image position.")
    parser.add_argument("--camera-view", choices=sorted(MAPPING), required=True)
    parser.add_argument(
        "--anchor-direction",
        choices=sorted({direction for view in MAPPING.values() for direction in view}),
        required=True,
    )
    parser.add_argument(
        "--relation",
        choices=sorted({relation for view in MAPPING.values() for mapping in view.values() for relation in mapping}),
        required=True,
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of plain text.")
    args = parser.parse_args()

    image_position = get_image_position(args.camera_view, args.anchor_direction, args.relation)
    if args.json:
        print(
            json.dumps(
                {
                    "camera_view": args.camera_view,
                    "anchor_direction": args.anchor_direction,
                    "relation": args.relation,
                    "image_position": image_position,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(image_position)


if __name__ == "__main__":
    main()
