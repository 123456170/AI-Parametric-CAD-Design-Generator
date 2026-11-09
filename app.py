from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import time
import hashlib
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import cadquery as cq
    CQ_AVAILABLE = True
    CQ_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    cq = None
    CQ_AVAILABLE = False
    CQ_IMPORT_ERROR = str(exc)


APP_NAME = "AI Parametric CAD Design Generator"
APP_BUILD = "milestone-1-single-file-demo-fixed-session-state"
DB_PATH = Path(os.environ.get("CAD_AI_DB", "cad_ai_demo.sqlite3"))

INSTRUCTION_WIDGET_KEY = "instruction_widget"

DEFAULT_INSTRUCTION = (
    "Create a mounting plate 120 mm long, 80 mm wide, 10 mm thick "
    "with four 6 mm holes and 12 mm edge margin."
)

UNIT_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
}

COUNT_PARAMS = {"hole_count", "bolt_count", "teeth"}


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    parameter: Optional[str] = None


OBJECT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rectangular_block": {
        "label": "Rectangular block / cube",
        "required": ["length", "width", "height"],
        "defaults": {
            "length": 100.0,
            "width": 80.0,
            "height": 20.0,
        },
    },
    "cylinder": {
        "label": "Cylinder",
        "required": ["height"],
        "defaults": {
            "diameter": 30.0,
            "radius": 15.0,
            "height": 80.0,
        },
    },
    "mounting_plate": {
        "label": "Mounting plate",
        "required": ["length", "width", "thickness"],
        "defaults": {
            "length": 120.0,
            "width": 80.0,
            "thickness": 10.0,
            "hole_count": 0,
            "hole_diameter": 6.0,
            "edge_margin": 12.0,
        },
    },
    "flange": {
        "label": "Flange",
        "required": ["outer_diameter", "thickness"],
        "defaults": {
            "outer_diameter": 100.0,
            "thickness": 12.0,
            "bore_diameter": 25.0,
            "bolt_count": 4,
            "bolt_circle_diameter": 75.0,
            "bolt_hole_diameter": 8.0,
        },
    },
    "simple_enclosure": {
        "label": "Simple enclosure",
        "required": ["length", "width", "height", "wall_thickness"],
        "defaults": {
            "length": 120.0,
            "width": 80.0,
            "height": 50.0,
            "wall_thickness": 4.0,
        },
    },
    "shaft": {
        "label": "Shaft",
        "required": ["diameter", "length"],
        "defaults": {
            "diameter": 20.0,
            "length": 150.0,
        },
    },
    "basic_gear": {
        "label": "Basic spur gear",
        "required": ["teeth", "module", "face_width", "bore_diameter"],
        "defaults": {
            "teeth": 20,
            "module": 2.0,
            "face_width": 10.0,
            "bore_diameter": 8.0,
        },
    },
    "mechanical_bracket": {
        "label": "Mechanical L-bracket",
        "required": ["base_length", "base_width", "thickness", "height"],
        "defaults": {
            "base_length": 100.0,
            "base_width": 60.0,
            "thickness": 8.0,
            "height": 70.0,
            "hole_count": 0,
            "hole_diameter": 6.0,
            "edge_margin": 12.0,
        },
    },
}

OBJECT_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("mechanical_bracket", ["bracket", "l bracket", "angle bracket", "l-bracket"]),
    ("mounting_plate", ["mounting plate", "mount plate", "plate"]),
    ("flange", ["flange"]),
    ("simple_enclosure", ["enclosure", "case", "cover", "box", "housing"]),
    ("basic_gear", ["gear", "spur gear", "cog"]),
    ("shaft", ["shaft", "axle", "pin shaft"]),
    ("cylinder", ["cylinder", "cylindrical", "rod", "pin"]),
    ("rectangular_block", ["block", "cube", "cuboid", "rectangular"]),
]

PARAM_KEYWORDS: Dict[str, List[str]] = {
    "length": ["length", "long", "len", "x dimension"],
    "width": ["width", "wide", "y dimension"],
    "height": ["height", "high", "tall", "z dimension"],
    "thickness": ["thickness", "thick", "plate thick"],
    "diameter": ["diameter", "dia", "diam"],
    "radius": ["radius"],
    "outer_diameter": ["outer diameter", "outside diameter", "od", "flange diameter", "diameter"],
    "bore_diameter": ["bore", "bore diameter", "inner diameter", "id"],
    "bolt_circle_diameter": ["bolt circle", "bolt circle diameter", "bcd", "pcd"],
    "bolt_hole_diameter": ["bolt hole diameter", "bolt diameter", "hole diameter", "holes diameter", "bolt hole"],
    "hole_diameter": ["hole diameter", "hole dia", "holes diameter", "hole"],
    "edge_margin": ["edge margin", "margin", "edge distance"],
    "wall_thickness": ["wall thickness", "wall", "shell thickness"],
    "module": ["module"],
    "face_width": ["face width", "gear face", "gear width", "width"],
    "base_length": ["base length", "base long", "length"],
    "base_width": ["base width", "base wide", "width"],
}

OBJECT_PARAM_PRIORITY: Dict[str, List[str]] = {
    "rectangular_block": ["length", "width", "height"],
    "cylinder": ["diameter", "radius", "height", "length"],
    "mounting_plate": ["length", "width", "thickness", "hole_diameter", "edge_margin"],
    "flange": ["outer_diameter", "thickness", "bolt_circle_diameter", "bolt_hole_diameter", "bore_diameter"],
    "simple_enclosure": ["length", "width", "height", "wall_thickness"],
    "shaft": ["diameter", "length", "height"],
    "basic_gear": ["module", "face_width", "bore_diameter"],
    "mechanical_bracket": ["base_length", "base_width", "thickness", "height", "hole_diameter", "edge_margin"],
}

WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

NUMBER_RE = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", re.IGNORECASE)
MEASURE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in|inch|inches)?", re.IGNORECASE)

COUNT_CONTEXTS = {
    "hole_count": ["hole", "holes"],
    "bolt_count": ["bolt", "bolts"],
    "teeth": ["tooth", "teeth"],
}


ARCHITECTURE_DOC = """
# Architecture

This single-file delivery keeps all runtime code in `app.py`, while preserving logical module boundaries.

## Logical modules in `app.py`

| Module | Main functions / classes |
|---|---|
| Requirement parser | `parse_instruction`, `extract_measurements`, `extract_counts`, `detect_object_type` |
| Engineering specification validator | `validate_spec`, `Issue` |
| Parametric geometry engine | `generate_model`, `gen_*` functions |
| CAD model generator | CadQuery / OpenCASCADE solid operations |
| Geometric constraint manager | hole spacing, edge margins, bore limits, gear root/bore checks |
| CAD validation engine | `validate_geometry`, solid validity, volume, bounding box |
| 3D preview system | `tessellate_shape`, `threejs_viewer` |
| Export manager | `export_geometry`, STEP/STL download controls |
| Project and revision manager | SQLite functions: `init_db`, `save_revision`, `list_revisions` |
| Automated testing system | `test_*` functions and `run_test_suite` |

## Generation pipeline

1. Receive natural-language or structured design request.
2. Extract dimensions, units, object type, and features.
3. Convert request into structured JSON specification.
4. Validate units, dimensions, and engineering constraints.
5. Generate real solid geometry using CadQuery / OpenCASCADE.
6. Validate the resulting solid.
7. Tessellate the solid for interactive Three.js preview.
8. Allow parameter modification and regeneration.
9. Export STEP / STL / JSON.
10. Persist project revisions and test reports in SQLite.

## Database

SQLite is used first through `DB_PATH`.

The database layer is isolated in functions, so a later PostgreSQL adapter can replace:

- `init_db`
- `create_project`
- `list_projects`
- `save_revision`
- `list_revisions`
- `save_test_run`

## Milestone status

This file implements a runnable Milestone 1 foundation and adds code paths for Milestone 2+ objects.

Do not treat feature completeness as verified until the built-in test suite or `pytest app.py` has been executed on your machine.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().replace("-", " ").split())


def word_to_number(token: str) -> Optional[int]:
    token = token.lower().strip()
    if token in WORD_NUMBERS:
        return WORD_NUMBERS[token]
    try:
        return int(float(token))
    except Exception:
        return None


def detect_object_type(text: str) -> str:
    scored: List[Tuple[int, str]] = []
    for obj, keywords in OBJECT_KEYWORDS:
        score = 0
        for kw in keywords:
            if kw in text:
                score += len(kw)
        if score > 0:
            scored.append((score, obj))
    if not scored:
        return "rectangular_block"
    scored.sort(reverse=True)
    return scored[0][1]


def extract_measurements(text: str) -> List[Dict[str, Any]]:
    measurements: List[Dict[str, Any]] = []
    for match in MEASURE_RE.finditer(text):
        value = float(match.group("value"))
        unit_token = (match.group("unit") or "mm").lower()
        had_unit = bool(match.group("unit"))
        mm_value = value * UNIT_TO_MM.get(unit_token, 1.0)

        start = max(0, match.start() - 45)
        end = min(len(text), match.end() + 45)
        context = text[start:end]

        measurements.append(
            {
                "value": value,
                "unit": unit_token,
                "had_unit": had_unit,
                "mm": mm_value,
                "context": context,
                "raw": match.group(0),
            }
        )
    return measurements


def extract_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for param, keywords in COUNT_CONTEXTS.items():
        for match in NUMBER_RE.finditer(text):
            start = max(0, match.start() - 5)
            end = min(len(text), match.end() + 35)
            window = text[start:end]
            if any(kw in window for kw in keywords):
                value = word_to_number(match.group(0))
                if value is not None:
                    counts[param] = int(value)
                    break
    return counts


def param_matches(param: str, context: str) -> bool:
    keywords = PARAM_KEYWORDS.get(param, [param.replace("_", " ")])
    return any(kw in context for kw in keywords)


def default_spec(object_type: str) -> Dict[str, Any]:
    object_type = object_type if object_type in OBJECT_SCHEMAS else "rectangular_block"
    params = dict(OBJECT_SCHEMAS[object_type]["defaults"])
    return normalize_spec(
        {
            "schema_version": "1.0",
            "object_type": object_type,
            "units": "mm",
            "parameters": params,
            "features": [],
            "constraints": {},
            "metadata": {"source": "template"},
        }
    )


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    object_type = spec.get("object_type", "rectangular_block")
    if object_type not in OBJECT_SCHEMAS:
        object_type = "rectangular_block"

    raw_params = spec.get("parameters", {}) or {}
    params: Dict[str, Any] = {}

    for key, value in raw_params.items():
        if value is None:
            continue
        if key in COUNT_PARAMS:
            try:
                params[key] = int(round(float(value)))
            except Exception:
                params[key] = value
        else:
            try:
                params[key] = float(value)
            except Exception:
                params[key] = value

    return {
        "schema_version": spec.get("schema_version", "1.0"),
        "object_type": object_type,
        "units": "mm",
        "parameters": params,
        "features": spec.get("features", []),
        "constraints": spec.get("constraints", {}),
        "metadata": spec.get("metadata", {}),
    }


def parse_instruction(text: str) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    norm = normalize_text(text)
    object_type = detect_object_type(norm)
    measurements = extract_measurements(norm)
    counts = extract_counts(norm)

    params: Dict[str, Any] = {}
    used = [False] * len(measurements)

    for param, value in counts.items():
        params[param] = int(value)

    for idx, meas in enumerate(measurements):
        for count_value in counts.values():
            if abs(meas["value"] - float(count_value)) < 1e-9 and any(
                kw in meas["context"] for kw in ["hole", "holes", "bolt", "bolts", "tooth", "teeth"]
            ):
                used[idx] = True
                break

    priority = OBJECT_PARAM_PRIORITY.get(object_type, [])

    for param in priority:
        if param in params:
            continue
        if param in COUNT_PARAMS:
            continue

        candidate_idx = None
        candidate_meas = None

        for idx, meas in enumerate(measurements):
            if used[idx]:
                continue
            if param_matches(param, meas["context"]):
                candidate_idx = idx
                candidate_meas = meas
                break

        if candidate_idx is not None and candidate_meas is not None:
            used[candidate_idx] = True
            if param in COUNT_PARAMS:
                params[param] = int(candidate_meas["value"])
            else:
                params[param] = round(float(candidate_meas["mm"]), 6)

    unused_measurements = [measurements[i] for i in range(len(measurements)) if not used[i]]

    for param in priority:
        if param in COUNT_PARAMS:
            continue
        if param in params:
            continue
        if not unused_measurements:
            break
        meas = unused_measurements.pop(0)
        params[param] = round(float(meas["mm"]), 6)

    if object_type == "rectangular_block":
        if "length" in params and "width" not in params and "height" not in params:
            params["width"] = params["length"]
            params["height"] = params["length"]
            issues.append(Issue("info", "CUBE_INFERRED", "A single dimension was interpreted as a cube.", None))

    if object_type == "cylinder":
        if "length" in params and "height" not in params:
            params["height"] = params.pop("length")

    if object_type == "shaft":
        if "height" in params and "length" not in params:
            params["length"] = params.pop("height")

    if "radius" in params and "diameter" not in params:
        params["diameter"] = round(float(params["radius"]) * 2.0, 6)
    if "diameter" in params and "radius" not in params:
        params["radius"] = round(float(params["diameter"]) / 2.0, 6)

    if object_type in {"mounting_plate", "mechanical_bracket"}:
        if "hole_diameter" in params and "hole_count" not in params:
            params["hole_count"] = 1
            issues.append(
                Issue(
                    "warning",
                    "HOLE_COUNT_INFERRED",
                    "A hole diameter was provided without a count; one hole was assumed.",
                    "hole_count",
                )
            )

    if object_type == "flange":
        if "bolt_hole_diameter" in params and "bolt_count" not in params:
            params["bolt_count"] = 4
            issues.append(
                Issue(
                    "warning",
                    "BOLT_COUNT_INFERRED",
                    "A bolt hole diameter was provided without a count; four bolts were assumed.",
                    "bolt_count",
                )
            )

    unitless_non_counts = [
        meas
        for idx, meas in enumerate(measurements)
        if not used[idx] and not meas["had_unit"] and meas["value"] not in {float(v) for v in counts.values()}
    ]
    if unitless_non_counts:
        issues.append(
            Issue(
                "warning",
                "UNIT_ASSUMED",
                "One or more numeric dimensions had no explicit unit; millimetres were assumed.",
                None,
            )
        )

    schema = OBJECT_SCHEMAS[object_type]

    for required in schema["required"]:
        if required not in params:
            default_value = schema["defaults"].get(required)
            if default_value is None:
                issues.append(
                    Issue(
                        "error",
                        "MISSING_PARAMETER_NO_DEFAULT",
                        f"Required parameter '{required}' was not found and has no safe default.",
                        required,
                    )
                )
            else:
                params[required] = default_value
                issues.append(
                    Issue(
                        "warning",
                        "MISSING_PARAMETER_DEFAULTED",
                        f"Required parameter '{required}' was not found; using default {default_value}.",
                        required,
                    )
                )

    for key, value in schema["defaults"].items():
        if key not in params:
            params[key] = value

    spec = normalize_spec(
        {
            "schema_version": "1.0",
            "object_type": object_type,
            "units": "mm",
            "parameters": params,
            "features": [],
            "constraints": {},
            "metadata": {
                "source": "natural_language",
                "raw_instruction": text,
                "parsed_at": utc_now(),
            },
        }
    )

    return spec, issues


def validate_spec(spec: Dict[str, Any]) -> List[Issue]:
    issues: List[Issue] = []
    spec = normalize_spec(spec)
    object_type = spec["object_type"]
    params = spec["parameters"]
    schema = OBJECT_SCHEMAS[object_type]

    for required in schema["required"]:
        if required not in params:
            issues.append(Issue("error", "MISSING_PARAMETER", f"Missing required parameter: {required}.", required))

    for key, value in params.items():
        if key in COUNT_PARAMS:
            try:
                ivalue = int(value)
            except Exception:
                issues.append(Issue("error", "NON_NUMERIC_COUNT", f"Count parameter '{key}' must be an integer.", key))
                continue
            if ivalue < 0:
                issues.append(Issue("error", "NEGATIVE_COUNT", f"Count parameter '{key}' cannot be negative.", key))
            if key == "teeth" and ivalue < 8:
                issues.append(Issue("error", "GEAR_TEETH_TOO_SMALL", "A usable basic gear needs at least 8 teeth.", key))
        else:
            try:
                fvalue = float(value)
            except Exception:
                issues.append(Issue("error", "NON_NUMERIC_DIMENSION", f"Parameter '{key}' must be numeric.", key))
                continue
            if not math.isfinite(fvalue):
                issues.append(Issue("error", "NON_FINITE_DIMENSION", f"Parameter '{key}' must be finite.", key))
            elif fvalue <= 0.0:
                issues.append(Issue("error", "NON_POSITIVE_DIMENSION", f"Parameter '{key}' must be positive.", key))
            elif fvalue > 10000.0:
                issues.append(Issue("warning", "LARGE_DIMENSION", f"Parameter '{key}' is unusually large.", key))

    if object_type == "mounting_plate":
        hole_count = int(params.get("hole_count", 0) or 0)
        if hole_count > 0:
            hd = float(params.get("hole_diameter", 0.0) or 0.0)
            em = float(params.get("edge_margin", 0.0) or 0.0)
            length = float(params.get("length", 0.0) or 0.0)
            width = float(params.get("width", 0.0) or 0.0)

            if hd <= 0:
                issues.append(
                    Issue(
                        "error",
                        "MISSING_HOLE_DIAMETER",
                        "Hole count is positive but hole diameter is missing.",
                        "hole_diameter",
                    )
                )
            if em <= 0:
                issues.append(
                    Issue(
                        "error",
                        "MISSING_EDGE_MARGIN",
                        "Hole count is positive but edge margin is missing.",
                        "edge_margin",
                    )
                )

            if hd > 0 and em > 0:
                if em < hd / 2.0:
                    issues.append(
                        Issue(
                            "error",
                            "EDGE_MARGIN_TOO_SMALL",
                            "Edge margin is smaller than half the hole diameter.",
                            "edge_margin",
                        )
                    )
                elif em < hd / 2.0 + 2.0:
                    issues.append(
                        Issue(
                            "warning",
                            "EDGE_MARGIN_TIGHT",
                            "Edge margin is mechanically tight; recommend at least hole radius + 2 mm.",
                            "edge_margin",
                        )
                    )

                if hole_count == 4 and (length <= 2.0 * em + 1e-6 or width <= 2.0 * em + 1e-6):
                    issues.append(
                        Issue(
                            "error",
                            "HOLES_DO_NOT_FIT",
                            "Four corner holes do not fit within the plate dimensions.",
                            None,
                        )
                    )

                if hole_count > 4:
                    pattern_radius = min(length, width) / 2.0 - em - hd / 2.0
                    if pattern_radius <= 0:
                        issues.append(
                            Issue(
                                "error",
                                "CIRCULAR_HOLE_PATTERN_INVALID",
                                "Circular hole pattern radius is non-positive.",
                                None,
                            )
                        )

    if object_type == "flange":
        bolt_count = int(params.get("bolt_count", 0) or 0)
        if bolt_count > 0:
            od = float(params.get("outer_diameter", 0.0) or 0.0)
            bcd = float(params.get("bolt_circle_diameter", 0.0) or 0.0)
            bd = float(params.get("bolt_hole_diameter", 0.0) or 0.0)
            bore = float(params.get("bore_diameter", 0.0) or 0.0)

            if bcd <= bd:
                issues.append(
                    Issue(
                        "error",
                        "BOLT_CIRCLE_TOO_SMALL",
                        "Bolt circle diameter must exceed bolt hole diameter.",
                        "bolt_circle_diameter",
                    )
                )
            if od < bcd + bd:
                issues.append(
                    Issue(
                        "error",
                        "FLANGE_OD_TOO_SMALL",
                        "Outer diameter is too small for the bolt circle and bolt holes.",
                        "outer_diameter",
                    )
                )
            if bore > 0 and bore + bd >= bcd:
                issues.append(
                    Issue(
                        "error",
                        "BORE_INTERFERES_WITH_BOLTS",
                        "Central bore interferes with bolt holes.",
                        "bore_diameter",
                    )
                )

    if object_type == "basic_gear":
        teeth = int(params.get("teeth", 0) or 0)
        module = float(params.get("module", 0.0) or 0.0)
        bore = float(params.get("bore_diameter", 0.0) or 0.0)
        face_width = float(params.get("face_width", 0.0) or 0.0)

        if teeth > 0 and module > 0:
            root_radius = (module * teeth / 2.0) - (1.25 * module)
            if bore > 0 and root_radius <= bore / 2.0:
                issues.append(
                    Issue(
                        "error",
                        "GEAR_ROOT_BELOW_BORE",
                        "Gear root diameter is not larger than the bore.",
                        "bore_diameter",
                    )
                )
            if face_width <= 0:
                issues.append(
                    Issue(
                        "error",
                        "GEAR_FACE_WIDTH_INVALID",
                        "Gear face width must be positive.",
                        "face_width",
                    )
                )

    if object_type == "simple_enclosure":
        length = float(params.get("length", 0.0) or 0.0)
        width = float(params.get("width", 0.0) or 0.0)
        height = float(params.get("height", 0.0) or 0.0)
        wall = float(params.get("wall_thickness", 0.0) or 0.0)

        if wall > 0:
            if 2.0 * wall >= min(length, width):
                issues.append(
                    Issue(
                        "error",
                        "ENCLOSURE_WALL_TOO_THICK",
                        "Wall thickness is too large for the enclosure footprint.",
                        "wall_thickness",
                    )
                )
            if wall >= height:
                issues.append(
                    Issue(
                        "error",
                        "ENCLOSURE_HEIGHT_TOO_SMALL",
                        "Enclosure height must exceed wall thickness.",
                        "height",
                    )
                )

    if object_type == "mechanical_bracket":
        hole_count = int(params.get("hole_count", 0) or 0)
        if hole_count > 0:
            hd = float(params.get("hole_diameter", 0.0) or 0.0)
            em = float(params.get("edge_margin", 0.0) or 0.0)
            base_length = float(params.get("base_length", 0.0) or 0.0)

            if hd <= 0:
                issues.append(
                    Issue(
                        "error",
                        "MISSING_HOLE_DIAMETER",
                        "Bracket hole count is positive but hole diameter is missing.",
                        "hole_diameter",
                    )
                )
            if em <= 0:
                issues.append(
                    Issue(
                        "error",
                        "MISSING_EDGE_MARGIN",
                        "Bracket hole count is positive but edge margin is missing.",
                        "edge_margin",
                    )
                )
            if hd > 0 and em > 0 and em < hd / 2.0:
                issues.append(
                    Issue(
                        "error",
                        "BRACKET_EDGE_MARGIN_TOO_SMALL",
                        "Bracket edge margin is smaller than half the hole diameter.",
                        "edge_margin",
                    )
                )
            if base_length <= 2.0 * em + 1e-6:
                issues.append(
                    Issue(
                        "error",
                        "BRACKET_HOLES_DO_NOT_FIT",
                        "Bracket holes do not fit along the base length.",
                        "base_length",
                    )
                )

    return issues


def _require_cq() -> None:
    if not CQ_AVAILABLE:
        raise RuntimeError(f"CadQuery is not available. Import error: {CQ_IMPORT_ERROR}")


def _linear_positions(length: float, count: int, margin: float) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    usable = length - 2.0 * margin
    if usable <= 0:
        raise ValueError("Linear hole pattern does not fit.")
    spacing = usable / float(count - 1)
    start = -length / 2.0 + margin
    return [start + i * spacing for i in range(count)]


def _plate_hole_positions(
    length: float,
    width: float,
    count: int,
    hole_diameter: float,
    margin: float,
) -> List[Tuple[float, float]]:
    if count <= 0:
        return []
    if count == 1:
        return [(0.0, 0.0)]
    if count == 4:
        x = length / 2.0 - margin
        y = width / 2.0 - margin
        if x <= 0 or y <= 0:
            raise ValueError("Four corner holes do not fit on the plate.")
        return [(-x, -y), (x, -y), (x, y), (-x, y)]

    radius = min(length, width) / 2.0 - margin - hole_diameter / 2.0
    if radius <= 0:
        raise ValueError("Circular hole pattern does not fit on the plate.")
    return [
        (radius * math.cos(2.0 * math.pi * i / count), radius * math.sin(2.0 * math.pi * i / count))
        for i in range(count)
    ]


def gen_rectangular_block(params: Dict[str, Any]):
    _require_cq()
    return cq.Workplane("XY").box(
        float(params["length"]),
        float(params["width"]),
        float(params["height"]),
        centered=(True, True, False),
    )


def gen_cylinder(params: Dict[str, Any]):
    _require_cq()
    diameter = float(params.get("diameter", 0.0) or 0.0)
    radius = float(params.get("radius", 0.0) or 0.0)
    if diameter <= 0 and radius > 0:
        diameter = 2.0 * radius
    if diameter <= 0:
        raise ValueError("Cylinder requires a positive diameter or radius.")
    height = float(params.get("height", params.get("length", 0.0)) or 0.0)
    return cq.Workplane("XY").circle(diameter / 2.0).extrude(height)


def gen_mounting_plate(params: Dict[str, Any]):
    _require_cq()
    length = float(params["length"])
    width = float(params["width"])
    thickness = float(params["thickness"])

    model = cq.Workplane("XY").box(length, width, thickness, centered=(True, True, False))

    hole_count = int(params.get("hole_count", 0) or 0)
    if hole_count > 0:
        hole_diameter = float(params.get("hole_diameter", 6.0) or 6.0)
        edge_margin = float(params.get("edge_margin", 12.0) or 12.0)
        positions = _plate_hole_positions(length, width, hole_count, hole_diameter, edge_margin)
        for x, y in positions:
            cutter = cq.Workplane("XY").center(x, y).circle(hole_diameter / 2.0).extrude(thickness + 2.0)
            model = model.cut(cutter.val())

    return model


def gen_flange(params: Dict[str, Any]):
    _require_cq()
    outer_diameter = float(params["outer_diameter"])
    thickness = float(params["thickness"])

    model = cq.Workplane("XY").circle(outer_diameter / 2.0).extrude(thickness)

    bore = float(params.get("bore_diameter", 0.0) or 0.0)
    if bore > 0:
        cutter = cq.Workplane("XY").circle(bore / 2.0).extrude(thickness + 2.0)
        model = model.cut(cutter.val())

    bolt_count = int(params.get("bolt_count", 0) or 0)
    if bolt_count > 0:
        bcd = float(params["bolt_circle_diameter"])
        bd = float(params["bolt_hole_diameter"])
        radius = bcd / 2.0
        for i in range(bolt_count):
            angle = 2.0 * math.pi * i / float(bolt_count)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            cutter = cq.Workplane("XY").center(x, y).circle(bd / 2.0).extrude(thickness + 2.0)
            model = model.cut(cutter.val())

    return model


def gen_simple_enclosure(params: Dict[str, Any]):
    _require_cq()
    length = float(params["length"])
    width = float(params["width"])
    height = float(params["height"])
    wall = float(params["wall_thickness"])

    inner_length = length - 2.0 * wall
    inner_width = width - 2.0 * wall
    pocket_depth = height - wall

    if inner_length <= 0 or inner_width <= 0 or pocket_depth <= 0:
        raise ValueError("Enclosure wall thickness leaves no valid cavity.")

    model = cq.Workplane("XY").box(length, width, height, centered=(True, True, False))
    model = (
        model.faces(">Z")
        .workplane(centerOption="CenterOfBoundBox")
        .rect(inner_length, inner_width)
        .cutBlind(pocket_depth)
    )
    return model


def gen_shaft(params: Dict[str, Any]):
    _require_cq()
    diameter = float(params["diameter"])
    length = float(params.get("length", params.get("height", 0.0)) or 0.0)
    return cq.Workplane("XY").circle(diameter / 2.0).extrude(length)


def gen_basic_gear(params: Dict[str, Any]):
    _require_cq()
    teeth = int(params["teeth"])
    module = float(params["module"])
    face_width = float(params["face_width"])
    bore = float(params.get("bore_diameter", 0.0) or 0.0)

    if teeth < 8:
        raise ValueError("Basic gear requires at least 8 teeth.")

    pitch_radius = module * teeth / 2.0
    outer_radius = pitch_radius + module
    root_radius = max(pitch_radius - 1.25 * module, 0.1)

    if bore > 0 and root_radius <= bore / 2.0:
        raise ValueError("Gear bore interferes with gear root diameter.")

    points: List[Any] = []
    tooth_angle = 2.0 * math.pi / float(teeth)

    for i in range(teeth):
        base_angle = i * tooth_angle
        fractions = [0.0, 0.25, 0.5, 0.75]
        radii = [root_radius, outer_radius, outer_radius, root_radius]
        for frac, radius in zip(fractions, radii):
            angle = base_angle + frac * tooth_angle
            points.append(cq.Vector(radius * math.cos(angle), radius * math.sin(angle), 0.0))

    model = cq.Workplane("XY").polyline(points).close().extrude(face_width)

    if bore > 0:
        cutter = cq.Workplane("XY").circle(bore / 2.0).extrude(face_width + 2.0)
        model = model.cut(cutter.val())

    return model


def gen_mechanical_bracket(params: Dict[str, Any]):
    _require_cq()
    base_length = float(params["base_length"])
    base_width = float(params["base_width"])
    thickness = float(params["thickness"])
    height = float(params["height"])

    base = cq.Workplane("XY").box(base_length, base_width, thickness, centered=(True, True, False))
    leg = (
        cq.Workplane("XY")
        .center(0.0, base_width / 2.0 - thickness / 2.0)
        .box(base_length, thickness, height, centered=(True, True, False))
    )

    model = base.union(leg.val())

    hole_count = int(params.get("hole_count", 0) or 0)
    if hole_count > 0:
        hole_diameter = float(params.get("hole_diameter", 6.0) or 6.0)
        edge_margin = float(params.get("edge_margin", 12.0) or 12.0)
        x_positions = _linear_positions(base_length, hole_count, edge_margin)
        for x in x_positions:
            cutter = cq.Workplane("XY").center(x, 0.0).circle(hole_diameter / 2.0).extrude(thickness + 2.0)
            model = model.cut(cutter.val())

    return model


GENERATORS = {
    "rectangular_block": gen_rectangular_block,
    "cylinder": gen_cylinder,
    "mounting_plate": gen_mounting_plate,
    "flange": gen_flange,
    "simple_enclosure": gen_simple_enclosure,
    "shaft": gen_shaft,
    "basic_gear": gen_basic_gear,
    "mechanical_bracket": gen_mechanical_bracket,
}


def validate_geometry(shape: Any, generation_time_ms: float, spec_issues: Optional[List[Issue]] = None) -> Dict[str, Any]:
    spec_issues = spec_issues or []
    errors = [issue for issue in spec_issues if issue.severity == "error"]
    warnings = [issue for issue in spec_issues if issue.severity == "warning"]

    report: Dict[str, Any] = {
        "valid": False,
        "volume_mm3": 0.0,
        "surface_area_mm2": 0.0,
        "bounding_box_mm": {},
        "generation_time_ms": float(generation_time_ms),
        "validation_errors": len(errors),
        "validation_warnings": len(warnings),
        "constraint_satisfied": False,
        "error_message": "",
    }

    try:
        is_valid = bool(shape.isValid())
        volume = float(shape.Volume())
        area = float(shape.Area())
        bb = shape.BoundingBox()

        report["valid"] = is_valid and volume > 1e-9
        report["volume_mm3"] = volume
        report["surface_area_mm2"] = area
        report["bounding_box_mm"] = {
            "x_min": float(getattr(bb, "xmin", 0.0)),
            "x_max": float(getattr(bb, "xmax", 0.0)),
            "y_min": float(getattr(bb, "ymin", 0.0)),
            "y_max": float(getattr(bb, "ymax", 0.0)),
            "z_min": float(getattr(bb, "zmin", 0.0)),
            "z_max": float(getattr(bb, "zmax", 0.0)),
        }
        report["constraint_satisfied"] = report["valid"] and len(errors) == 0
    except Exception as exc:
        report["valid"] = False
        report["constraint_satisfied"] = False
        report["error_message"] = f"{type(exc).__name__}: {exc}"

    return report


def generate_model(spec: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    _require_cq()

    spec = normalize_spec(spec)
    issues = validate_spec(spec)
    errors = [issue for issue in issues if issue.severity == "error"]

    if errors:
        message = "; ".join(issue.message for issue in errors)
        raise ValueError(f"Specification validation failed: {message}")

    object_type = spec["object_type"]
    generator = GENERATORS.get(object_type)
    if generator is None:
        raise ValueError(f"Unsupported object type: {object_type}")

    start = time.perf_counter()
    model = generator(spec["parameters"])
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    shape = model.val()
    report = validate_geometry(shape, elapsed_ms, issues)

    if not report["valid"]:
        raise RuntimeError(report.get("error_message") or "Generated geometry is invalid.")

    return model, report


def _vector_xyz(vector: Any) -> List[float]:
    if hasattr(vector, "x"):
        return [float(vector.x), float(vector.y), float(vector.z)]
    if isinstance(vector, (list, tuple)):
        return [float(vector[0]), float(vector[1]), float(vector[2])]
    return [float(vector[0]), float(vector[1]), float(vector[2])]


def tessellate_shape(shape: Any, tolerance: float = 0.35, angular_tolerance: float = 0.45) -> Dict[str, Any]:
    _require_cq()

    vertices, triangles = shape.tessellate(tolerance, angular_tolerance)

    flat_positions: List[float] = []
    for vertex in vertices:
        flat_positions.extend(_vector_xyz(vertex))

    if not flat_positions:
        return {
            "positions": [],
            "indices": [],
            "vertex_count": 0,
            "triangle_count": 0,
            "scale": 1.0,
            "extent_mm": [0.0, 0.0, 0.0],
        }

    array = np.array(flat_positions, dtype=float).reshape(-1, 3)
    minimum = array.min(axis=0)
    maximum = array.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = maximum - minimum
    max_extent = float(max(extent.max(), 1e-6))
    scale = 8.0 / max_extent
    scaled = (array - center) * scale

    indices: List[int] = []
    for tri in triangles:
        try:
            indices.extend([int(tri[0]), int(tri[1]), int(tri[2])])
        except Exception:
            continue

    return {
        "positions": scaled.reshape(-1).tolist(),
        "indices": indices,
        "vertex_count": int(array.shape[0]),
        "triangle_count": len(indices) // 3,
        "scale": float(scale),
        "extent_mm": [float(extent[0]), float(extent[1]), float(extent[2])],
    }


def mesh_fingerprint(mesh_data: Dict[str, Any]) -> str:
    positions = [round(float(x), 6) for x in mesh_data.get("positions", [])]
    indices = [int(x) for x in mesh_data.get("indices", [])]
    payload = json.dumps({"positions": positions, "indices": indices}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_geometry(model: Any, export_format: str) -> bytes:
    _require_cq()

    if model is None:
        raise ValueError("No model available for export.")

    export_format = export_format.upper()
    if export_format not in {"STEP", "STL"}:
        raise ValueError("Only STEP and STL exports are supported in this demo.")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"model.{export_format.lower()}"
        cq.exporters.export(model, str(path), exportFormat=export_format)
        data = path.read_bytes()
        if not data:
            raise RuntimeError(f"{export_format} export produced an empty file.")
        return data


def spec_to_json_bytes(spec: Dict[str, Any]) -> bytes:
    return json.dumps(spec, indent=2, sort_keys=True).encode("utf-8")


def issues_to_dataframe(issues: List[Issue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["severity", "code", "message", "parameter"])
    return pd.DataFrame([asdict(issue) for issue in issues])


def issues_to_csv_bytes(issues: List[Issue]) -> bytes:
    return issues_to_dataframe(issues).to_csv(index=False).encode("utf-8")


def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    connection = db_connect()
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                revision_no INTEGER NOT NULL,
                instruction TEXT,
                spec_json TEXT NOT NULL,
                object_type TEXT,
                valid INTEGER,
                volume_mm3 REAL,
                generation_time_ms REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );

            CREATE TABLE IF NOT EXISTS test_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                total INTEGER,
                passed INTEGER,
                failed INTEGER,
                duration_ms REAL,
                metrics_json TEXT,
                report_json TEXT
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def create_project(name: str) -> int:
    name = (name or "Untitled CAD Project").strip()
    connection = db_connect()
    try:
        cursor = connection.execute(
            "INSERT INTO projects (name, created_at) VALUES (?, ?)",
            (name, utc_now()),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def list_projects() -> List[Dict[str, Any]]:
    connection = db_connect()
    try:
        rows = connection.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def save_revision(
    project_id: int,
    instruction: str,
    spec: Dict[str, Any],
    report: Optional[Dict[str, Any]],
) -> int:
    report = report or {}
    connection = db_connect()
    try:
        row = connection.execute(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 AS next_rev FROM revisions WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        revision_no = int(row["next_rev"])

        connection.execute(
            """
            INSERT INTO revisions (
                project_id,
                revision_no,
                instruction,
                spec_json,
                object_type,
                valid,
                volume_mm3,
                generation_time_ms,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                revision_no,
                instruction,
                json.dumps(spec, sort_keys=True),
                spec.get("object_type", ""),
                1 if report.get("valid") else 0,
                float(report.get("volume_mm3", 0.0) or 0.0),
                float(report.get("generation_time_ms", 0.0) or 0.0),
                utc_now(),
            ),
        )
        connection.commit()
        cursor = connection.execute("SELECT last_insert_rowid() AS id").fetchone()
        return int(cursor["id"])
    finally:
        connection.close()


def list_revisions(project_id: int) -> List[Dict[str, Any]]:
    connection = db_connect()
    try:
        rows = connection.execute(
            "SELECT * FROM revisions WHERE project_id = ? ORDER BY revision_no DESC",
            (project_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def save_test_run(
    total: int,
    passed: int,
    failed: int,
    duration_ms: float,
    metrics: Dict[str, Any],
    report: List[Dict[str, Any]],
) -> int:
    connection = db_connect()
    try:
        cursor = connection.execute(
            """
            INSERT INTO test_runs (
                created_at,
                total,
                passed,
                failed,
                duration_ms,
                metrics_json,
                report_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                total,
                passed,
                failed,
                float(duration_ms),
                json.dumps(metrics, sort_keys=True, default=str),
                json.dumps(report, sort_keys=True, default=str),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def _spec(object_type: str, **params: Any) -> Dict[str, Any]:
    return normalize_spec(
        {
            "schema_version": "1.0",
            "object_type": object_type,
            "units": "mm",
            "parameters": params,
            "features": [],
            "constraints": {},
            "metadata": {"source": "test"},
        }
    )


def test_invalid_dimensions() -> None:
    issues = validate_spec(_spec("rectangular_block", length=-10.0, width=20.0, height=5.0))
    assert any(issue.code == "NON_POSITIVE_DIMENSION" for issue in issues)


def test_missing_parameters() -> None:
    bad_spec = {
        "schema_version": "1.0",
        "object_type": "rectangular_block",
        "units": "mm",
        "parameters": {"width": 20.0, "height": 5.0},
        "features": [],
        "constraints": {},
        "metadata": {},
    }
    issues = validate_spec(bad_spec)
    assert any(issue.code == "MISSING_PARAMETER" for issue in issues)


def test_incorrect_units() -> None:
    spec, _ = parse_instruction("block 1 m long 10 cm wide 20 mm high")
    assert abs(spec["parameters"]["length"] - 1000.0) < 1e-6
    assert abs(spec["parameters"]["width"] - 100.0) < 1e-6
    assert abs(spec["parameters"]["height"] - 20.0) < 1e-6

    _, issues2 = parse_instruction("block 50 long 50 wide 10 high")
    assert any(issue.code == "UNIT_ASSUMED" for issue in issues2)


def test_geometry_failures() -> None:
    try:
        generate_model(_spec("basic_gear", teeth=6, module=1.0, face_width=5.0, bore_diameter=8.0))
        raise AssertionError("Expected gear generation to fail for invalid tooth count.")
    except ValueError:
        pass


def test_invalid_solids() -> None:
    model, report = generate_model(_spec("rectangular_block", length=20.0, width=10.0, height=5.0))
    assert report["valid"] is True
    assert report["volume_mm3"] > 0.0

    issues = validate_spec(_spec("rectangular_block", length=0.0, width=10.0, height=5.0))
    assert any(issue.code == "NON_POSITIVE_DIMENSION" for issue in issues)


def test_export_failures() -> None:
    model, _ = generate_model(_spec("rectangular_block", length=20.0, width=10.0, height=5.0))
    step_bytes = export_geometry(model, "STEP")
    stl_bytes = export_geometry(model, "STL")
    assert len(step_bytes) > 0
    assert len(stl_bytes) > 0

    export_failed = False
    try:
        export_geometry(None, "STEP")
    except Exception:
        export_failed = True
    assert export_failed is True


def test_parameter_regeneration() -> None:
    _, report_a = generate_model(_spec("rectangular_block", length=10.0, width=10.0, height=10.0))
    _, report_b = generate_model(_spec("rectangular_block", length=20.0, width=10.0, height=10.0))
    assert abs(report_b["volume_mm3"] - 2.0 * report_a["volume_mm3"]) < 1e-3


def test_repeatability() -> None:
    spec = _spec(
        "mounting_plate",
        length=100.0,
        width=60.0,
        thickness=8.0,
        hole_count=4,
        hole_diameter=6.0,
        edge_margin=12.0,
    )
    model_a, report_a = generate_model(spec)
    model_b, report_b = generate_model(spec)

    assert abs(report_a["volume_mm3"] - report_b["volume_mm3"]) < 1e-6

    mesh_a = tessellate_shape(model_a.val())
    mesh_b = tessellate_shape(model_b.val())
    assert mesh_fingerprint(mesh_a) == mesh_fingerprint(mesh_b)


TEST_CASES: List[Tuple[str, Any]] = [
    ("Invalid dimensions", test_invalid_dimensions),
    ("Missing parameters", test_missing_parameters),
    ("Incorrect units", test_incorrect_units),
    ("Geometry failures", test_geometry_failures),
    ("Invalid solids", test_invalid_solids),
    ("Export failures", test_export_failures),
    ("Parameter regeneration", test_parameter_regeneration),
    ("Repeatability", test_repeatability),
]


def run_test_suite() -> Tuple[Dict[str, Any], pd.DataFrame]:
    results: List[Dict[str, Any]] = []
    suite_start = time.perf_counter()

    for name, function in TEST_CASES:
        test_start = time.perf_counter()
        try:
            function()
            status = "PASS"
            error = ""
        except Exception as exc:
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
        test_ms = (time.perf_counter() - test_start) * 1000.0

        results.append(
            {
                "test": name,
                "status": status,
                "duration_ms": round(test_ms, 3),
                "error": error,
            }
        )

    total_duration_ms = (time.perf_counter() - suite_start) * 1000.0
    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = len(results) - passed

    metrics = {
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "test_pass_rate": passed / float(len(results)) if results else None,
        "duration_ms": round(total_duration_ms, 3),
        "generated_at": utc_now(),
    }

    save_test_run(
        total=len(results),
        passed=passed,
        failed=failed,
        duration_ms=total_duration_ms,
        metrics=metrics,
        report=results,
    )

    return metrics, pd.DataFrame(results)


def log_event(
    event_type: str,
    success: bool,
    generation_time_ms: Optional[float] = None,
    valid: Optional[bool] = None,
    constraint_satisfied: Optional[bool] = None,
) -> None:
    if "events" not in st.session_state:
        st.session_state.events = []
    st.session_state.events.append(
        {
            "timestamp": utc_now(),
            "type": event_type,
            "success": bool(success),
            "generation_time_ms": generation_time_ms,
            "valid": valid,
            "constraint_satisfied": constraint_satisfied,
        }
    )


def compute_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    generation_events = [event for event in events if event["type"] in {"generation", "parameter_modification"}]
    export_events = [event for event in events if event["type"] == "export"]
    modification_events = [event for event in events if event["type"] == "parameter_modification"]

    def rate(items: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [item.get(key) for item in items if item.get(key) is not None]
        if not values:
            return None
        return float(sum(1 for value in values if value)) / float(len(values))

    generation_times = [
        float(event["generation_time_ms"])
        for event in generation_events
        if event.get("generation_time_ms") is not None
    ]

    return {
        "generation_attempts": len(generation_events),
        "export_attempts": len(export_events),
        "modification_attempts": len(modification_events),
        "geometry_validity_rate": rate(generation_events, "valid"),
        "successful_export_rate": rate(export_events, "success"),
        "parameter_modification_success_rate": rate(modification_events, "success"),
        "constraint_satisfaction_rate": rate(generation_events, "constraint_satisfied"),
        "average_generation_time_ms": float(np.mean(generation_times)) if generation_times else None,
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100.0:.1f}%"


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f} ms"


def set_flash(level: str, message: str) -> None:
    st.session_state["_flash"] = {"level": level, "message": message}


def render_flash() -> None:
    flash = st.session_state.pop("_flash", None)
    if not flash:
        return
    level = flash.get("level", "info")
    message = flash.get("message", "")
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def sync_instruction_state() -> None:
    if "pending_instruction" in st.session_state:
        st.session_state.instruction = st.session_state.pop("pending_instruction")
        st.session_state[INSTRUCTION_WIDGET_KEY] = st.session_state.instruction
    elif INSTRUCTION_WIDGET_KEY in st.session_state:
        st.session_state.instruction = st.session_state[INSTRUCTION_WIDGET_KEY]
    else:
        st.session_state[INSTRUCTION_WIDGET_KEY] = st.session_state.get("instruction", DEFAULT_INSTRUCTION)


def render_issues(issues: List[Issue]) -> None:
    if not issues:
        st.success("No validation issues detected.")
        return

    dataframe = issues_to_dataframe(issues)
    for _, row in dataframe.iterrows():
        message = f"[{row['code']}] {row['message']}"
        if row["severity"] == "error":
            st.error(message)
        elif row["severity"] == "warning":
            st.warning(message)
        else:
            st.info(message)


def render_metrics(metrics: Dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric(
        "Geometry validity rate",
        fmt_pct(metrics["geometry_validity_rate"]),
        f"{metrics['generation_attempts']} attempts",
    )
    cols[1].metric(
        "Successful export rate",
        fmt_pct(metrics["successful_export_rate"]),
        f"{metrics['export_attempts']} attempts",
    )
    cols[2].metric("Avg generation time", fmt_ms(metrics["average_generation_time_ms"]))
    cols[3].metric(
        "Param mod success",
        fmt_pct(metrics["parameter_modification_success_rate"]),
        f"{metrics['modification_attempts']} attempts",
    )
    cols[4].metric("Constraint satisfaction", fmt_pct(metrics["constraint_satisfaction_rate"]))


VIEWER_HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: #0b1020;
  font-family: Inter, system-ui, -apple-system, sans-serif;
}
#container {
  width: 100%;
  height: 100%;
}
#hud {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  color: #e6edf7;
  background: rgba(8, 12, 24, 0.58);
  padding: 8px 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 10px;
  backdrop-filter: blur(6px);
  font-size: 13px;
}
button, a.button {
  cursor: pointer;
  background: #1f6feb;
  color: white;
  border: none;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 13px;
  text-decoration: none;
  display: inline-block;
}
button.secondary {
  background: #30363d;
}
#status {
  opacity: 0.88;
}
</style>
</head>
<body>
<div id="container"></div>
<div id="hud">
  <strong>Live CAD preview</strong>
  <button id="playBtn">Pause rotation</button>
  <button id="resetBtn" class="secondary">Reset view</button>
  <button id="recordBtn">Record live demo</button>
  <a id="downloadLink" class="button" style="display:none;">Download WebM</a>
  <span id="status">Auto-demo starting...</span>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three/build/three.module.js",
    "three/addons/": "https://unpkg.com/three/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MESH = __MESH_JSON__;
const container = document.getElementById('container');
const statusEl = document.getElementById('status');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const recordBtn = document.getElementById('recordBtn');
const downloadLink = document.getElementById('downloadLink');

if (!MESH.positions || MESH.positions.length === 0) {
  statusEl.textContent = 'No mesh data available.';
} else {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);

  const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(13, 10, 16);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(MESH.positions, 3));
  geometry.setIndex(MESH.indices);
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    color: 0x4da3ff,
    metalness: 0.18,
    roughness: 0.42,
    side: THREE.DoubleSide
  });

  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  const grid = new THREE.GridHelper(24, 24, 0x4466aa, 0x223044);
  grid.position.y = -5.5;
  scene.add(grid);

  const ambient = new THREE.AmbientLight(0xffffff, 0.45);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(8, 12, 10);
  scene.add(key);

  const fill = new THREE.DirectionalLight(0x88aaff, 0.45);
  fill.position.set(-10, 6, -8);
  scene.add(fill);

  let autoRotate = true;

  playBtn.addEventListener('click', () => {
    autoRotate = !autoRotate;
    playBtn.textContent = autoRotate ? 'Pause rotation' : 'Play rotation';
  });

  resetBtn.addEventListener('click', () => {
    camera.position.set(13, 10, 16);
    controls.target.set(0, 0, 0);
    mesh.rotation.set(0, 0, 0);
    controls.update();
  });

  function animate() {
    requestAnimationFrame(animate);
    if (autoRotate) {
      mesh.rotation.y += 0.006;
    }
    controls.update();
    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  let recorder = null;
  let chunks = [];

  function startRecording() {
    chunks = [];
    const stream = renderer.domElement.captureStream(30);
    const mimeType =
      window.MediaRecorder && MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';

    recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      downloadLink.href = url;
      downloadLink.download = `ai-cad-live-demo-${Date.now()}.webm`;
      downloadLink.style.display = 'inline-block';
      recorder = null;
    };
    recorder.start();
    recordBtn.textContent = 'Stop recording';
    statusEl.textContent = 'Recording live demo...';
  }

  function stopRecording() {
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }
    recordBtn.textContent = 'Record live demo';
    statusEl.textContent = 'Live demo video ready for download.';
  }

  function toggleRecording() {
    if (recorder && recorder.state === 'recording') {
      stopRecording();
    } else {
      startRecording();
    }
  }

  recordBtn.addEventListener('click', toggleRecording);

  setTimeout(() => {
    try {
      if (!recorder) {
        toggleRecording();
      }
      setTimeout(() => {
        if (recorder && recorder.state === 'recording') {
          toggleRecording();
        }
      }, 7000);
    } catch (error) {
      statusEl.textContent = 'Auto-record unavailable in this browser. Use Record live demo button.';
    }
  }, 1000);

  statusEl.textContent = 'Live animated demo running. Auto-record starting...';
}
</script>
</body>
</html>
"""


def threejs_viewer(mesh_data: Dict[str, Any], height: int = 580) -> None:
    mesh_json = json.dumps(mesh_data, separators=(",", ":"))
    html = VIEWER_HTML_TEMPLATE.replace("__MESH_JSON__", mesh_json)
    st.components.v1.html(html, height=height, scrolling=False)


def render_parameter_editor(spec: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(spec.get("parameters", {}))
    epoch = int(st.session_state.get("ui_epoch", 0))
    cols = st.columns(3)

    for index, (key, value) in enumerate(params.items()):
        col = cols[index % 3]
        widget_key = f"param_{spec['object_type']}_{key}_{epoch}"

        if key in COUNT_PARAMS:
            min_value = 1 if key == "teeth" else 0
            params[key] = col.number_input(
                label=key.replace("_", " ").title(),
                min_value=min_value,
                step=1,
                value=int(value),
                key=widget_key,
            )
        else:
            params[key] = col.number_input(
                label=key.replace("_", " ").title(),
                min_value=0.0,
                step=1.0,
                format="%.3f",
                value=float(value),
                key=widget_key,
            )

    return params


def ui_generate(spec: Dict[str, Any], event_type: str = "generation", instruction: Optional[str] = None) -> None:
    if instruction is None:
        instruction = st.session_state.get("instruction", "")
    else:
        st.session_state.instruction = instruction

    with st.spinner("Generating real CAD geometry..."):
        try:
            model, report = generate_model(spec)
            mesh_data = tessellate_shape(model.val())

            st.session_state.spec = spec
            st.session_state.model = model
            st.session_state.geometry_report = report
            st.session_state.mesh_data = mesh_data
            st.session_state.validation_issues = validate_spec(spec)
            st.session_state.step_bytes = None
            st.session_state.stl_bytes = None
            st.session_state.ui_epoch = int(st.session_state.get("ui_epoch", 0)) + 1

            log_event(
                event_type=event_type,
                success=True,
                generation_time_ms=report.get("generation_time_ms"),
                valid=report.get("valid"),
                constraint_satisfied=report.get("constraint_satisfied"),
            )

            save_revision(
                project_id=st.session_state.project_id,
                instruction=instruction,
                spec=spec,
                report=report,
            )

            set_flash("success", f"Generated valid {spec['object_type']} geometry in {report['generation_time_ms']:.1f} ms.")

        except Exception as exc:
            fallback_report = {
                "valid": False,
                "volume_mm3": 0.0,
                "surface_area_mm2": 0.0,
                "bounding_box_mm": {},
                "generation_time_ms": 0.0,
                "validation_errors": 1,
                "validation_warnings": 0,
                "constraint_satisfied": False,
                "error_message": f"{type(exc).__name__}: {exc}",
            }

            st.session_state.spec = spec
            st.session_state.model = None
            st.session_state.geometry_report = fallback_report
            st.session_state.mesh_data = None
            st.session_state.validation_issues = [
                Issue("error", "GENERATION_FAILURE", fallback_report["error_message"], None)
            ]
            st.session_state.step_bytes = None
            st.session_state.stl_bytes = None
            st.session_state.ui_epoch = int(st.session_state.get("ui_epoch", 0)) + 1

            log_event(
                event_type=event_type,
                success=False,
                generation_time_ms=0.0,
                valid=False,
                constraint_satisfied=False,
            )

            try:
                save_revision(
                    project_id=st.session_state.project_id,
                    instruction=instruction,
                    spec=spec,
                    report=fallback_report,
                )
            except Exception:
                pass

            set_flash("error", f"Geometry generation failed: {fallback_report['error_message']}")


def ui_parse_and_generate(instruction: str, generate: bool = True) -> None:
    st.session_state.instruction = instruction
    spec, parse_issues = parse_instruction(instruction)
    st.session_state.spec = spec
    st.session_state.parse_issues = parse_issues
    st.session_state.validation_issues = parse_issues + validate_spec(spec)

    if generate:
        ui_generate(spec, "generation", instruction=instruction)
    else:
        set_flash("info", "Requirement parsed into a structured specification. Press generate when ready.")


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(APP_NAME)
    st.caption(f"Single-file Streamlit prototype | build: {APP_BUILD} | local rule-based NL parser | no API key")

    if not CQ_AVAILABLE:
        st.error(
            "CadQuery could not be imported. Install dependencies with `pip install -r requirements.txt`. "
            f"Import error: {CQ_IMPORT_ERROR}"
        )

    init_db()

    if "events" not in st.session_state:
        st.session_state.events = []
    if "test_results" not in st.session_state:
        st.session_state.test_results = None
    if "ui_epoch" not in st.session_state:
        st.session_state.ui_epoch = 0
    if "instruction" not in st.session_state:
        st.session_state.instruction = DEFAULT_INSTRUCTION
    if "parse_issues" not in st.session_state:
        st.session_state.parse_issues = []
    if "validation_issues" not in st.session_state:
        st.session_state.validation_issues = []

    projects = list_projects()
    if not projects:
        project_id = create_project("AI CAD Demo Project")
        projects = list_projects()
    else:
        project_id = projects[0]["id"]

    if "project_id" not in st.session_state:
        st.session_state.project_id = project_id

    sync_instruction_state()

    if "spec" not in st.session_state:
        ui_parse_and_generate(st.session_state.instruction, generate=True)

    render_flash()

    unique_issues: List[Issue] = []

    with st.sidebar:
        st.header("Control panel")

        projects = list_projects()
        project_map = {project["id"]: project["name"] for project in projects}
        project_ids = list(project_map.keys())

        if st.session_state.project_id not in project_ids:
            st.session_state.project_id = project_ids[0]

        if "project_select" in st.session_state and st.session_state["project_select"] not in project_ids:
            del st.session_state["project_select"]

        selected_project = st.selectbox(
            "Project",
            project_ids,
            index=project_ids.index(st.session_state.project_id),
            format_func=lambda pid: project_map.get(pid, str(pid)),
            key="project_select",
        )
        st.session_state.project_id = selected_project

        new_project_name = st.text_input("Create new project", value="", key="new_project_name")
        if st.button("Create project"):
            if new_project_name.strip():
                created_id = create_project(new_project_name.strip())
                st.session_state.project_id = created_id
                st.rerun()
            else:
                st.warning("Enter a project name.")

        st.divider()
        st.subheader("Design instruction panel")

        instruction = st.text_area(
            "Natural-language requirement",
            value=st.session_state.instruction,
            height=130,
            key=INSTRUCTION_WIDGET_KEY,
        )
        st.session_state.instruction = instruction

        col_parse, col_generate = st.columns(2)
        if col_parse.button("Parse only"):
            ui_parse_and_generate(instruction, generate=False)
            render_flash()
        if col_generate.button("Parse + generate"):
            ui_parse_and_generate(instruction, generate=True)
            render_flash()

        st.subheader("Object template")
        object_types = list(OBJECT_SCHEMAS.keys())
        current_spec_for_template = st.session_state.get("spec") or default_spec("rectangular_block")
        current_object = current_spec_for_template.get("object_type", "rectangular_block")
        if current_object not in object_types:
            current_object = "rectangular_block"

        if "object_template_select" in st.session_state and st.session_state["object_template_select"] not in object_types:
            del st.session_state["object_template_select"]

        object_index = object_types.index(current_object)
        selected_object = st.selectbox(
            "Object type",
            object_types,
            index=object_index,
            format_func=lambda key: OBJECT_SCHEMAS[key]["label"],
            key="object_template_select",
        )

        if st.button("Load selected template"):
            template_spec = default_spec(selected_object)
            template_instruction = f"Template: {OBJECT_SCHEMAS[selected_object]['label']}"
            st.session_state.pending_instruction = template_instruction
            ui_generate(template_spec, "generation", instruction=template_instruction)
            st.rerun()

        st.divider()
        st.subheader("Parameter editor")

        current_spec = st.session_state.get("spec") or default_spec("rectangular_block")
        edited_params = render_parameter_editor(current_spec)

        if st.button("Apply parameter changes and regenerate"):
            modified_spec = normalize_spec(dict(current_spec))
            modified_spec["parameters"] = edited_params
            modified_spec["metadata"] = dict(modified_spec.get("metadata", {}))
            modified_spec["metadata"]["modified_at"] = utc_now()
            ui_generate(modified_spec, "parameter_modification", instruction=st.session_state.instruction)
            st.rerun()

        st.divider()
        st.subheader("Validation messages")

        all_issues = list(st.session_state.get("parse_issues", [])) + list(st.session_state.get("validation_issues", []))
        seen = set()
        unique_issues = []
        for issue in all_issues:
            marker = (issue.severity, issue.code, issue.message, issue.parameter)
            if marker not in seen:
                seen.add(marker)
                unique_issues.append(issue)
        render_issues(unique_issues)

        st.divider()
        st.subheader("Model generation history")

        revisions = list_revisions(st.session_state.project_id)
        if revisions:
            labels = [
                f"r{rev['revision_no']} | {rev['object_type']} | valid={bool(rev['valid'])} | {rev['created_at'][:19]}"
                for rev in revisions
            ]

            if "revision_select" in st.session_state and st.session_state["revision_select"] >= len(labels):
                del st.session_state["revision_select"]

            selected_revision_index = st.selectbox(
                "Revision",
                range(len(labels)),
                format_func=lambda idx: labels[idx],
                key="revision_select",
            )
            if st.button("Load selected revision"):
                selected_revision = revisions[selected_revision_index]
                loaded_spec = json.loads(selected_revision["spec_json"])
                rev_instruction = selected_revision.get("instruction") or ""
                st.session_state.pending_instruction = rev_instruction
                ui_generate(loaded_spec, "generation", instruction=rev_instruction)
                st.rerun()
        else:
            st.info("No revisions saved yet.")

        st.divider()
        st.subheader("Export controls")

        model = st.session_state.get("model")
        if model is not None:
            if st.button("Prepare STEP export"):
                try:
                    step_bytes = export_geometry(model, "STEP")
                    st.session_state.step_bytes = step_bytes
                    log_event("export", True)
                    set_flash("success", "STEP export prepared.")
                except Exception as exc:
                    st.session_state.step_bytes = None
                    log_event("export", False)
                    set_flash("error", f"STEP export failed: {exc}")

            if st.session_state.get("step_bytes"):
                st.download_button(
                    "Download STEP",
                    data=st.session_state.step_bytes,
                    file_name=f"ai_cad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.step",
                    mime="application/step",
                )

            if st.button("Prepare STL export"):
                try:
                    stl_bytes = export_geometry(model, "STL")
                    st.session_state.stl_bytes = stl_bytes
                    log_event("export", True)
                    set_flash("success", "STL export prepared.")
                except Exception as exc:
                    st.session_state.stl_bytes = None
                    log_event("export", False)
                    set_flash("error", f"STL export failed: {exc}")

            if st.session_state.get("stl_bytes"):
                st.download_button(
                    "Download STL",
                    data=st.session_state.stl_bytes,
                    file_name=f"ai_cad_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.stl",
                    mime="model/stl",
                )

            st.download_button(
                "Download specification JSON",
                data=spec_to_json_bytes(st.session_state.spec),
                file_name="ai_cad_specification.json",
                mime="application/json",
            )

            st.download_button(
                "Download validation report CSV",
                data=issues_to_csv_bytes(unique_issues),
                file_name="ai_cad_validation_report.csv",
                mime="text/csv",
            )
        else:
            st.info("Generate a valid model before exporting.")

        st.divider()
        st.subheader("Automated testing")

        if st.button("Run test suite"):
            with st.spinner("Executing pytest-compatible tests..."):
                metrics, dataframe = run_test_suite()
                st.session_state.test_results = (metrics, dataframe)

        if st.session_state.test_results:
            metrics, dataframe = st.session_state.test_results
            st.write(f"Passed: {metrics['passed']} / {metrics['total_tests']}")
            st.write(f"Duration: {metrics['duration_ms']:.1f} ms")
            st.dataframe(dataframe, use_container_width=True)
            st.download_button(
                "Download test report JSON",
                data=json.dumps(metrics, indent=2, default=str).encode("utf-8"),
                file_name="ai_cad_test_report.json",
                mime="application/json",
            )

    tab_preview, tab_spec, tab_validation, tab_metrics, tab_tests, tab_docs = st.tabs(
        ["3D Preview", "Specification", "Validation", "Metrics", "Tests", "Architecture / Docs"]
    )

    with tab_preview:
        st.subheader("Interactive 3D viewport")
        st.caption("Orbit with mouse. The demo auto-rotates and attempts to record a short WebM live video.")
        mesh_data = st.session_state.get("mesh_data")
        if mesh_data:
            threejs_viewer(mesh_data)
        else:
            st.warning("No tessellated geometry is available. Generate a model first.")

    with tab_spec:
        st.subheader("Structured JSON specification")
        st.json(st.session_state.get("spec", {}))

    with tab_validation:
        st.subheader("Validation engine output")
        render_issues(unique_issues)

        report = st.session_state.get("geometry_report")
        if report:
            st.subheader("Geometry report")
            st.json(report)

    with tab_metrics:
        st.subheader("Evaluation metrics from executed session events")
        metrics = compute_metrics(st.session_state.events)
        render_metrics(metrics)

        if st.session_state.events:
            st.subheader("Event log")
            st.dataframe(pd.DataFrame(st.session_state.events), use_container_width=True)
        else:
            st.info("No executed events yet.")

    with tab_tests:
        st.subheader("Automated test system")
        st.markdown(
            """
            This app includes pytest-compatible tests. You can run them from the sidebar button or from a terminal:

            ```bash
            pytest app.py -v
            ```

            Tests cover:

            - invalid dimensions
            - missing parameters
            - incorrect or missing units
            - geometry failures
            - invalid solids
            - export failures
            - parameter regeneration
            - repeatability
            """
        )

        if st.session_state.test_results:
            metrics, dataframe = st.session_state.test_results
            st.subheader("Latest executed test report")
            st.json(metrics)
            st.dataframe(dataframe, use_container_width=True)
        else:
            st.info("No test report has been executed in this session yet.")

    with tab_docs:
        st.subheader("Architecture and delivery notes")
        st.markdown(ARCHITECTURE_DOC)

        st.subheader("Setup commands")
        st.code(
            """
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
pytest app.py -v
            """.strip(),
            language="bash",
        )

        st.subheader("Milestone roadmap")
        st.markdown(
            """
            - Milestone 1: parametric rectangular block.
            - Milestone 2: cylinders, holes, mounting plates.
            - Milestone 3: structured specification parsing.
            - Milestone 4: interactive 3D viewer.
            - Milestone 5: CAD validation and export.
            - Milestone 6: natural-language instruction processing.
            - Milestone 7: project management and commercial user accounts.

            This single-file prototype contains code paths for Milestones 1–6 and a local persistence foundation for Milestone 7.
            Features should be considered verified only after executing the test suite on your machine.
            """
        )


if __name__ == "__main__":
    main()