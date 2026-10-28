"""Size charts as text: read a store's chart, map a body measurement to a size, and write size tasks around it.

Why this detail. A shopper usually knows a measurement, not the store's size: "my chest is 102 cm" fits an L in one
brand and an XL in another. A text-only agent can turn that into the right variant only when the chart reaches it as
text (a table in the description, a metafield, a policy page), never as an image. That is clothing's first checklist
hypothesis (clothing-adult.size-chart-text), and the same holds for foot length in footwear, height in kids' wear and
body measurements in sportswear.

Reading. Charts come as HTML tables, markdown or tab tables, "S: chest 86-91 cm" lines, or a structured metafield.
Sizes may run down the first column or across the first row. Units come from the header ("Chest (cm)"), the cell
("34-36 in"), a note such as "all measurements in inches", or, failing those, the plausible range for the measure.
Values are kept in whole millimetres (Decimal arithmetic, rounded half up), so the same chart always gives the same
sizes. A range cell covers its range; a column of single values reads as "up to": a size fits measurements above
the previous size's value and up to its own.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
from random import Random
from typing import Any

from pydantic import JsonValue

from carttrace.agents.details.base import Proposal
from carttrace.agents.memo import per_product
from carttrace.agents.products import Grid, Product, Variant
from carttrace.agents.spec import Op
from carttrace.agents.text import fold, has_phrase

Range = tuple[int, int]
"""(low, high) in millimetres; equal for a single value."""

MEASURES: dict[str, tuple[str, ...]] = {
    "chest": ("chest", "pit to pit", "pit-to-pit"),
    "bust": ("bust",),
    "under_bust": ("under bust", "underbust", "under-bust"),
    "waist": ("waist",),
    "hips": ("hips", "hip", "seat"),
    "inseam": ("inseam", "inside leg", "inner leg", "leg length"),
    "foot": ("foot length", "foot", "heel to toe", "insole length", "insole"),
    "height": ("height", "body height"),
    "neck": ("neck", "collar"),
    "sleeve": ("sleeve", "arm length"),
    "head": ("head circumference", "head"),
    "shoulder": ("shoulders", "shoulder"),
    "length": ("garment length", "length"),
    "thigh": ("thigh",),
}
"""Each measure's names in chart headers, as phrases."""
TASK_MEASURES = ("chest", "bust", "waist", "hips", "foot", "height", "inseam", "neck", "head")
"""Body measurements a shopper knows about themselves, in the order a task prefers them."""
_SAY = {
    "chest": "my chest measures {amount} {unit}",
    "bust": "my bust measures {amount} {unit}",
    "waist": "my waist measures {amount} {unit}",
    "hips": "my hips measure {amount} {unit}",
    "foot": "my foot is {amount} {unit} long",
    "height": "I'm {amount} {unit} tall",
    "inseam": "my inside leg measures {amount} {unit}",
    "neck": "my neck measures {amount} {unit}",
    "head": "my head measures {amount} {unit} around",
}
_UNIT_WORDS = {"cm": "cm", "mm": "mm", "in": "inches"}
_SLOT_NAMES = {"foot": "foot length", "inseam": "inside leg", "head": "head circumference"}
_PER_UNIT_MM = {"cm": Decimal(10), "mm": Decimal(1), "in": Decimal("25.4")}
_STEP = {"cm": Decimal(1), "mm": Decimal(5), "in": Decimal("0.5")}
"""How finely a shopper states a measurement in each unit."""
_PLAUSIBLE_INCHES_MAX = {
    "chest": 60,
    "bust": 60,
    "under_bust": 50,
    "waist": 60,
    "hips": 64,
    "inseam": 40,
    "foot": 13.5,
    "neck": 21,
    "head": 26,
    "sleeve": 40,
    "shoulder": 24,
    "thigh": 34,
}
"""Without a unit, a column whose largest value is at most this many is read as inches, otherwise as centimetres.
Heights and lengths have no safe guess and need a stated unit."""

_ALPHA_WORDS = (
    ("extra extra large", "xxl"),
    ("xx-large", "xxl"),
    ("xx large", "xxl"),
    ("2x-large", "xxl"),
    ("extra large", "xl"),
    ("x-large", "xl"),
    ("x large", "xl"),
    ("extra small", "xs"),
    ("x-small", "xs"),
    ("x small", "xs"),
    ("small", "s"),
    ("medium", "m"),
    ("large", "l"),
    ("one size", "os"),
)
_SYSTEM_WORDS = re.compile(r"\b(?:us|uk|eu|it|fr|jp|au|mx|cn|mens|men's|womens|women's|size|sz)\b\.?")
_NUMBER_X = re.compile(r"^([2-6])x([sl])$")
_ALPHA = re.compile(r"^x{0,5}[sl]$|^m$|^os$")
_NUMERIC = re.compile(r"^\d{1,3}(?:\.5)?$")
_KIDS = re.compile(r"^(\d{1,2})(?:-(\d{1,2}))?(y|yrs|years|m|mths|months|t)$")
_NUM = r"\d+(?:[.,]\d+)?"
_RANGE = re.compile(rf"({_NUM})\s*(?:\"|''|cm|mm|in(?:ch(?:es)?)?\.?)?\s*(?:-|to|/)\s*({_NUM})")
_SINGLE = re.compile(rf"({_NUM})")
_TEXT_ROW = re.compile(r"^(?:size\s+)?([a-z0-9.\- ]{1,14}?)\s*[:=(]\s*(.+)$")
_SIZE_HEADERS = ("size", "sizes", "us", "uk", "eu", "it", "fr", "jp", "au", "mx", "label")
SIZE_CHART_KEYS = ("size chart", "size guide", "sizechart", "sizing", "size table", "fit guide")
"""Words in a metafield's name that say it holds a size chart."""


@dataclass(frozen=True, slots=True)
class SizeRow:
    labels: tuple[str, ...]
    """The size as each size column of the chart names it, e.g. ("M",) or ("9", "8", "42")."""
    keys: tuple[str | None, ...]
    """`size_key` of each label: what variant options are matched against."""
    ranges: Mapping[str, Range]
    """Measure to its range in millimetres."""


@dataclass(frozen=True, slots=True)
class SizeChart:
    rows: tuple[SizeRow, ...]
    units: Mapping[str, str]
    """Measure to the unit the chart states it in (cm, mm or in), for wording a shopper's measurement."""

    def measures(self) -> list[str]:
        return [m for m in MEASURES if any(m in row.ranges for row in self.rows)]

    def column_for(self, keys: Collection[str]) -> int:
        """The size column whose keys match the most of `keys` (a product's variant sizes); the first on a tie."""
        width = max((len(row.keys) for row in self.rows), default=1)
        hits = [sum(1 for row in self.rows if i < len(row.keys) and row.keys[i] in keys) for i in range(width)]
        return max(range(width), key=lambda i: (hits[i], -i))

    def fitting(self, measure: str, mm: int) -> list[int]:
        """Indexes of the rows whose `measure` fits a body measurement of `mm` millimetres."""
        rows = [(i, row.ranges[measure]) for i, row in enumerate(self.rows) if measure in row.ranges]
        if not rows:
            return []
        if all(low == high for _, (low, high) in rows):
            return _up_to(rows, mm)
        return [i for i, (low, high) in rows if low <= mm <= high]


def _up_to(rows: list[tuple[int, Range]], mm: int) -> list[int]:
    ordered = sorted(rows, key=lambda r: r[1][0])
    values = [r[1][0] for r in ordered]
    for position, (index, (value, _)) in enumerate(ordered):
        if position == 0:
            gap = values[1] - values[0] if len(values) > 1 else value
            floor = value - gap
        else:
            floor = values[position - 1]
        if floor < mm <= value:
            return [index]
    return []


@lru_cache(maxsize=1 << 15)
def size_key(label: str) -> str | None:
    """A size label in one comparable form: "Medium" and "M" give "m", "UK 10" and "10" give "10", "2XL" gives
    "xxl", "3-4 Years" gives "3-4y". None when the text is not a size. Remembered: a pure function of the label."""
    text = fold(label).strip(" :.")
    for phrase, key in _ALPHA_WORDS:
        if text in (phrase, f"size {phrase}"):
            return key
    text = _SYSTEM_WORDS.sub(" ", text).strip(" :.-")
    text = text.replace(" ", "")
    number_x = _NUMBER_X.match(text)
    if number_x:
        return "x" * int(number_x.group(1)) + number_x.group(2)
    if _ALPHA.match(text) or _NUMERIC.match(text):
        return text
    kids = _KIDS.match(text)
    if kids:
        unit = {"y": "y", "yrs": "y", "years": "y", "m": "m", "mths": "m", "months": "m", "t": "t"}[kids.group(3)]
        span = kids.group(1) + (f"-{kids.group(2)}" if kids.group(2) else "")
        return span + unit
    return None


def variant_size(variant: Variant) -> str | None:
    """The `size_key` of a variant's size option (any option whose name mentions size)."""
    for name, value in variant.options:
        if "size" in fold(name) or fold(name) in ("taille", "größe", "talla", "taglia"):
            return size_key(value)
    return None


def measure_of(header: str) -> str | None:
    """The measure a chart header names: "Chest (cm)" gives chest, "Inside leg" gives inseam."""
    text = fold(header)
    found = [(len(name), measure) for measure, names in MEASURES.items() for name in names if has_phrase(text, name)]
    return max(found)[1] if found else None


def unit_in(text: str) -> str | None:
    folded = fold(text)
    if re.search(r"\bcm\b|centimet", folded):
        return "cm"
    if re.search(r"\bmm\b|millimet", folded):
        return "mm"
    if re.search(r"\binch|\bins?\b\.?|\"|''", folded):
        return "in"
    return None


def to_mm(amount: Decimal, unit: str) -> int:
    return int((amount * _PER_UNIT_MM[unit]).to_integral_value(rounding=ROUND_HALF_UP))


def from_mm(mm: int, unit: str) -> Decimal:
    step = _STEP[unit]
    exact = Decimal(mm) / _PER_UNIT_MM[unit]
    return (exact / step).to_integral_value(rounding=ROUND_HALF_UP) * step


def show(amount: Decimal) -> str:
    """2 -> "2", 34.50 -> "34.5"."""
    text = format(amount.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_value(text: str) -> tuple[Decimal, Decimal, str | None] | None:
    """(low, high, unit) of a chart cell such as "86-91", "34 - 36 in", '35"' or "92 cm". In a cell that gives
    both units ("86-91 cm / 34-36 in") the first value is read with the unit written right after it."""
    folded = re.sub(r"(\d),(\d{1,2})(?!\d)", r"\1.\2", fold(text))
    found = _RANGE.search(folded) or _SINGLE.search(folded)
    if found is None:
        return None
    try:
        low = Decimal(found.group(1))
        high = Decimal(found.group(2)) if found.re is _RANGE else low
    except InvalidOperation:
        return None
    after = folded[found.end() : found.end() + 12]
    unit = unit_in(after.split("/")[0]) or (unit_in(folded) if _one_unit(folded) else None)
    return (min(low, high), max(low, high), unit)


def _one_unit(text: str) -> bool:
    return len({u for u in (unit_in(part) for part in re.split(r"[/|(),]", text)) if u}) <= 1


# -- reading a product's chart ---------------------------------------------------------------------------------------


def read_chart(product: Product) -> SizeChart | None:
    """The product's size chart, from a structured metafield first, then its tables, then "S: chest 86-91 cm"
    lines. None when it has no chart a text-only agent could read. Read once per product within a memo run
    (carttrace.agents.memo)."""
    return per_product("size_chart", product, None, lambda: _read_chart(product))


def _read_chart(product: Product) -> SizeChart | None:
    default_unit = _note_unit(product.lines)
    candidates: list[SizeChart | None] = [_from_metadata(product.metadata, default_unit)]
    candidates += [_from_table(table, default_unit) for table in product.tables]
    candidates.append(_from_lines(product.lines, default_unit))
    return next((c for c in candidates if c is not None), None)


def _note_unit(lines: Iterable[str]) -> str | None:
    """A unit the page states for its charts, e.g. "All measurements are in inches"."""
    for line in lines:
        folded = fold(line)
        if ("measure" in folded or "size" in folded) and re.search(r"\b(?:in|are)\s+(?:cm|centimet|inch|mm)", folded):
            return unit_in(folded.split(" in ", 1)[-1]) or unit_in(folded)
    return None


def _from_table(table: Grid, default_unit: str | None) -> SizeChart | None:
    rows = [list(r) for r in table if any(c.strip() for c in r)]
    for start in range(min(3, max(0, len(rows) - 2)) + 1):  # a title row may sit above the header
        chart = _sizes_down(rows[start:], default_unit) or _sizes_across(rows[start:], default_unit)
        if chart is not None:
            return chart
    return None


def _sizes_down(rows: list[list[str]], default_unit: str | None) -> SizeChart | None:
    if len(rows) < 3:
        return None
    header = rows[0]
    measures = {i: measure_of(cell) for i, cell in enumerate(header)}
    measure_cols = {i: m for i, m in measures.items() if m is not None}
    size_cols = [i for i, cell in enumerate(header) if i not in measure_cols and _is_size_header(cell)]
    if not size_cols and header and 0 not in measure_cols:
        size_cols = [0]
    if not measure_cols or not size_cols:
        return None
    raw: list[tuple[list[str], dict[str, tuple[Decimal, Decimal, str | None]]]] = []
    for row in rows[1:]:
        labels = [row[i] if i < len(row) else "" for i in size_cols]
        if not any(size_key(label) for label in labels):
            continue
        values = {}
        for i, measure in measure_cols.items():
            parsed = parse_value(row[i]) if i < len(row) else None
            if parsed is not None and measure not in values:
                values[measure] = parsed
        if values:
            raw.append((labels, values))
    header_units = {m: unit_in(header[i]) for i, m in measure_cols.items()}
    return _chart(raw, header_units, default_unit)


def _sizes_across(rows: list[list[str]], default_unit: str | None) -> SizeChart | None:
    if len(rows) < 2:
        return None
    header = rows[0]
    labels = header[1:]
    if sum(1 for label in labels if size_key(label)) < 2:
        return None
    per_size: list[dict[str, tuple[Decimal, Decimal, str | None]]] = [{} for _ in labels]
    header_units: dict[str, str | None] = {}
    for row in rows[1:]:
        measure = measure_of(row[0]) if row else None
        if measure is None:
            continue
        header_units.setdefault(measure, unit_in(row[0]))
        for j, cell in enumerate(row[1 : len(labels) + 1]):
            parsed = parse_value(cell)
            if parsed is not None and measure not in per_size[j]:
                per_size[j][measure] = parsed
    raw = [([label], values) for label, values in zip(labels, per_size, strict=True) if size_key(label) and values]
    return _chart(raw, header_units, default_unit)


Cells = dict[str, tuple[Decimal, Decimal, str | None]]


def text_row(line: str) -> tuple[str, Cells] | None:
    """A line that reads as one row of a size chart written as text ("S: chest 86-91 cm, waist 71-76 cm"): its size
    label and the measures it gives, or None when the line is not one. Hiding a chart (carttrace.collect) removes exactly
    the lines read here."""
    found = _TEXT_ROW.match(fold(line))
    if not found or size_key(found.group(1)) is None:
        return None
    values: Cells = {}
    for part in re.split(r"[,;|]", found.group(2)):
        measure = measure_of(part)
        parsed = parse_value(part)
        if measure is not None and parsed is not None and measure not in values:
            values[measure] = parsed
    return (found.group(1).strip(), values) if values else None


def _from_lines(lines: Sequence[str], default_unit: str | None) -> SizeChart | None:
    raw: list[tuple[list[str], Cells]] = []
    for line in lines:
        row = text_row(line)
        if row is not None:
            raw.append(([row[0]], row[1]))
    return _chart(raw, {}, default_unit) if len(raw) >= 2 else None


def _from_metadata(metadata: Mapping[str, Any], default_unit: str | None) -> SizeChart | None:
    """A chart stored as a structured metafield: a list of {"size": "M", "chest_cm": "91-97"} rows, or a mapping
    of size to its measurements."""
    for key in sorted(metadata):
        name = fold(str(key).replace("_", " ").replace("-", " "))
        if any(k in name for k in SIZE_CHART_KEYS):
            chart = _structured_chart(_entries(metadata[key]), default_unit)
            if chart is not None:
                return chart
    return None


def _entries(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [(str(k), v) for k, v in value.items()]
    entries = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, Mapping):
            label = next((item[k] for k in ("size", "label", "name") if isinstance(item.get(k), str)), None)
            if label is not None:
                entries.append((label, item))
    return entries


def _structured_chart(entries: list[tuple[str, Any]], default_unit: str | None) -> SizeChart | None:
    raw: list[tuple[list[str], Cells]] = []
    header_units: dict[str, str | None] = {}
    for label, item in entries:
        values: Cells = {}
        for field_name, cell in item.items() if isinstance(item, Mapping) else []:
            header = str(field_name).replace("_", " ")
            measure = measure_of(header)
            parsed = _structured_value(cell) if measure is not None else None
            if measure is None or parsed is None or measure in values:
                continue
            header_units.setdefault(measure, unit_in(header))
            values[measure] = parsed
        if size_key(label) and values:
            raw.append(([label], values))
    return _chart(raw, header_units, default_unit)


def _structured_value(cell: Any) -> tuple[Decimal, Decimal, str | None] | None:
    if isinstance(cell, list) and len(cell) == 2 and all(isinstance(v, int | float) for v in cell):
        low, high = (Decimal(str(v)) for v in cell)
        return (min(low, high), max(low, high), None)
    if isinstance(cell, int | float) and not isinstance(cell, bool):
        value = Decimal(str(cell))
        return (value, value, None)
    return parse_value(cell) if isinstance(cell, str) else None


def _chart(
    raw: list[tuple[list[str], dict[str, tuple[Decimal, Decimal, str | None]]]],
    header_units: Mapping[str, str | None],
    default_unit: str | None,
) -> SizeChart | None:
    if len(raw) < 2:
        return None
    units: dict[str, str] = {}
    for measure in MEASURES:
        cells = [values[measure] for _, values in raw if measure in values]
        if len(cells) < 2:
            continue
        unit = header_units.get(measure) or next((c[2] for c in cells if c[2]), None) or default_unit
        unit = unit or _guess_unit(measure, [c[1] for c in cells])
        if unit is not None:
            units[measure] = unit
    if not units:
        return None
    rows = []
    for labels, values in raw:
        ranges = {
            m: (to_mm(low, units[m]), to_mm(high, units[m])) for m, (low, high, _) in values.items() if m in units
        }
        if ranges:
            rows.append(SizeRow(tuple(labels), tuple(size_key(label) for label in labels), ranges))
    return SizeChart(tuple(rows), units) if len(rows) >= 2 else None


def _guess_unit(measure: str, highs: list[Decimal]) -> str | None:
    limit = _PLAUSIBLE_INCHES_MAX.get(measure)
    if limit is None or not highs:
        return None
    return "in" if max(highs) <= Decimal(str(limit)) else "cm"


def _is_size_header(cell: str) -> bool:
    folded = fold(cell)
    return any(has_phrase(folded, word) for word in _SIZE_HEADERS) or folded == ""


# -- the detail ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SizeChartDetail:
    """The size chart decides: the shopper states a measurement, and only the chart says which size fits it."""

    name: str = "size_chart"
    field: str = "measurements"
    op: Op = "fits"
    missing: str = "size_chart"
    forbidden_slots: frozenset[str] = frozenset({"size", "size_system", "fit"})
    hypotheses: tuple[str, ...] = ("size-chart-text", "foot-length-chart", "kids-size-chart", "sizing-by-body")

    def applies(self, fields: Collection[str]) -> bool:
        return "measurements" in fields and "size" in fields

    def propose(self, target: Product, variant: Variant, shelf: Sequence[Product], rng: Random) -> Proposal | None:
        """A body measurement that the target's own chart maps to this variant's size and to no other size."""
        chart = read_chart(target)
        key = variant_size(variant)
        if chart is None or key is None:
            return None
        column = chart.column_for({k for v in target.variants if (k := variant_size(v))})
        for measure in [m for m in TASK_MEASURES if m in chart.measures()]:
            row = next((i for i, r in enumerate(chart.rows) if _key_at(r, column) == key), None)
            if row is None or measure not in chart.rows[row].ranges:
                continue
            amount = _inside(chart, row, measure)
            if amount is not None:
                return _proposal(measure, amount, chart.units[measure])
        return None

    def impossible(self, pool: Sequence[Product], rng: Random) -> Proposal | None:
        charts = [c for c in (read_chart(p) for p in pool) if c is not None]
        for measure in TASK_MEASURES:
            highs = [row.ranges[measure][1] for chart in charts for row in chart.rows if measure in row.ranges]
            if not highs:
                continue
            unit = next(chart.units[measure] for chart in charts if measure in chart.units)
            amount = from_mm(max(highs) * 115 // 100, unit)
            mm = to_mm(amount, unit)
            if all(not chart.fitting(measure, mm) for chart in charts):
                return _proposal(measure, amount, unit)
        return None

    def check(self, value: JsonValue, product: Product, variant: Variant) -> bool | None:
        wanted = measurement_of(value)
        chart = read_chart(product)
        key = variant_size(variant)
        if wanted is None or chart is None or key is None:
            return None
        measure, mm = wanted
        if measure not in chart.measures():
            return None
        column = chart.column_for({k for v in product.variants if (k := variant_size(v))})
        return key in {_key_at(chart.rows[i], column) for i in chart.fitting(measure, mm)}


def _key_at(row: SizeRow, column: int) -> str | None:
    if column < len(row.keys) and row.keys[column]:
        return row.keys[column]
    return next((k for k in row.keys if k), None)


def _inside(chart: SizeChart, row: int, measure: str) -> Decimal | None:
    """A measurement, in the chart's unit and a shopper's precision, that fits this row and no other."""
    unit = chart.units[measure]
    low, high = chart.rows[row].ranges[measure]
    singles = all(r.ranges[measure][0] == r.ranges[measure][1] for r in chart.rows if measure in r.ranges)
    if singles:
        values = sorted(r.ranges[measure][0] for r in chart.rows if measure in r.ranges)
        position = values.index(low)
        below = values[position - 1] if position else low - (values[1] - values[0] if len(values) > 1 else low)
        tries = [(below + low) // 2, low]
    else:
        tries = [(low + high) // 2, low + (high - low) // 3, high - (high - low) // 3]
    for mm in tries:
        amount = from_mm(mm, unit)
        if chart.fitting(measure, to_mm(amount, unit)) == [row]:
            return amount
    return None


def _proposal(measure: str, amount: Decimal, unit: str) -> Proposal:
    say = _SAY[measure].format(amount=show(amount), unit=_UNIT_WORDS[unit])
    sentence = f"{say[0].upper()}{say[1:]}, but I don't know my size in your sizing."
    value: dict[str, JsonValue] = {"measure": measure, "amount": show(amount), "unit": unit}
    slot = f"{_SLOT_NAMES.get(measure, measure)} {show(amount)} {unit}"
    return Proposal(value=value, say=say, sentence=sentence, slot=slot, aliases=("size", "size_system", "fit"))


def measurement_of(value: JsonValue) -> tuple[str, int] | None:
    """(measure, millimetres) of a `fits` predicate's value {"measure", "amount", "unit"}; None when malformed."""
    if not isinstance(value, Mapping):
        return None
    measure, amount, unit = value.get("measure"), value.get("amount"), value.get("unit")
    if not isinstance(measure, str) or unit not in _PER_UNIT_MM or not isinstance(amount, str | int | float):
        return None
    try:
        return measure, to_mm(Decimal(str(amount)), str(unit))
    except InvalidOperation:
        return None
