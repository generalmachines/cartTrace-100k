"""A standard-library reader for .xlsx workbooks: enough of SpreadsheetML to snapshot the team sheet.

Why our own reader. The store queue lives in a Google Sheet, and the snapshot in `data/stores/` must be reproducible
byte for byte from the exported workbook. A third-party reader would be one more dependency deciding how dates,
numbers and rich text become strings; here that decision is written down once:

- shared strings and inline strings, with rich-text runs joined and phonetic runs dropped;
- formula cells as their cached result, booleans as TRUE or FALSE, error cells as their code (#N/A);
- numbers without float noise (at most 15 significant digits, the precision a spreadsheet shows), integers without a
  trailing ".0";
- date-formatted numbers as ISO dates (or ISO date-times when they carry a time of day), in the 1900 or 1904 system.

Merged ranges are reported, never expanded, so the caller decides what a merged title or header means. Safety: the
archive comes from the network, so every part is size-capped before it is decompressed, and any part that declares a
DTD is refused rather than parsed. SpreadsheetML never needs one, and refusing it rules out entity expansion.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from xml.etree import ElementTree

from carttrace.core.errors import NotFoundError, ValidationError

MAX_PART_BYTES = 64 * 1024 * 1024
"""Largest decompressed part we read. The team sheet's biggest part is under 1 MB; this only stops a zip bomb."""

_REL_ID_ATTRS = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
    "{http://purl.oclc.org/ooxml/officeDocument/relationships}id",
)
_CELL_REF = re.compile(r"^\$?([A-Za-z]{1,3})\$?([0-9]{1,7})$")
_BUILTIN_DATE_FORMATS = frozenset({*range(14, 23), *range(27, 37), 45, 46, 47, *range(50, 59)})
_QUOTED = re.compile(r'"[^"]*"')
_ESCAPED = re.compile(r"\\.")
_BRACKETED = re.compile(r"\[[^\]]*\]")
_DATE_TOKENS = re.compile(r"[dmyhs]", re.IGNORECASE)
_MAX_SERIAL = Decimal(2958465)
"""Serial of 9999-12-31, the last date a spreadsheet can hold."""


@dataclass(frozen=True, slots=True)
class CellRange:
    """A rectangle of cells, 1-based and inclusive, as a mergeCell ref such as "A2:L2" names it."""

    first_row: int
    first_col: int
    last_row: int
    last_col: int

    def contains(self, row: int, col: int) -> bool:
        return self.first_row <= row <= self.last_row and self.first_col <= col <= self.last_col

    @property
    def ref(self) -> str:
        return f"{column_letter(self.first_col)}{self.first_row}:{column_letter(self.last_col)}{self.last_row}"


@dataclass(frozen=True, slots=True)
class Sheet:
    """One tab: a dense grid of display strings up to the last non-empty row and column, plus its merged ranges."""

    name: str
    state: str
    """visible, hidden or veryHidden, as the workbook declares it."""
    rows: tuple[tuple[str, ...], ...]
    """rows[r - 1][c - 1] is cell (r, c); an empty cell is ""."""
    merges: tuple[CellRange, ...]

    @property
    def width(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    def cell(self, row: int, col: int) -> str:
        if 1 <= row <= len(self.rows) and 1 <= col <= self.width:
            return self.rows[row - 1][col - 1]
        return ""


@dataclass(frozen=True, slots=True)
class Workbook:
    sheets: tuple[Sheet, ...]
    date1904: bool

    def sheet(self, name: str) -> Sheet:
        for sheet in self.sheets:
            if sheet.name == name:
                return sheet
        raise NotFoundError(f"workbook has no tab named {name!r}", hint=f"tabs are {[s.name for s in self.sheets]}")


def column_letter(index: int) -> str:
    """1 -> "A", 26 -> "Z", 27 -> "AA"."""
    if index < 1:
        raise ValueError(f"column index starts at 1, got {index}")
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def column_index(letters: str) -> int:
    """ "A" -> 1, "Z" -> 26, "AA" -> 27."""
    if not letters or not letters.isascii() or not letters.isalpha():
        raise ValueError(f"not a column name: {letters!r}")
    index = 0
    for ch in letters.upper():
        index = index * 26 + ord(ch) - 64
    return index


def parse_ref(ref: str) -> tuple[int, int]:
    """ "B12" -> (12, 2): a cell reference as (row, column), both 1-based."""
    match = _CELL_REF.match(ref.strip())
    if not match:
        raise ValidationError(f"not a cell reference: {ref!r}")
    return int(match.group(2)), column_index(match.group(1))


def read_workbook(data: bytes) -> Workbook:
    """Parse every tab of an .xlsx workbook held in memory."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValidationError(
            "not an xlsx workbook: the bytes are not a zip archive",
            hint="export the sheet as Microsoft Excel (.xlsx), not as CSV or a web page",
        ) from exc
    with archive:
        package = _Package(archive)
        workbook_part = package.main_part()
        root = package.xml(workbook_part)
        ns = _namespace(root)
        rels = package.relationships(workbook_part)
        shared = _shared_strings(package, _rel_target(rels, "/sharedStrings", "xl/sharedStrings.xml", package))
        styles_part = _rel_target(rels, "/styles", "xl/styles.xml", package)
        date_styles = _date_styles(package, styles_part) if styles_part else frozenset()
        props = root.find(_q(ns, "workbookPr"))
        date1904 = props is not None and props.get("date1904", "").lower() in ("1", "true")
        sheets_el = root.find(_q(ns, "sheets"))
        sheets: list[Sheet] = []
        for el in sheets_el if sheets_el is not None else ():
            if _local(el.tag) != "sheet":
                continue
            name = el.get("name", "")
            rid = next((el.get(attr) for attr in _REL_ID_ATTRS if el.get(attr)), None)
            if rid is None or rid not in rels:
                raise ValidationError(f"tab {name!r} points at no worksheet part")
            _, part = rels[rid]
            cells = _CellReader(shared, date_styles, date1904)
            sheets.append(_read_sheet(package, part, name, el.get("state", "visible"), cells))
    return Workbook(sheets=tuple(sheets), date1904=date1904)


class _Package:
    """The zip archive, read through size caps and the DTD refusal."""

    def __init__(self, archive: zipfile.ZipFile) -> None:
        self._archive = archive
        self._names = frozenset(archive.namelist())

    def has(self, part: str) -> bool:
        return part in self._names

    def read(self, part: str) -> bytes:
        try:
            info = self._archive.getinfo(part)
        except KeyError as exc:
            raise ValidationError(f"workbook part {part} is missing", hint="re-export the workbook") from exc
        if info.file_size > MAX_PART_BYTES:
            raise ValidationError(f"workbook part {part} is {info.file_size} bytes, over the {MAX_PART_BYTES} cap")
        with self._archive.open(info) as fh:
            data = fh.read(MAX_PART_BYTES + 1)
        if len(data) > MAX_PART_BYTES:
            raise ValidationError(f"workbook part {part} decompresses past the {MAX_PART_BYTES} byte cap")
        return data

    def xml(self, part: str) -> ElementTree.Element:
        data = self.read(part)
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            raise ValidationError(
                f"workbook part {part} declares a DTD, which SpreadsheetML never needs; refusing to parse it",
                hint="re-export the workbook from Google Sheets or Excel",
            )
        try:
            return ElementTree.fromstring(data)  # noqa: S314 - DTDs are refused above, so no entity can expand
        except ElementTree.ParseError as exc:
            raise ValidationError(f"workbook part {part} is not well-formed XML: {exc}") from exc

    def relationships(self, part: str) -> dict[str, tuple[str, str]]:
        """Relationship id to (type, target part) for a part; the package's own when `part` is ""."""
        base = posixpath.dirname(part)
        rels_part = posixpath.join(base, "_rels", f"{posixpath.basename(part)}.rels")
        if not self.has(rels_part):
            return {}
        out: dict[str, tuple[str, str]] = {}
        for rel in self.xml(rels_part):
            if _local(rel.tag) != "Relationship" or rel.get("TargetMode") == "External":
                continue
            out[rel.get("Id", "")] = (rel.get("Type", ""), _resolve(base, rel.get("Target", "")))
        return out

    def main_part(self) -> str:
        for rel_type, target in self.relationships("").values():
            if rel_type.endswith("/officeDocument"):
                return target
        return "xl/workbook.xml"


@dataclass(frozen=True, slots=True)
class _CellReader:
    shared: tuple[str, ...]
    date_styles: frozenset[int]
    date1904: bool

    def text(self, cell: ElementTree.Element, ns: str) -> str:
        kind = cell.get("t", "n")
        if kind == "inlineStr":
            inline = cell.find(_q(ns, "is"))
            return _rich_text(inline) if inline is not None else ""
        value_el = cell.find(_q(ns, "v"))
        raw = value_el.text if value_el is not None and value_el.text is not None else ""
        return self._typed(kind, raw, _int_attr(cell, "s", 0))

    def _typed(self, kind: str, raw: str, style: int) -> str:
        if kind == "s":
            return self._shared(raw)
        if kind == "b":
            return {"1": "TRUE", "0": "FALSE"}.get(raw.strip(), raw)
        if kind in ("str", "e", "d") or not raw.strip():
            return raw if kind in ("str", "e", "d") else ""
        return _serial_text(raw, self.date1904) if style in self.date_styles else number_text(raw)

    def _shared(self, raw: str) -> str:
        if not raw.strip():
            return ""
        try:
            return self.shared[int(raw)]
        except (ValueError, IndexError) as exc:
            raise ValidationError(f"cell points at shared string {raw!r}, which the workbook does not have") from exc


def number_text(raw: str) -> str:
    """A stored number as a spreadsheet shows it: "46306.0" -> "46306", "4.4000000000000004" -> "4.4"."""
    try:
        value = Decimal(raw.strip())
    except InvalidOperation:
        return raw
    if not value.is_finite():
        return raw
    if value.is_zero():
        return "0"
    if len(value.as_tuple().digits) > 15:
        try:
            value = value.quantize(Decimal(1).scaleb(value.adjusted() - 14), rounding=ROUND_HALF_EVEN)
        except InvalidOperation:
            return raw
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _serial_text(raw: str, date1904: bool) -> str:
    """A date serial as an ISO date, or an ISO date-time when it carries a time of day.

    The 1900 system counts from 1899-12-30, which is right for every serial from 61 (1900-03-01) on; earlier serials
    inherit the old Lotus leap-year bug and are not dates anyone types into a store queue.
    """
    try:
        serial = Decimal(raw.strip())
    except InvalidOperation:
        return raw
    if not serial.is_finite() or serial < 0 or serial > _MAX_SERIAL:
        return number_text(raw)
    base = datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)
    days = int(serial.to_integral_value(rounding=ROUND_FLOOR))
    seconds = int(((serial - days) * 86400).to_integral_value(rounding=ROUND_HALF_UP))
    moment = base + timedelta(days=days, seconds=seconds)
    return moment.date().isoformat() if moment.time() == time(0) else moment.isoformat(timespec="seconds")


def _is_date_format(code: str) -> bool:
    """Whether a number format code shows a date or a time. Quoted text, escapes and [brackets] are not tokens."""
    section = code.split(";", 1)[0]
    cleaned = _BRACKETED.sub("", _ESCAPED.sub("", _QUOTED.sub("", section)))
    return bool(_DATE_TOKENS.search(cleaned))


def _read_sheet(package: _Package, part: str, name: str, state: str, cells: _CellReader) -> Sheet:
    root = package.xml(part)
    ns = _namespace(root)
    values: dict[tuple[int, int], str] = {}
    data = root.find(_q(ns, "sheetData"))
    row_no = 0
    for row in data if data is not None else ():
        if _local(row.tag) != "row":
            continue
        row_no = _int_attr(row, "r", row_no + 1)
        col_no = 0
        for cell in row:
            if _local(cell.tag) != "c":
                continue
            ref = cell.get("r")
            if ref:
                row_no, col_no = parse_ref(ref)
            else:
                col_no += 1
            text = cells.text(cell, ns)
            if text != "":
                values[(row_no, col_no)] = text
    merges: list[CellRange] = []
    merge_el = root.find(_q(ns, "mergeCells"))
    for merge in merge_el if merge_el is not None else ():
        ref = merge.get("ref", "")
        if _local(merge.tag) == "mergeCell" and ref:
            merges.append(_parse_range(ref))
    height = max((r for r, _ in values), default=0)
    width = max((c for _, c in values), default=0)
    grid = tuple(tuple(values.get((r, c), "") for c in range(1, width + 1)) for r in range(1, height + 1))
    return Sheet(name=name, state=state, rows=grid, merges=tuple(merges))


def _shared_strings(package: _Package, part: str | None) -> tuple[str, ...]:
    if part is None:
        return ()
    return tuple(_rich_text(si) for si in package.xml(part) if _local(si.tag) == "si")


def _date_styles(package: _Package, part: str) -> frozenset[int]:
    """Indexes into cellXfs whose number format shows a date or a time."""
    root = package.xml(part)
    ns = _namespace(root)
    custom: dict[int, str] = {}
    formats = root.find(_q(ns, "numFmts"))
    for fmt in formats if formats is not None else ():
        custom[_int_attr(fmt, "numFmtId", -1)] = fmt.get("formatCode", "")
    xfs_el = root.find(_q(ns, "cellXfs"))
    xfs = [xf for xf in (xfs_el if xfs_el is not None else ()) if _local(xf.tag) == "xf"]
    dated: set[int] = set()
    for index, xf in enumerate(xfs):
        fmt_id = _int_attr(xf, "numFmtId", 0)
        if fmt_id in _BUILTIN_DATE_FORMATS or (fmt_id in custom and _is_date_format(custom[fmt_id])):
            dated.add(index)
    return frozenset(dated)


def _rich_text(el: ElementTree.Element) -> str:
    """Plain text of a string item: <t>, or the <t> of each <r> run. Phonetic runs (<rPh>) are not the text."""
    parts: list[str] = []
    for child in el:
        tag = _local(child.tag)
        if tag == "t":
            parts.append(child.text or "")
        elif tag == "r":
            parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
    return "".join(parts)


def _rel_target(rels: dict[str, tuple[str, str]], type_suffix: str, fallback: str, package: _Package) -> str | None:
    """The part an optional relationship points at, when the part exists; a dangling relationship counts as absent."""
    for rel_type, target in rels.values():
        if rel_type.endswith(type_suffix) and package.has(target):
            return target
    return fallback if package.has(fallback) else None


def _parse_range(ref: str) -> CellRange:
    first, _, last = ref.partition(":")
    r1, c1 = parse_ref(first)
    r2, c2 = parse_ref(last) if last else (r1, c1)
    return CellRange(min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2))


def _resolve(base: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base, target))


def _int_attr(el: ElementTree.Element, name: str, default: int) -> int:
    raw = el.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(f"attribute {name}={raw!r} on <{_local(el.tag)}> is not an integer") from exc


def _namespace(root: ElementTree.Element) -> str:
    return root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""


def _q(ns: str, name: str) -> str:
    """The ElementTree name of a child element in the part's namespace (or none)."""
    return f"{{{ns}}}{name}" if ns else name


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
