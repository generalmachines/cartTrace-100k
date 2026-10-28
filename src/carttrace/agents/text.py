"""Product text as an agent reads it: HTML descriptions turned into lines and tables, and small word helpers.

Why one reader. The details that decide a task (a size chart, an ingredient list, a warning) live in product
descriptions written for people: HTML tables, paragraphs, markdown pipes, "Ingredients:" headings. Task generation,
the reference agent and the shopper's keyword rule must read them the same way, or a task would be written around a
size chart the reference agent cannot see. Standard library only (html.parser), so it reads any store's HTML without
a browser.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from html.parser import HTMLParser

Table = list[list[str]]
"""Rows of cells, each cell's text with whitespace collapsed."""

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "thead",
        "tfoot",
        "tr",
        "ul",
    }
)
"""Elements that start a new line of text: a line of `read_html` never crosses one."""
CELL_TAGS = frozenset({"td", "th"})
SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
"""Elements whose content is not text a reader sees."""
_SPACE = re.compile(r"\s+")
_SEPARATOR_ROW = re.compile(r"^[\s|:\-+=]+$")
_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[str] = []
        self.tables: list[Table] = []
        self._open_tables: list[Table] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "table":
            self._open_tables.append([])
        elif tag == "tr" and self._open_tables:
            self._row = []
        elif tag in CELL_TAGS and self._row is not None:
            self._cell = []
            self.pieces.append(" | ")
        if tag in BLOCK_TAGS:
            self.pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in CELL_TAGS and self._cell is not None and self._row is not None:
            self._row.append(collapse("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._open_tables:
            if any(self._row):
                self._open_tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._open_tables:
            table = self._open_tables.pop()
            if table:
                self.tables.append(table)
        if tag in BLOCK_TAGS:
            self.pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.pieces.append(data)
        if self._cell is not None:
            self._cell.append(data)


def read_html(html: str) -> tuple[list[str], list[Table]]:
    """(text lines, tables) of an HTML fragment. Table cells inside a line are joined with " | ", so a table also
    reads as pipe-separated text; the tables themselves keep every cell, empty ones included."""
    reader = _Reader()
    reader.feed(html)
    reader.close()
    return lines_of("".join(reader.pieces)), reader.tables


def lines_of(text: str) -> list[str]:
    """Lines with spaces collapsed (tabs kept, since they separate cells), dropping lines that hold nothing but
    spaces and pipes."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"[ \f\v\r]+", " ", raw).strip(" ")
        if line.strip(" |\t"):
            out.append(line)
    return out


def pipe_tables(lines: list[str]) -> list[Table]:
    """Tables written as text: runs of two or more lines that split into the same number (two or more) of cells on
    "|" or tabs. Markdown separator rows are skipped; an empty corner cell is kept, so columns stay aligned."""
    return [
        [cells(lines[start]), *(cells(line) for line in lines[start + 1 : end] if not separator_row(line))]
        for start, end in table_runs(lines)
    ]


def table_runs(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Where `pipe_tables` finds its tables: (first, end) line indexes of each run, markdown separator rows inside it
    included. Whoever cuts a table out of a text cuts exactly the lines it was read from."""
    runs: list[tuple[int, int]] = []
    start, width, rows = 0, 0, 0
    for index, line in enumerate([*lines, ""]):
        if rows and separator_row(line):
            continue
        row = cells(line)
        if len(row) >= 2 and (not rows or len(row) == width):
            if not rows:
                start, width = index, len(row)
            rows += 1
            continue
        if rows >= 2:
            runs.append((start, index))
        start, width, rows = (index, len(row), 1) if len(row) >= 2 else (index, 0, 0)
    return runs


def separator_row(line: str) -> bool:
    """A markdown table's separator row ("|---|:---:|")."""
    return bool(line) and _SEPARATOR_ROW.match(line) is not None


def cells(line: str) -> list[str]:
    """The cells of a line of a text table, split on "|" or tabs; [] when the line is no table row."""
    if "|" in line:
        parts = line.split("|")
    elif "\t" in line:
        parts = line.split("\t")
    else:
        return []
    if parts and not parts[0].strip():
        parts = parts[1:]  # the text before a leading pipe
    if parts and not parts[-1].strip():
        parts = parts[:-1]  # the text after a trailing pipe
    found = [collapse(p) for p in parts]
    return found if any(found) else []


def looks_html(text: str) -> bool:
    """Whether text is HTML rather than plain text: it closes a tag or breaks a line with <br>."""
    return ("</" in text and ">" in text) or "<br" in text.lower()


def collapse(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


@lru_cache(maxsize=1 << 17)
def fold(text: str) -> str:
    """Lowercase with whitespace collapsed and typographic dashes and quotes made plain, for matching. Remembered:
    task generation folds the same titles and option names millions of times."""
    plain = text.replace("–", "-").replace("—", "-").replace("’", "'").replace("″", '"').replace("”", '"')
    return collapse(plain.lower())


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


@lru_cache(maxsize=1 << 17)
def has_phrase(text: str, phrase: str) -> bool:
    """True when `phrase` occurs in `text` at word boundaries, case-insensitively. A phrase ending in a hyphen or
    a letter run marked with a trailing "*" matches as a prefix ("allerg*" matches allergy and allergic).
    Remembered, like `fold`: it is a pure function of two strings."""
    return _phrase_pattern(fold(phrase)).search(fold(text)) is not None


@lru_cache(maxsize=1 << 14)
def _phrase_pattern(needle: str) -> re.Pattern[str]:
    if needle.endswith("*"):
        return re.compile(rf"(?<![a-z0-9]){re.escape(needle[:-1])}")
    return re.compile(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])")
