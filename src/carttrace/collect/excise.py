"""Cutting the statements of hidden details out of product text (plain, markdown or HTML), with every other character
kept where it was.

Why cut and never rewrite. The two arms of a detail_hidden pair must differ by the hidden detail and nothing else:
the same words, the same markup, the same order. So text is only ever cut: a line, a sentence inside a line, a table
row, a whole table, or a heading with the block it heads. In HTML only text is cut, and an element goes only when the
cut has emptied it, so the markup the agent reads stays well formed and shows no gap.

What a unit is. The readers of product text (`carttrace.agents.text`) read lines: a line ends at a block element, a <br> or
a newline, and a table row reads as its cells joined. Those lines are the units here, split into sentences at . ! and
?, so a paragraph that states one detail among others loses only that sentence. A heading that names only hidden
details ("Ingredients", "Care") goes with the block it heads: the next line, list or table. A table the size chart
reader reads as a chart goes whole. Which units state a detail is the matchers' call (`carttrace.collect.matchers`).
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser

from carttrace.agents.products import Grid
from carttrace.agents.text import BLOCK_TAGS, CELL_TAGS, SKIP_TAGS, cells, collapse, looks_html, separator_row, table_runs
from carttrace.collect.matchers import HEADING_CHARS, HiddenSet, heading

Span = tuple[int, int]
"""A [start, end) range of the source text."""

VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
"""Elements that never have content or an end tag."""
CLOSES_P = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)
"""Start tags that close an open <p> (the HTML standard's list); a <br> is not one of them."""
LIST_TAGS = {"li": ("ul", "ol"), "dt": ("dl",), "dd": ("dl",), "tr": ("table",), "td": ("table",), "th": ("table",)}
"""Items whose heading heads the whole list or table they sit in: a "Care" heading before a list heads every item."""
ABBREVIATIONS = frozenset(
    {
        "approx",
        "corp",
        "denat",
        "dept",
        "dr",
        "e.g",
        "e.u",
        "eg",
        "est",
        "excl",
        "ext",
        "extr",
        "fig",
        "fl",
        "i.e",
        "ie",
        "inc",
        "incl",
        "ltd",
        "mfg",
        "mr",
        "mrs",
        "ms",
        "no",
        "nos",
        "nr",
        "pkg",
        "qty",
        "ref",
        "st",
        "u.k",
        "u.s",
        "vol",
        "vs",
        "wt",
    }
)
"""Words whose full stop does not end a sentence. Units are not here: "8 fl oz. Shake well." is two sentences, and
"Alcohol Denat." inside an INCI list is not."""

_PREFIX = re.compile(r"^\s*(?:(?:[-*+•·‣◦▪●○►✓✔]|\d{1,2}[.)])\s+)?")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WORD_BEFORE = re.compile(r"[A-Za-z.]+$")
_STARTS_SENTENCE = re.compile(r"[A-Z0-9\"'(\[•*-]")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•·‣◦▪●○►✓✔]|\d{1,2}[.)])\s+\S")


def excise(text: str, hidden: HiddenSet, *, as_html: bool | None = None) -> tuple[str, Counter[str]]:
    """`text` without what states a hidden detail, and how many statements of each detail were cut. HTML is read as
    HTML (`as_html`, or by its look when None). Text that states none comes back as the same string."""
    use_html = looks_html(text) if as_html is None else as_html
    doc: _Doc = _HtmlDoc(text) if use_html else _TextDoc(text)
    return doc.cut(hidden)


@dataclass(slots=True)
class _Unit:
    """One line of the text, as the readers read lines."""

    text: str
    """The line's text, entities decoded."""
    spans: list[Span] = field(default_factory=list)
    """Where each character of `text` sits in the source."""
    before: Span | None = None
    """The separator before the line (a newline or a <br>), when one joins it to the line before."""
    after: Span | None = None
    block: int = 0
    """The innermost block element that holds the line (HTML); 0 is the fragment itself."""
    owners: list[int] = field(default_factory=list)
    """The innermost element each character sits in (HTML)."""

    @property
    def blank(self) -> bool:
        return not self.text.strip()

    @property
    def at(self) -> int:
        """Where the line starts in the source."""
        return self.spans[0][0] if self.spans else -1


@dataclass(slots=True)
class _Row:
    units: list[int]
    first: str
    """The text of the row's first cell: a label such as "Material" when the table is a list of facts."""
    text: str
    """The row as a reader's line: its cells joined with " | "."""


@dataclass(slots=True)
class _Table:
    units: list[int]
    """Every line of the table, separator rows and captions included."""
    rows: list[_Row]
    grid: Grid
    """The cells' text, as the size chart reader reads a table."""


class _Doc:
    """Units and tables over one source text, and the cut that removes what states a hidden detail."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.units: list[_Unit] = []
        self.tables: list[_Table] = []

    def region(self, head: int, first: int) -> list[int]:
        """The units a heading heads, given the first unit after it."""
        raise NotImplementedError

    def emptied(self, kept: Sequence[Sequence[bool]]) -> list[Span]:
        """Markup the cut leaves without text, to go with it; `kept` says which characters of each unit stay."""
        return []

    def cut(self, hidden: HiddenSet) -> tuple[str, Counter[str]]:
        counts: Counter[str] = Counter()
        gone: set[int] = set()
        partial: dict[int, list[Span]] = {}
        handled = {u for table in self.tables for u in table.units}
        for table in self.tables:
            self._cut_table(table, hidden, gone, counts)
        index = 0
        while index < len(self.units):
            unit = self.units[index]
            if index in handled or index in gone or unit.blank:
                index += 1
                continue
            named = heading(unit.text)
            if named is not None:
                if named <= hidden.details:
                    region = self._headed(index)
                    if not region or not set(region) <= gone:
                        counts.update(named)
                    gone.add(index)
                    gone.update(region)
                index += 1
                continue
            sentences = sentence_spans(unit.text)
            stated = [hidden.stated(unit.text[start:end]) for start, end in sentences]
            for found in stated:
                counts.update(found)
            if sentences and all(stated):
                gone.add(index)
            elif any(stated):
                partial[index] = _sentence_cuts(sentences, [bool(found) for found in stated])
            index += 1
        if not gone and not partial:
            return self.source, counts
        return self._apply(gone, partial), counts

    def _cut_table(self, table: _Table, hidden: HiddenSet, gone: set[int], counts: Counter[str]) -> None:
        if hidden.chart(table.grid):
            gone.update(table.units)
            counts["size_chart"] += 1
            caption = self._caption(table)
            if caption is not None and hidden.keyed(self.units[caption].text):
                gone.add(caption)
            return
        for row in table.rows:
            named = heading(row.first)
            found = named if named is not None and named <= hidden.details else hidden.stated(row.text)
            if found:
                gone.update(row.units)
                counts.update(found)

    def _caption(self, table: _Table) -> int | None:
        """The short line right above a table that titles it ("Size guide for all shirts"), if there is one: it goes
        with a chart it names, since a title over nothing still tells the agent a chart was there."""
        first = min(table.units)
        above = next((i for i in range(first - 1, -1, -1) if not self.units[i].blank), None)
        if above is None or any(above in other.units for other in self.tables):
            return None
        text = self.units[above].text
        return above if len(text.strip()) <= HEADING_CHARS and len(sentence_spans(text)) == 1 else None

    def _headed(self, head: int) -> list[int]:
        following = next((i for i in range(head + 1, len(self.units)) if not self.units[i].blank), None)
        if following is None or heading(self.units[following].text) is not None:
            return []
        return self.region(head, following)

    def _apply(self, gone: set[int], partial: dict[int, list[Span]]) -> str:
        spans: list[Span] = []
        kept = [[True] * len(unit.text) for unit in self.units]
        for index, cuts in partial.items():
            if index in gone:
                continue
            unit = self.units[index]
            for start, end in cuts:
                spans += unit.spans[start:end]
                kept[index][start:end] = [False] * (end - start)
        for first, last in _runs(sorted(gone)):
            spans += self._run_spans(first, last, gone)
            for index in range(first, last + 1):
                kept[index] = [False] * len(self.units[index].text)
        spans += self.emptied(kept)
        return _without(self.source, spans)

    def _run_spans(self, first: int, last: int, gone: set[int]) -> list[Span]:
        """A run of whole lines, with the separators inside it and one at its edge, so no empty line is left. When a
        blank line (or the start or end of the text) stood on both sides, one blank line goes too."""
        units = self.units
        spans = [span for unit in units[first : last + 1] for span in unit.spans]
        spans += [unit.after for unit in units[first:last] if unit.after is not None]
        edge = units[first].before if units[first].before is not None else units[last].after
        if edge is not None:
            spans.append(edge)
        after = last + 1
        blank_before = first == 0 or (units[first - 1].blank and units[first].before is not None)
        blank_after = after < len(units) and units[after].blank and after not in gone and units[last].after is not None
        extra: Span | None = None
        if blank_before and first == 0 and blank_after:
            extra = units[after].after
        elif blank_before and blank_after:
            extra = units[last].after
        elif blank_before and first > 0 and after == len(units):
            extra = units[first - 1].before
        if extra is not None:
            spans.append(extra)
        return spans


class _TextDoc(_Doc):
    """Plain text or markdown: a unit per line, tables where `carttrace.agents.text.pipe_tables` finds them."""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        offset = 0
        previous: Span | None = None
        for line in source.split("\n"):
            end = offset + len(line)
            body = line[:-1] if line.endswith("\r") else line
            separator = (offset + len(body), end + 1) if end < len(source) else None
            self.units.append(
                _Unit(body, [(offset + i, offset + i + 1) for i in range(len(body))], previous, separator)
            )
            previous = separator
            offset = end + 1
        lines = [unit.text for unit in self.units]
        for start, end in table_runs(lines):
            rows = [start, *(i for i in range(start + 1, end) if not separator_row(lines[i]))]
            grid = tuple(tuple(cells(lines[i])) for i in rows)
            self.tables.append(
                _Table(
                    list(range(start, end)),
                    [_Row([i], cells(lines[i])[0], " | ".join(cells(lines[i]))) for i in rows],
                    grid,
                )
            )

    def region(self, head: int, first: int) -> list[int]:
        between = list(range(head + 1, first))
        table = next((t for t in self.tables if first in t.units), None)
        if table is not None:
            return between + table.units
        if not _LIST_ITEM.match(self.units[first].text):
            return [*between, first]
        last = first
        while last + 1 < len(self.units) and _LIST_ITEM.match(self.units[last + 1].text):
            last += 1
        return between + list(range(first, last + 1))


@dataclass(slots=True)
class _Element:
    tag: str
    start: int
    end: int = -1
    parent: int = 0


class _HtmlDoc(_Doc):
    """An HTML fragment: a unit per line as `carttrace.agents.text.read_html` reads lines, over a tree of its elements."""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.tokens = _tokens(source)
        self.elements: list[_Element] = [_Element("", 0, len(source), -1)]
        self.owner: list[int] = []
        """The element each token sits in."""
        self._build_tree()
        self._skipped = self._skipped_elements()
        self._build_units()
        self._build_tables()

    # -- reading ---------------------------------------------------------------------------------------------------

    def _build_tree(self) -> None:
        stack = [0]
        for kind, start, end, tag in self.tokens:
            if kind == "start":
                self._imply_end(stack, tag, start)
            self.owner.append(stack[-1])
            if kind in ("start", "void"):
                element = _Element(tag, start, end if kind == "void" or tag in VOID_TAGS else -1, stack[-1])
                self.elements.append(element)
                if element.end == -1:
                    stack.append(len(self.elements) - 1)
            elif kind == "end" and tag not in VOID_TAGS:
                match = next((i for i in range(len(stack) - 1, 0, -1) if self.elements[stack[i]].tag == tag), None)
                if match is None:
                    continue  # a stray end tag closes nothing
                while len(stack) > match + 1:
                    self.elements[stack.pop()].end = start
                self.elements[stack.pop()].end = end
        while len(stack) > 1:
            self.elements[stack.pop()].end = len(self.source)

    def _imply_end(self, stack: list[int], tag: str, at: int) -> None:
        """Close what a start tag closes without an end tag: a <p> at a block, an <li> at the next, a cell at the
        next cell or row."""

        def close_if(*tags: str) -> None:
            if len(stack) > 1 and self.elements[stack[-1]].tag in tags:
                self.elements[stack.pop()].end = at

        if tag in CLOSES_P:
            close_if("p")
        if tag == "li":
            close_if("li")
        elif tag in ("dt", "dd"):
            close_if("dt", "dd")
        elif tag in CELL_TAGS:
            close_if(*CELL_TAGS)
        elif tag == "tr":
            close_if(*CELL_TAGS)
            close_if("tr")
        elif tag == "option":
            close_if("option")

    def _skipped_elements(self) -> list[bool]:
        skipped = [False] * len(self.elements)
        for index, element in enumerate(self.elements):
            if index:
                skipped[index] = skipped[element.parent] or element.tag in SKIP_TAGS
        return skipped

    def _block_of(self, element: int) -> int:
        while element and self.elements[element].tag not in BLOCK_TAGS and self.elements[element].tag not in CELL_TAGS:
            element = self.elements[element].parent
        return element

    def _build_units(self) -> None:
        current = _Unit("")
        pieces: list[str] = []

        def finish(separator: Span | None) -> _Unit:
            current.text = "".join(pieces)
            current.after = separator
            self.units.append(current)
            pieces.clear()
            return _Unit("", before=separator)

        def add(text: str, spans: list[Span], owner: int) -> None:
            if not text:
                return
            if not current.spans:
                current.block = self._block_of(owner)
            pieces.append(text)
            current.spans.extend(spans)
            current.owners.extend([owner] * len(text))

        for (kind, start, end, tag), owner in zip(self.tokens, self.owner, strict=True):
            if kind in ("start", "end", "void"):
                if tag == "br":
                    current = finish((start, end))
                elif tag in BLOCK_TAGS or tag in CELL_TAGS:
                    current = finish(None)
                continue
            if kind not in ("text", "ref") or self._skipped[owner]:
                continue
            if kind == "ref":
                decoded = html.unescape(self.source[start:end])
                if decoded == "\n":
                    current = finish((start, end))
                else:
                    add(decoded, [(start, end)] + [(end, end)] * (len(decoded) - 1), owner)
                continue
            offset = start
            for number, line in enumerate(self.source[start:end].split("\n")):
                if number:
                    current = finish((offset - 1, offset))
                add(line, [(offset + i, offset + i + 1) for i in range(len(line))], owner)
                offset += len(line) + 1
        finish(None)
        self.units = [unit for unit in self.units if unit.text or unit.before or unit.after]

    def _build_tables(self) -> None:
        """Each table with its rows and cells as the size chart reader reads them: every cell kept, empty ones too, so
        columns stay aligned; rows with no text left out."""
        cell_units: dict[int, list[int]] = {}
        table_units: dict[int, list[int]] = {}
        for index, unit in enumerate(self.units):
            table = self._nearest(unit.block, "table")
            if table is None:
                continue
            table_units.setdefault(table, []).append(index)
            cell = self._nearest(unit.block, *CELL_TAGS, stop="table")
            if cell is not None:
                cell_units.setdefault(cell, []).append(index)
        rows_of: dict[int, list[int]] = {}
        cells_of: dict[int, list[int]] = {}
        for index, element in enumerate(self.elements):
            if element.tag == "tr":
                table = self._nearest(element.parent, "table")
                if table is not None:
                    rows_of.setdefault(table, []).append(index)
            elif element.tag in CELL_TAGS:
                row = self._nearest(element.parent, "tr", stop="table")
                if row is not None:
                    cells_of.setdefault(row, []).append(index)
        for table, members in table_units.items():
            rows: list[_Row] = []
            grid: list[tuple[str, ...]] = []
            for row in rows_of.get(table, []):
                units = [cell_units.get(cell, []) for cell in cells_of.get(row, [])]
                texts = [collapse(" ".join(self.units[i].text for i in found)) for found in units]
                if not any(texts):
                    continue
                grid.append(tuple(texts))
                rows.append(_Row([i for found in units for i in found], texts[0], " | ".join(texts)))
            self.tables.append(_Table(members, rows, tuple(grid)))

    def _nearest(self, element: int | None, *tags: str, stop: str | None = None) -> int | None:
        """The element itself or its nearest ancestor with one of `tags`, not looking past a `stop` element."""
        while element:
            tag = self.elements[element].tag
            if tag in tags:
                return element
            if stop is not None and tag == stop:
                return None
            element = self.elements[element].parent
        return None

    # -- cutting ---------------------------------------------------------------------------------------------------

    def _contains(self, element: int, position: int) -> bool:
        return self.elements[element].start <= position < self.elements[element].end

    def region(self, head: int, first: int) -> list[int]:
        container = self.units[first].block
        tag = self.elements[container].tag
        if tag in LIST_TAGS:
            whole = self._nearest(container, *LIST_TAGS[tag])
            if whole is not None and not self._contains(whole, self.units[head].at):
                container = whole
        if container == 0 or self._contains(container, self.units[head].at):
            return [first]
        return [i for i, unit in enumerate(self.units) if unit.spans and self._contains(container, unit.at)]

    def emptied(self, kept: Sequence[Sequence[bool]]) -> list[Span]:
        """Elements whose text the cut removed entirely, outermost first, with the white space that set them apart
        from their neighbours; an element that never had text (an empty cell, an image) is left as it was."""
        tally: dict[int, list[int]] = {}
        """Per element holding text directly: its characters, and how many of them stay."""
        for unit, flags in zip(self.units, kept, strict=True):
            for char, owner, stays in zip(unit.text, unit.owners, flags, strict=True):
                if not char.isspace():
                    counts = tally.setdefault(owner, [0, 0])
                    counts[0] += 1
                    counts[1] += stays
        had: Counter[int] = Counter()
        has: Counter[int] = Counter()
        for owner, (seen, staying) in tally.items():
            ancestor = owner
            while ancestor > 0:
                had[ancestor] += seen
                has[ancestor] += staying
                ancestor = self.elements[ancestor].parent
        removed: list[Span] = []
        for index in sorted(had, key=lambda i: self.elements[i].start):
            element = self.elements[index]
            if has[index] or (removed and element.start < removed[-1][1]):
                continue
            if removed and not self.source[removed[-1][1] : element.start].strip():
                removed[-1] = (removed[-1][0], element.end)  # neighbours with only white space between go as one
            else:
                removed.append((element.start, element.end))
        return [edge for span in removed for edge in (span, self._spacing(*span))]

    def _spacing(self, start: int, end: int) -> Span:
        """The white space between removed markup and the tag or edge before it, or else after it: what kept it on a
        line of its own."""
        before = len(self.source[:start].rstrip())
        if before < start and (before == 0 or self.source[before - 1] == ">"):
            return (before, start)
        after = len(self.source) - len(self.source[end:].lstrip())
        if after > end and (after == len(self.source) or self.source[after] == "<"):
            return (end, after)
        return (start, start)


class _Tokens(HTMLParser):
    """Every token of an HTML fragment with where it starts, so a cut can address the source exactly."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.line_starts = [0, *(i + 1 for i, char in enumerate(source) if char == "\n")]
        self.found: list[tuple[str, int, str]] = []

    def _add(self, kind: str, tag: str = "") -> None:
        line, column = self.getpos()
        self.found.append((kind, self.line_starts[line - 1] + column, tag))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._add("start", tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._add("void", tag)

    def handle_endtag(self, tag: str) -> None:
        self._add("end", tag)

    def handle_data(self, data: str) -> None:
        self._add("text")

    def handle_entityref(self, name: str) -> None:
        self._add("ref")

    def handle_charref(self, name: str) -> None:
        self._add("ref")

    def handle_comment(self, data: str) -> None:
        self._add("other")

    def handle_decl(self, decl: str) -> None:
        self._add("other")

    def handle_pi(self, data: str) -> None:
        self._add("other")

    def unknown_decl(self, data: str) -> None:
        self._add("other")


def _tokens(source: str) -> list[tuple[str, int, int, str]]:
    """(kind, start, end, tag) for every token of the fragment, in order and covering it."""
    parser = _Tokens(source)
    parser.feed(source)
    parser.close()
    found = sorted(parser.found, key=lambda token: token[1])
    if not found or found[0][1] > 0:
        found.insert(0, ("other", 0, ""))
    ends = [start for _, start, _ in found[1:]] + [len(source)]
    return [(kind, start, end, tag) for (kind, start, tag), end in zip(found, ends, strict=True) if end > start]


def sentence_spans(text: str) -> list[Span]:
    """Where the sentences of a line are, after any list marker: split at . ! or ? followed by a space and a capital,
    a digit or a quote, except after an abbreviation such as "approx." or "Denat."."""
    prefix = _PREFIX.match(text)
    begin = prefix.end() if prefix is not None else 0
    spans: list[Span] = []
    for boundary in _BOUNDARY.finditer(text, begin):
        end, following = boundary.start(), boundary.end()
        if following >= len(text) or not _STARTS_SENTENCE.match(text, following):
            continue
        word = _WORD_BEFORE.search(text, begin, end - 1)
        if text[end - 1] == "." and word is not None and word.group(0).lower().rstrip(".") in ABBREVIATIONS:
            continue
        if end > begin:
            spans.append((begin, end))
        begin = following
    tail = len(text.rstrip())
    if tail > begin:
        spans.append((begin, tail))
    return spans


def _sentence_cuts(sentences: Sequence[Span], stated: Sequence[bool]) -> list[Span]:
    """What to cut for each run of stated sentences, with the space that joined it to the rest: the space before it,
    or after it when the run opens the line. At least one sentence is not stated."""
    cuts: list[Span] = []
    index = 0
    while index < len(sentences):
        if not stated[index]:
            index += 1
            continue
        last = index
        while last + 1 < len(sentences) and stated[last + 1]:
            last += 1
        if index == 0:
            cuts.append((sentences[0][0], sentences[last + 1][0]))
        else:
            cuts.append((sentences[index - 1][1], sentences[last][1]))
        index = last + 1
    return cuts


def _runs(indexes: Sequence[int]) -> list[Span]:
    """Consecutive indexes as (first, last) pairs."""
    runs: list[Span] = []
    for index in indexes:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def _without(source: str, spans: Iterable[Span]) -> str:
    out: list[str] = []
    position = 0
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if end <= position:
            continue
        out.append(source[position : max(position, start)])
        position = max(position, end)
    out.append(source[position:])
    return "".join(out)
