from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Iterator, Sequence

import polars as pl
import pyarrow.parquet as pq

DEFAULT_BATCH_SIZE = 50_000


def parquet_row_count(path: str | Path) -> int:
    parquet_file = pq.ParquetFile(Path(path))
    metadata = parquet_file.metadata
    return 0 if metadata is None else metadata.num_rows


def aligned_parquet_row_count(paths: Sequence[str | Path]) -> int:
    row_counts = [parquet_row_count(path) for path in paths]
    if not row_counts:
        return 0

    if len(set(row_counts)) != 1:
        raise ValueError("Aligned parquet files do not contain the same number of rows.")

    return row_counts[0]


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressTracker:
    def __init__(
        self,
        description: str,
        total: int | None = None,
        unit: str = "rows",
        min_interval_seconds: float = 0.5,
    ):
        self.description = description
        self.total = None if total is None else max(int(total), 0)
        self.unit = unit
        self.min_interval_seconds = min_interval_seconds
        self._interactive = sys.stdout.isatty()
        self._started_at = time.perf_counter()
        self._current = 0
        self._steps = 0
        self._last_render_at = 0.0
        self._last_percent = -1
        self._last_message_length = 0

    def __enter__(self) -> "ProgressTracker":
        self._started_at = time.perf_counter()
        self._render(force=True)
        return self

    def advance(self, amount: int, steps: int = 1) -> None:
        if amount < 0:
            raise ValueError("Progress increment must be non-negative.")
        if steps < 0:
            raise ValueError("Step increment must be non-negative.")

        self._current += int(amount)
        self._steps += int(steps)
        self._render()

    def close(self) -> None:
        self._render(force=True, final=True)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def _should_render(self, *, force: bool, final: bool) -> bool:
        if force or final or self._steps <= 1:
            return True

        now = time.perf_counter()
        if now - self._last_render_at >= self.min_interval_seconds:
            return True

        if self.total and self.total > 0:
            percent = int((min(self._current, self.total) / self.total) * 100)
            if percent > self._last_percent:
                return True

        return False

    def _build_message(self) -> str:
        elapsed = time.perf_counter() - self._started_at
        parts = [self.description]

        if self.total is not None and self.total > 0:
            completed = min(self._current, self.total)
            fraction = completed / self.total
            bar_width = 20
            filled = min(bar_width, int(round(fraction * bar_width)))
            bar = "#" * filled + "-" * (bar_width - filled)
            parts.append(f"[{bar}] {fraction * 100:6.2f}%")
            parts.append(f"{completed:,}/{self.total:,} {self.unit}")

            if 0 < completed < self.total:
                eta = elapsed * (self.total - completed) / completed
                parts.append(f"eta {_format_duration(eta)}")
        elif self.total == 0:
            parts.append(f"0 {self.unit}")
        else:
            parts.append(f"{self._current:,} {self.unit}")

        batch_label = "batch" if self._steps == 1 else "batches"
        parts.append(f"{self._steps:,} {batch_label}")
        parts.append(f"elapsed {_format_duration(elapsed)}")
        return " | ".join(parts)

    def _render(self, *, force: bool = False, final: bool = False) -> None:
        if not self._should_render(force=force, final=final):
            return

        if self.total and self.total > 0:
            self._last_percent = int((min(self._current, self.total) / self.total) * 100)

        message = self._build_message()
        self._last_render_at = time.perf_counter()

        if self._interactive:
            padded = message.ljust(self._last_message_length)
            ending = "\n" if final else "\r"
            print(padded, end=ending, flush=True)
            self._last_message_length = max(self._last_message_length, len(message))
            return

        print(message, flush=True)
        self._last_message_length = len(message)


def iter_parquet_batches(
    path: str | Path,
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pl.DataFrame]:
    parquet_file = pq.ParquetFile(Path(path))

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(columns) if columns is not None else None,
        use_threads=True,
    ):
        yield pl.from_arrow(record_batch)


def iter_aligned_parquet_batches(
    paths: Sequence[str | Path],
    columns: Sequence[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[list[pl.DataFrame]]:
    parquet_files = [pq.ParquetFile(Path(path)) for path in paths]
    iterators = [
        parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns) if columns is not None else None,
            use_threads=True,
        )
        for parquet_file in parquet_files
    ]

    while True:
        batches: list[pl.DataFrame] = []
        ended = 0

        for iterator in iterators:
            try:
                batches.append(pl.from_arrow(next(iterator)))
            except StopIteration:
                ended += 1

        if ended == len(iterators):
            return

        if ended:
            raise ValueError("Parquet files do not contain the same number of rows.")

        heights = {batch.height for batch in batches}
        if len(heights) != 1:
            raise ValueError("Aligned parquet batches must have matching row counts.")

        yield batches


class ParquetBatchWriter:
    def __init__(self, output_path: str | Path, compression: str = "zstd"):
        self.output_path = Path(output_path)
        self.compression = compression
        self._tmp_path = self.output_path.with_suffix(f"{self.output_path.suffix}.tmp")
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> "ParquetBatchWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp_path.unlink(missing_ok=True)
        return self

    def write(self, dataframe: pl.DataFrame) -> None:
        if dataframe.height == 0:
            return

        table = dataframe.to_arrow()

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._tmp_path,
                table.schema,
                compression=self.compression,
            )

        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._tmp_path.exists():
            self.output_path.unlink(missing_ok=True)
            os.replace(self._tmp_path, self.output_path)

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if exc_type is not None:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._tmp_path.unlink(missing_ok=True)
            return

        self.close()
