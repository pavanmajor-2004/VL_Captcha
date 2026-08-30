r"""Recognition metrics and latency profiling for VL-KAN.

This module provides the standard evaluation surface used across validation,
ablation, and inference:

* **Character Accuracy** (``CAcc``) derived from the aggregate Character Error
  Rate (``CER``), itself based on the Levenshtein edit distance.
* **Sequence Accuracy** (``SAcc``) -- the exact-match rate over whole strings.
* **Length Accuracy** (``LAcc``) -- the rate at which the predicted transcription
  length equals the ground-truth length.

A lightweight :class:`LatencyTracker` and :class:`Timer` are included to profile
end-to-end processing latency and throughput in a framework-agnostic way.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

__all__ = [
    "edit_distance",
    "character_error_rate",
    "RecognitionMetrics",
    "MetricAccumulator",
    "evaluate_predictions",
    "Timer",
    "LatencyStats",
    "LatencyTracker",
]


# ---------------------------------------------------------------------------
# Edit distance / CER
# ---------------------------------------------------------------------------

def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Levenshtein edit distance between two sequences.

    Uses the standard two-row dynamic-programming recurrence (insertions,
    deletions, substitutions each cost 1) in ``O(len(reference) * len(hypothesis))``
    time and ``O(min(...))`` space.

    Parameters
    ----------
    reference:
        The ground-truth sequence (e.g. a string).
    hypothesis:
        The predicted sequence.

    Returns
    -------
    int
        Minimum number of single-token edits to turn ``hypothesis`` into
        ``reference``.
    """
    # Keep the inner loop over the shorter sequence for a smaller row.
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, r_tok in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, h_tok in enumerate(hypothesis, start=1):
            cost = 0 if r_tok == h_tok else 1
            current[j] = min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost,  # substitution / match
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Per-sample Character Error Rate ``edit_distance / len(reference)``.

    Parameters
    ----------
    reference:
        Ground-truth string.
    hypothesis:
        Predicted string.

    Returns
    -------
    float
        The CER. For an empty reference, returns ``0.0`` if the hypothesis is
        also empty, else ``1.0``.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


# ---------------------------------------------------------------------------
# Aggregate recognition metrics
# ---------------------------------------------------------------------------

@dataclass
class RecognitionMetrics:
    """Container for the headline recognition metrics.

    Attributes
    ----------
    char_accuracy:
        ``CAcc = 1 - CER`` aggregated over all characters.
    sequence_accuracy:
        ``SAcc`` -- fraction of exactly-correct transcriptions.
    length_accuracy:
        ``LAcc`` -- fraction of transcriptions with the correct length.
    cer:
        Aggregate Character Error Rate.
    num_samples:
        Number of samples scored.
    """

    char_accuracy: float
    sequence_accuracy: float
    length_accuracy: float
    cer: float
    num_samples: int

    def as_dict(self) -> Dict[str, float]:
        """Return the metrics as a plain dict (percentages for accuracies)."""
        return {
            "CAcc": self.char_accuracy,
            "SAcc": self.sequence_accuracy,
            "LAcc": self.length_accuracy,
            "CER": self.cer,
            "N": float(self.num_samples),
        }


class MetricAccumulator:
    """Streaming accumulator for recognition metrics.

    Call :meth:`update` with batches of predictions/references, then
    :meth:`compute` to obtain the aggregate :class:`RecognitionMetrics`. The
    Character Accuracy is computed from the *total* edit distance over the *total*
    reference length (the standard corpus-level CER), not by averaging per-sample
    rates.
    """

    def __init__(self) -> None:
        self._total_edits = 0
        self._total_ref_chars = 0
        self._exact_matches = 0
        self._length_matches = 0
        self._num_samples = 0

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.__init__()

    def update(self, predictions: Sequence[str], references: Sequence[str]) -> None:
        """Accumulate a batch of predicted/reference string pairs.

        Parameters
        ----------
        predictions:
            Predicted transcriptions.
        references:
            Ground-truth transcriptions (same length as ``predictions``).

        Raises
        ------
        ValueError
            If the two sequences differ in length.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references "
                f"({len(references)}) must align."
            )
        for pred, ref in zip(predictions, references):
            self._total_edits += edit_distance(ref, pred)
            self._total_ref_chars += len(ref)
            self._exact_matches += int(pred == ref)
            self._length_matches += int(len(pred) == len(ref))
            self._num_samples += 1

    def compute(self) -> RecognitionMetrics:
        """Finalize and return the aggregate metrics."""
        n = max(1, self._num_samples)
        ref_chars = max(1, self._total_ref_chars)
        cer = self._total_edits / ref_chars
        return RecognitionMetrics(
            # CER can exceed 1 (e.g. an over-long fixed-length prediction), so the
            # derived accuracy is floored at 0 to keep it a valid [0, 1] rate.
            char_accuracy=max(0.0, 1.0 - cer),
            sequence_accuracy=self._exact_matches / n,
            length_accuracy=self._length_matches / n,
            cer=cer,
            num_samples=self._num_samples,
        )


def evaluate_predictions(
    predictions: Sequence[str], references: Sequence[str],
) -> RecognitionMetrics:
    """One-shot convenience wrapper around :class:`MetricAccumulator`.

    Parameters
    ----------
    predictions, references:
        Aligned predicted/ground-truth strings.

    Returns
    -------
    RecognitionMetrics
        The aggregate metrics.
    """
    acc = MetricAccumulator()
    acc.update(predictions, references)
    return acc.compute()


# ---------------------------------------------------------------------------
# Latency / throughput profiling
# ---------------------------------------------------------------------------

class Timer(AbstractContextManager):
    """A minimal high-resolution context-manager timer.

    Examples
    --------
    >>> with Timer() as t:
    ...     run_inference()
    >>> print(t.elapsed)  # seconds
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.elapsed = time.perf_counter() - self._start


@dataclass
class LatencyStats:
    """Summary statistics of a latency profile.

    Attributes
    ----------
    mean_ms:
        Mean per-item latency in milliseconds.
    p50_ms, p95_ms, p99_ms:
        Latency percentiles in milliseconds.
    throughput:
        Items processed per second.
    num_items:
        Total items measured.
    """

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float
    num_items: int

    def as_dict(self) -> Dict[str, float]:
        """Return the stats as a plain dict."""
        return {
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "throughput": self.throughput,
            "num_items": float(self.num_items),
        }


class LatencyTracker:
    """Records timed processing events and reports latency/throughput.

    Each :meth:`record` call logs the wall-clock duration of processing a given
    number of items, enabling both per-item latency percentiles and overall
    throughput.
    """

    def __init__(self) -> None:
        self._per_item_ms: List[float] = []
        self._total_items = 0
        self._total_seconds = 0.0

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self.__init__()

    def record(self, seconds: float, num_items: int = 1) -> None:
        """Log one timed event.

        Parameters
        ----------
        seconds:
            Wall-clock duration of the event.
        num_items:
            Number of items processed during the event (e.g. a batch size).
        """
        if num_items < 1:
            raise ValueError("num_items must be >= 1.")
        per_item = (seconds / num_items) * 1000.0
        # Attribute the event's per-item latency to each item it covered.
        self._per_item_ms.extend([per_item] * num_items)
        self._total_items += num_items
        self._total_seconds += seconds

    def measure(self, num_items: int = 1) -> Timer:
        """Return a :class:`Timer` whose result is auto-recorded on exit.

        Parameters
        ----------
        num_items:
            Items covered by the timed block.

        Returns
        -------
        Timer
            A timer subclass that records itself.
        """
        tracker = self

        class _RecordingTimer(Timer):
            def __exit__(self, *exc):  # noqa: ANN002
                super().__exit__(*exc)
                tracker.record(self.elapsed, num_items)

        return _RecordingTimer()

    def compute(self) -> LatencyStats:
        """Finalize and return the latency summary."""
        if not self._per_item_ms:
            return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        ordered = sorted(self._per_item_ms)

        def _pct(p: float) -> float:
            idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
            return ordered[idx]

        mean_ms = sum(ordered) / len(ordered)
        throughput = (
            self._total_items / self._total_seconds
            if self._total_seconds > 0 else 0.0
        )
        return LatencyStats(
            mean_ms=mean_ms,
            p50_ms=_pct(0.50),
            p95_ms=_pct(0.95),
            p99_ms=_pct(0.99),
            throughput=throughput,
            num_items=self._total_items,
        )
