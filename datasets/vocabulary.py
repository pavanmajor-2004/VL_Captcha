"""Alphabet and token-mapping utilities for VL-KAN.

This module defines the :class:`Vocabulary` abstraction used across the entire
VL-KAN pipeline. It maps human-readable CAPTCHA strings to integer index
sequences (and back) and owns the **CTC blank token** convention used by every
sequence-recognition component in the project.

Conventions (must stay consistent with the approved tensor contracts):

* The blank token occupies index ``0``.
* Real alphabet symbols occupy contiguous indices ``1 .. V`` where ``V`` is the
  alphabet size (``10`` for numeric, ``62`` for case-sensitive alphanumeric).
* The number of CTC output classes is therefore ``V + 1`` (alphabet + blank),
  which matches the ``CTCHead`` output dimension in the Contract Table.

The case-sensitive alphanumeric alphabet intentionally keeps visually ambiguous
glyphs (``0``/``O``, ``1``/``l``/``I`` ...). The :attr:`Vocabulary.ambiguous_pairs`
attribute documents these so downstream evaluation/analysis can account for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "NUMERIC_ALPHABET",
    "ALPHANUMERIC_ALPHABET",
    "AMBIGUOUS_PAIRS",
    "Vocabulary",
    "build_vocabulary",
]

# --- Canonical alphabets -----------------------------------------------------

#: Digits ``0-9`` (10 symbols).
NUMERIC_ALPHABET: str = "0123456789"

#: Digits + uppercase + lowercase Latin letters, case sensitive (62 symbols).
ALPHANUMERIC_ALPHABET: str = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)

#: Visually confusable glyph groups that historically defeat solvers/humans.
#: Documented (not removed) so analyses can quantify their error contribution.
AMBIGUOUS_PAIRS: Tuple[Tuple[str, ...], ...] = (
    ("0", "O", "o"),
    ("1", "l", "I"),
    ("2", "Z", "z"),
    ("5", "S", "s"),
    ("8", "B"),
    ("6", "G"),
    ("9", "g", "q"),
    ("u", "v"),
    ("w", "vv"),
)

_PRESET_ALPHABETS: Dict[str, str] = {
    "numeric": NUMERIC_ALPHABET,
    "alphanumeric": ALPHANUMERIC_ALPHABET,
}


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional mapping between CAPTCHA strings and CTC index sequences.

    Parameters
    ----------
    alphabet:
        Ordered string of unique symbols. Order defines the index assignment.
    blank_index:
        Reserved index for the CTC blank token. Must be ``0`` for compatibility
        with the project-wide convention (alphabet symbols start at ``1``).
    name:
        Optional human-readable identifier (e.g. ``"numeric"``).

    Attributes
    ----------
    char_to_index:
        Mapping from symbol to its integer class index (``1 .. V``).
    index_to_char:
        Inverse mapping from integer class index to symbol.
    ambiguous_pairs:
        Tuple of confusable glyph groups that are present in this alphabet.
    """

    alphabet: str
    blank_index: int = 0
    name: str = "custom"
    char_to_index: Dict[str, int] = field(init=False, repr=False)
    index_to_char: Dict[int, str] = field(init=False, repr=False)
    ambiguous_pairs: Tuple[Tuple[str, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.blank_index != 0:
            raise ValueError(
                "Vocabulary requires blank_index == 0 to satisfy the "
                f"project CTC contract; got {self.blank_index}."
            )
        if len(self.alphabet) == 0:
            raise ValueError("Alphabet must not be empty.")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("Alphabet contains duplicate symbols.")

        # Symbols are assigned indices 1 .. V (blank owns index 0).
        char_to_index = {ch: i + 1 for i, ch in enumerate(self.alphabet)}
        index_to_char = {i + 1: ch for i, ch in enumerate(self.alphabet)}

        present: List[Tuple[str, ...]] = []
        symbol_set = set(self.alphabet)
        for group in AMBIGUOUS_PAIRS:
            members = tuple(g for g in group if len(g) == 1 and g in symbol_set)
            if len(members) >= 2:
                present.append(members)

        # ``frozen=True`` forbids normal assignment; use object.__setattr__.
        object.__setattr__(self, "char_to_index", char_to_index)
        object.__setattr__(self, "index_to_char", index_to_char)
        object.__setattr__(self, "ambiguous_pairs", tuple(present))

    # --- Sizes ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of real alphabet symbols ``V`` (excludes blank)."""
        return len(self.alphabet)

    @property
    def num_classes(self) -> int:
        """Number of CTC output classes ``V + 1`` (alphabet + blank)."""
        return self.size + 1

    def __len__(self) -> int:
        return self.size

    def __contains__(self, char: str) -> bool:
        return char in self.char_to_index

    # --- Encoding ------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of class indices in ``1 .. V``.

        Parameters
        ----------
        text:
            String whose characters all belong to the alphabet.

        Returns
        -------
        list of int
            Class indices (never contains the blank index).

        Raises
        ------
        KeyError
            If any character is not part of the alphabet.
        """
        try:
            return [self.char_to_index[ch] for ch in text]
        except KeyError as exc:  # pragma: no cover - defensive message
            raise KeyError(
                f"Character {exc.args[0]!r} is not in alphabet {self.name!r}."
            ) from None

    def encode_batch(self, texts: Sequence[str]) -> Tuple[List[int], List[int]]:
        """Encode a batch of strings into a flat target buffer for CTC.

        ``torch.nn.CTCLoss`` accepts targets either as a 2-D padded tensor or as
        a 1-D concatenation paired with per-sample lengths. This helper returns
        the concatenated form, which avoids a padding value colliding with real
        class indices.

        Parameters
        ----------
        texts:
            Sequence of strings to encode.

        Returns
        -------
        tuple
            ``(flat_targets, target_lengths)`` where ``flat_targets`` is the
            concatenation of all encoded sequences and ``target_lengths`` holds
            the length of each individual sequence.
        """
        flat: List[int] = []
        lengths: List[int] = []
        for text in texts:
            encoded = self.encode(text)
            flat.extend(encoded)
            lengths.append(len(encoded))
        return flat, lengths

    # --- Decoding ------------------------------------------------------------

    def decode(self, indices: Iterable[int], strip_blank: bool = True) -> str:
        """Decode a sequence of class indices back into a string.

        This is a *literal* decode: it does not collapse CTC repeats. Use
        :meth:`ctc_greedy_decode` for raw network output.

        Parameters
        ----------
        indices:
            Iterable of integer class indices.
        strip_blank:
            If ``True``, blank indices are skipped instead of raising.

        Returns
        -------
        str
            The decoded string.
        """
        chars: List[str] = []
        for idx in indices:
            idx = int(idx)
            if idx == self.blank_index:
                if strip_blank:
                    continue
                raise ValueError("Encountered blank index during literal decode.")
            char = self.index_to_char.get(idx)
            if char is None:
                raise ValueError(f"Index {idx} is out of range for {self.name!r}.")
            chars.append(char)
        return "".join(chars)

    def ctc_greedy_decode(self, indices: Iterable[int]) -> str:
        """Collapse a raw per-timestep argmax path into a string (CTC rule).

        Applies the standard CTC collapse function ``B``: first merge runs of
        identical consecutive labels, then remove blanks.

        Parameters
        ----------
        indices:
            Per-timestep argmax indices of length ``T`` (the network's greedy
            path over the CTC logits).

        Returns
        -------
        str
            The decoded, de-duplicated string.
        """
        chars: List[str] = []
        previous = None
        for idx in indices:
            idx = int(idx)
            if idx != previous:
                if idx != self.blank_index:
                    chars.append(self.index_to_char[idx])
                previous = idx
        return "".join(chars)


def build_vocabulary(name: str) -> Vocabulary:
    """Factory for the two project-standard alphabets.

    Parameters
    ----------
    name:
        Either ``"numeric"`` (10 classes) or ``"alphanumeric"`` (62 classes).

    Returns
    -------
    Vocabulary
        A configured vocabulary with blank at index 0.

    Raises
    ------
    KeyError
        If ``name`` is not a recognized preset.
    """
    key = name.lower().strip()
    if key not in _PRESET_ALPHABETS:
        valid = ", ".join(sorted(_PRESET_ALPHABETS))
        raise KeyError(f"Unknown alphabet {name!r}. Expected one of: {valid}.")
    return Vocabulary(alphabet=_PRESET_ALPHABETS[key], blank_index=0, name=key)
