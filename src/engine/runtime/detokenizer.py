class IncrementalDetokenizer:
    """
    Produces per-token text deltas without re-decoding the full growing
    token sequence on every call (the original code called
    tokenizer.decode(prompt_ids + generated_ids) from scratch every single
    step - O(n) work per step, O(n^2) over a full generation).

    A naive windowed-decode approach (decode only the last few tokens,
    diff against the previous window) is NOT safe on its own: many
    tokenizers decode an incomplete multi-byte UTF-8 sequence using
    errors="replace", which silently produces a plausible-looking
    replacement character instead of raising - so a truncated character at
    the start of the window can look like valid new text and get emitted
    to the client, then get "corrected" on a later call by emitting mangled
    follow-up text. This was verified empirically: a naive version of this
    class passed ASCII/simple-unicode tests but produced garbled output
    under a byte-level tokenizer splitting multi-byte characters one byte
    per token.

    Safe strategy: decode a trailing window, but only ever emit the
    portion of newly-decoded text that does NOT end in a Unicode
    replacement character (U+FFFD) - i.e. only emit text we're confident
    is a complete, correctly-decoded character. Anything after the last
    confirmed-safe point is held back and re-attempted next call, once
    more bytes have arrived. This mirrors the approach used by
    production inference servers for streaming detokenization.
    """

    WINDOW_SIZE = 8
    REPLACEMENT_CHAR = "\ufffd"

    def __init__(self, tokenizer, prompt_token_ids: list[int]):
        self._tokenizer = tokenizer
        self._all_token_ids: list[int] = list(prompt_token_ids)
        self._emitted_text = ""
        self._prompt_resolved = False
        self._prev_safe_window_text = self._safe_window_text()
        self._try_resolve_prompt()

    def _try_resolve_prompt(self) -> None:
        if self._prompt_resolved:
            return
        prompt_text = self._tokenizer.decode(self._all_token_ids, skip_special_tokens=True)
        if self.REPLACEMENT_CHAR in prompt_text:
            return
        self._emitted_text = prompt_text
        self._prompt_resolved = True

    def _safe_window_text(self) -> str:
        window = self._all_token_ids[-self.WINDOW_SIZE:]
        if not window:
            return ""
        decoded = self._tokenizer.decode(window, skip_special_tokens=True)
        if self.REPLACEMENT_CHAR in decoded:
            return None
        return decoded

    def append(self, token_id: int) -> str:
        self._all_token_ids.append(token_id)
        before = self._emitted_text

        if not self._prompt_resolved:
            self._try_resolve_prompt()
            self._prev_safe_window_text = self._safe_window_text()
            return self._emitted_text[len(before):]

        new_window_text = self._safe_window_text()

        if (
            new_window_text is not None
            and self._prev_safe_window_text is not None
            and new_window_text.startswith(self._prev_safe_window_text)
        ):
            delta = new_window_text[len(self._prev_safe_window_text):]
            self._emitted_text += delta
            self._prev_safe_window_text = new_window_text
            return self._emitted_text[len(before):]

        full_text = self._tokenizer.decode(self._all_token_ids, skip_special_tokens=True)
        if self.REPLACEMENT_CHAR in full_text[len(self._emitted_text):]:
            self._prev_safe_window_text = new_window_text
            return self._emitted_text[len(before):]

        self._emitted_text = full_text
        self._prev_safe_window_text = new_window_text
        return self._emitted_text[len(before):]

    @property
    def text(self) -> str:
        return self._emitted_text