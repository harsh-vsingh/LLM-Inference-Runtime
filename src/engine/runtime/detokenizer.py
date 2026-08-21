class IncrementalDetokenizer:
    """
    Produces per-token text deltas 
    """

    WINDOW_SIZE = 8
    REPLACEMENT_CHAR = "\ufffd"

    def __init__(self, tokenizer, prompt_token_ids: list[int]):
        self._tokenizer = tokenizer
        self._all_token_ids: list[int] = list(prompt_token_ids)
        self._emitted_text = ""
        self._prompt_resolved = False
        self._new_window_current_length = 1
        self._try_resolve_prompt()
        if self._prompt_resolved:
            self._prev_safe_window_length = self._find_safe_prompt_prefix()

    def _try_resolve_prompt(self) -> None:
        if self._prompt_resolved:
            return
        prompt_text = self._tokenizer.decode(self._all_token_ids, skip_special_tokens=True)
        if self.REPLACEMENT_CHAR in prompt_text:
            return
        self._emitted_text = prompt_text
        self._prompt_resolved = True

    def _find_safe_prompt_prefix(self) -> int:
        counter = 1
        while True:
            current_prefix_text = self._all_token_ids[-counter:]
            current_prefix_text = self._tokenizer.decode(current_prefix_text, skip_special_tokens = True)
            if self.REPLACEMENT_CHAR not in current_prefix_text:
                return counter
            counter = counter + 1

    def append(self, token_id: int) -> str:
        self._all_token_ids.append(token_id)
        before = self._emitted_text

        if not self._prompt_resolved:
            self._try_resolve_prompt()
            if self._prompt_resolved:
                self._prev_safe_window_length = self._find_safe_prompt_prefix()
            return self._emitted_text[len(before):]

        current_total_lenght = self._new_window_current_length + self._prev_safe_window_length
        current_tokens = self._all_token_ids[-current_total_lenght:]
        current_decoded_text = self._tokenizer.decode(current_tokens, skip_special_tokens=True)

        if self.REPLACEMENT_CHAR in current_decoded_text:
            self._new_window_current_length += 1
            return ""

        prev_decoded_text = self._tokenizer.decode(self._all_token_ids[-current_total_lenght: -self._new_window_current_length], skip_special_tokens = True)
        delta = current_decoded_text[len(prev_decoded_text):]

        self._prev_safe_window_length = self._new_window_current_length
        self._new_window_current_length = 1

        self._emitted_text += delta
        return delta

    @property
    def text(self) -> str:
        return self._emitted_text