from cobaldclusterisation.embeddings import _ensure_padding_token


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"
        self.unk_token = "<unk>"


def test_ensure_padding_token_uses_eos_fallback() -> None:
    tokenizer = DummyTokenizer()

    _ensure_padding_token(tokenizer)

    assert tokenizer.pad_token == "</s>"
