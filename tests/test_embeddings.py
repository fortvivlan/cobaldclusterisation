from cobaldclusterisation.embeddings import (
    _coerce_rope_parameter_floats,
    _ensure_padding_token,
    _model_load_kwargs,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"
        self.unk_token = "<unk>"


def test_ensure_padding_token_uses_eos_fallback() -> None:
    tokenizer = DummyTokenizer()

    _ensure_padding_token(tokenizer)

    assert tokenizer.pad_token == "</s>"


def test_model_load_kwargs_uses_dtype_keyword() -> None:
    dtype = object()

    kwargs = _model_load_kwargs(trust_remote_code=True, dtype=dtype)

    assert kwargs == {"trust_remote_code": True, "dtype": dtype}
    assert "torch_dtype" not in kwargs


def test_coerce_rope_parameter_floats() -> None:
    rope_parameters = {
        "rope_type": "yarn",
        "beta_fast": 32,
        "beta_slow": 1,
        "nested": {"beta_fast": 16, "beta_slow": 2.5},
    }

    _coerce_rope_parameter_floats({"rope_parameters": rope_parameters})

    assert rope_parameters["beta_fast"] == 32.0
    assert rope_parameters["beta_slow"] == 1.0
    assert rope_parameters["nested"]["beta_fast"] == 16.0
    assert rope_parameters["nested"]["beta_slow"] == 2.5
