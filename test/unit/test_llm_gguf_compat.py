import unittest
from unittest.mock import patch

from tinygrad import nn
from tinygrad.llm.model import Transformer, TransformerConfig


class TestLegacyLlamaGGUF(unittest.TestCase):
  def test_defaults_missing_kv_head_count_and_rope_base(self):
    config = TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=32, head_dim=4, rope_theta=10000.0, rope_dim=4, v_head_dim=4, max_context=16)
    state_dict = nn.state.get_state_dict(Transformer(config))
    kv = {
      "general.architecture": "llama", "llama.context_length": 16, "llama.attention.head_count": 2,
      "llama.block_count": 1, "llama.embedding_length": 8, "llama.feed_forward_length": 16,
      "llama.attention.layer_norm_rms_epsilon": 1e-5, "llama.rope.dimension_count": 4,
      "tokenizer.ggml.tokens": [str(i) for i in range(32)],
    }

    with patch("tinygrad.llm.model.gguf_load", return_value=(kv, state_dict)):
      model, _ = Transformer.from_gguf("unused.gguf", max_context=16)

    self.assertEqual(model.blk[0].config.n_kv_heads, 2)
    self.assertEqual(model.blk[0].config.rope_theta, 10000.0)


if __name__ == "__main__": unittest.main()
