"""语义指令匹配：bge-small-zh ONNX 句向量 + 余弦相似度。

相比关键词匹配，可识别自由表述（"宝贝往前走两步"）。
模型来自 HuggingFace Xenova/bge-small-zh-v1.5（MIT），首次使用自动下载。
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from lite3_sim.policy import VelocityCommand
from lite3_sim.voice import Action, MotionCommand, _to_simplified

MODEL_REPO = "Xenova/bge-small-zh-v1.5"
_MODEL_FILE = "onnx/model_quantized.onnx"
_TOKENIZER_FILE = "tokenizer.json"

# 每条指令的语义原型例句（简体）
_PROTOTYPES: tuple[tuple[MotionCommand, tuple[str, ...]], ...] = (
    (MotionCommand(Action.MOVE, VelocityCommand(vx=0.4)), ("前进", "向前走", "直行", "往前走")),
    (MotionCommand(Action.MOVE, VelocityCommand(vx=-0.3)), ("后退", "退后", "往后走")),
    (MotionCommand(Action.MOVE, VelocityCommand(wz=0.5)), ("左转", "向左转", "往左拐")),
    (MotionCommand(Action.MOVE, VelocityCommand(wz=-0.5)), ("右转", "向右转", "往右拐")),
    (MotionCommand(Action.STAND), ("停下", "停止", "别动", "起立", "站起来", "站住")),
    (MotionCommand(Action.LIE), ("趴下", "躺下", "卧倒")),
)

_FLOAT32 = np.float32


def _download(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MODEL_REPO, filename))


class SemanticCommandParser:
    """语义解析器：转写文本 → 最近指令原型（余弦相似度超阈值）。"""

    def __init__(self, threshold: float = 0.6) -> None:
        from tokenizers import Tokenizer

        self._threshold = threshold
        self._tokenizer = Tokenizer.from_file(str(_download(_TOKENIZER_FILE)))
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(
            str(_download(_MODEL_FILE)), providers=["CPUExecutionProvider"]
        )
        self._commands: list[MotionCommand] = []
        phrases: list[str] = []
        for command, examples in _PROTOTYPES:
            for phrase in examples:
                self._commands.append(command)
                phrases.append(phrase)
        self._prototype_embeddings = self._embed(phrases)

    def _embed(self, texts: list[str]) -> NDArray[_FLOAT32]:
        """mean pooling + L2 归一化的句向量。"""
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_types = np.zeros_like(input_ids)
        (token_embeddings,) = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention,
                "token_type_ids": token_types,
            },
        )
        mask = attention[..., None].astype(_FLOAT32)
        pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / norms).astype(_FLOAT32)

    def parse(self, text: str) -> MotionCommand | None:
        """返回语义最近的指令；最高相似度低于阈值返回 None。"""
        (query,) = self._embed([_to_simplified(text)])
        scores = self._prototype_embeddings @ query
        best = int(np.argmax(scores))
        if float(scores[best]) < self._threshold:
            return None
        return self._commands[best]

    def score(self, text: str) -> float:
        """调试用：返回当前文本的最高相似度。"""
        (query,) = self._embed([_to_simplified(text)])
        return float(np.max(self._prototype_embeddings @ query))


__all__ = ["SemanticCommandParser"]
