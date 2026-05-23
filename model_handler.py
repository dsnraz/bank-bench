"""
模型处理器模块 — 对标 hyperbolic_memory 的 model_handler.py。

支持两种后端:
  - transformers: 本地加载 llama / qwen / deepseek 等 HuggingFace 模型
  - openai: 通过 OpenAI 兼容 API 调用 deepseek 等远端模型
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class BaseModelHandler:
    """模型处理器基类。"""

    def load(self, model_source: str, device: str = "auto", **kwargs) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None,
                 max_new_tokens: int = 1024, **kwargs) -> str:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        raise NotImplementedError


class TransformersModelHandler(BaseModelHandler):
    """
    Transformers 本地模型处理器。
    自动检测模型类型 (qwen / llama / deepseek / mistral / chatglm)。
    """

    MODEL_TYPE_SIGNATURES = {
        "qwen": ["qwen"],
        "llama": ["llama", "vicuna", "alpaca", "belle"],
        "deepseek": ["deepseek"],
        "mistral": ["mistral"],
        "chatglm": ["chatglm"],
        "baichuan": ["baichuan"],
        "internlm": ["internlm"],
    }

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_type = "default"

    def detect_model_type(self, model_path: str) -> str:
        lower = model_path.lower()
        for t, sigs in self.MODEL_TYPE_SIGNATURES.items():
            if any(s in lower for s in sigs):
                return t
        return "default"

    def load(self, model_source: str, device: str = "auto", **kwargs) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._model_type = self.detect_model_type(model_source)
            print(f"[本地模型] {model_source}  (类型: {self._model_type})")

            tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": True, "use_fast": False}
            decoder_only = ("qwen", "llama", "deepseek", "mistral", "baichuan", "internlm")
            if self._model_type in decoder_only:
                tokenizer_kwargs["padding_side"] = "left"

            self._tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or "<|extra_0|>"
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            device_map = "auto" if device == "auto" else device
            self._model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            if (gc := getattr(self._model, "generation_config", None)) and gc.pad_token_id is None:
                gc.pad_token_id = self._tokenizer.pad_token_id
            self._model.eval()
            print("模型加载完成")
            return True
        except ImportError as e:
            print(f"缺少依赖: {e}")
            return False
        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None,
                 max_new_tokens: int = 1024, **kwargs) -> str:
        import torch

        if not self.is_loaded():
            raise RuntimeError("模型未加载")

        if self._model_type in ("qwen", "deepseek", "llama"):
            if messages is not None:
                formatted = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
        elif self._model_type == "chatglm":
            formatted = f"[Round 0]\n问：{prompt}\n答："
        else:
            formatted = prompt

        inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True)
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return self._clean_response(response)

    def batch_generate(self, prompts: List[str], max_new_tokens: int = 1024, **kwargs) -> List[str]:
        import torch

        if not self.is_loaded():
            raise RuntimeError("模型未加载")
        if len(prompts) == 0:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens=max_new_tokens, **kwargs)]

        if self._model_type in ("qwen", "deepseek", "llama"):
            formatted = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in prompts
            ]
        else:
            formatted = prompts

        inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True, padding=True)
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return [
            self._clean_response(
                self._tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            )
            for out in outputs
        ]

    def _clean_response(self, response: str) -> str:
        if self._model_type == "deepseek" and "</think>" in response:
            response = response.rsplit("</think>", 1)[-1]
        if self._model_type == "qwen" and "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        return response.strip()

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_type(self) -> str:
        return self._model_type


class OpenAICompatibleHandler(BaseModelHandler):
    """
    OpenAI 兼容 API 处理器 — 可用于 DeepSeek 等远端 API。
    自动兼容 openai 0.x (旧版 ChatCompletion.create) 和 1.x (新版 OpenAI 客户端)。
    """

    def __init__(self, api_base: str = "https://api.deepseek.com"):
        self._api_base = api_base
        self._client = None       # openai>=1.0: OpenAI 实例; openai<1.0: None
        self._model_name = None
        self._use_legacy = False  # True 表示用旧版 openai.ChatCompletion.create

    def load(self, model_source: str, device: str = "auto", **kwargs) -> bool:
        try:
            import openai as _openai
            api_key = kwargs.get("api_key") or None

            # 尝试新版 API (openai>=1.0)
            if hasattr(_openai, "OpenAI"):
                self._client = _openai.OpenAI(base_url=self._api_base, api_key=api_key)
                self._use_legacy = False
            else:
                # 旧版 API (openai<1.0)
                _openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
                if self._api_base:
                    _openai.api_base = self._api_base
                self._use_legacy = True

            self._model_name = model_source
            print(f"[API 模型] {self._model_name} @ {self._api_base}"
                  f" ({'legacy' if self._use_legacy else 'v1'})")
            return True
        except ImportError:
            print("请安装 openai: pip install openai")
            return False

    def generate(self, prompt: str, messages: Optional[List[Dict[str, str]]] = None,
                 max_new_tokens: int = 1024, **kwargs) -> str:
        if not self.is_loaded():
            raise RuntimeError("API 客户端未初始化")
        if messages is not None:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": prompt}]
        # 默认值与 MemoryBank 官方 chatgpt_config 一致
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        frequency_penalty = kwargs.get("frequency_penalty", 0.4)
        presence_penalty = kwargs.get("presence_penalty", 0.2)

        if self._use_legacy:
            import openai as _openai
            resp = _openai.ChatCompletion.create(
                model=self._model_name,
                messages=msgs,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            return resp["choices"][0]["message"]["content"]
        else:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=msgs,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            return resp.choices[0].message.content

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(p, **kwargs) for p in prompts]

    def is_loaded(self) -> bool:
        return self._client is not None or self._use_legacy


def create_model_handler(handler_type: str = "transformers", **kwargs) -> BaseModelHandler:
    if handler_type == "transformers":
        return TransformersModelHandler()
    elif handler_type == "openai":
        return OpenAICompatibleHandler(api_base=kwargs.get("api_base", "https://api.deepseek.com"))
    else:
        raise ValueError(f"未知的处理器类型: {handler_type}")
