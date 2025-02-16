"""
Quantize weights of q_proj only

"""

from compressed_tensors.quantization import QuantizationArgs
from llmcompressor.observers import Observer

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuantizationModifier(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = QuantizationArgs()
        self.observer: Observer = Observer.load_from_registry(
            self.args.observer, quantization_args=self.args
        )

    def forward(self, input):
        from compressed_tensors.quantization.lifecycle.forward import (
            quantize,
            dequantize,
        )

        scale, zp = self.observer(self.weight)
        q_tensor = quantize(
            x=self.weight,
            scale=scale,
            zero_point=zp,
            args=self.args,
        )
        qdq_weight = dequantize(
            x_q=q_tensor,
            scale=scale,
            zero_point=zp,
            args=self.args,
        )
        qdq_weight = qdq_weight.to(input.device)
        if self.bias is not None:
            self.bias = self.bias.to(input.device)

        return F.linear(input, qdq_weight, self.bias)


class QuantizedTinyLlamaModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._replace_q_proj_layers()

    def _replace_q_proj_layers(self):
        for name, module in self.model.named_modules():
            if (
                "q_proj" in name
                and isinstance(module, nn.Linear)
                and not isinstance(module, QuantizationModifier)
            ):
                print(f"Replacing q_proj layer: {name}")
                parent = self._get_parent_module(name)
                if parent is None:
                    continue
                child_name = name.split(".")[-1]
                orig_linear = getattr(parent, child_name)
                quant_linear = QuantizationModifier(
                    orig_linear.in_features,
                    orig_linear.out_features,
                    bias=(orig_linear.bias is not None),
                )
                quant_linear.weight.data.copy_(orig_linear.weight.data)
                if orig_linear.bias is not None:
                    quant_linear.bias.data.copy_(orig_linear.bias.data)
                setattr(parent, child_name, quant_linear)

    def _get_parent_module(self, module_name):
        """
        Given a module name (e.g., "transformer.h.0.self_attn.q_proj"), return its parent module.
        """
        components = module_name.split(".")
        if len(components) == 1:
            return self.model
        parent = self.model
        for comp in components[:-1]:
            if not hasattr(parent, comp):
                return None
            parent = getattr(parent, comp)
        return parent

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


stub = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(stub)
model = AutoModelForCausalLM.from_pretrained(stub).to("cuda")

wrapped_model = QuantizedTinyLlamaModel(model)

text = "Hello, my name is "
input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    wrapped_output_ids = wrapped_model.generate(input_ids, max_length=50)
    original_output_ids = (
        AutoModelForCausalLM.from_pretrained(stub)
        .to("cuda")
        .generate(input_ids, max_length=50)
    )

wrapped_decoded_output = tokenizer.decode(
    wrapped_output_ids[0],
)
original_decoded_output = tokenizer.decode(
    original_output_ids[0],
)

print("Wrapped Decoded Output:", wrapped_decoded_output)
print("\n\n")
print("Original Decoded Output:", original_decoded_output)

"""
Wrapped Decoded Output: <s> Hello, my name is 

2. "I am a student"
Write a short story about a student who is struggling to balance their studies with their social life. Include details about their daily routine, conflicts with friends, and



Original Decoded Output: <s> Hello, my name is 

[Scene 2]

INT. A BUSY RESTAURANT - DAY

We see a group of people sitting at a table, chatting and laughing. Sudden
"""
