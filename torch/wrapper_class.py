"""
Quantize weights of q_proj only

"""

from compressed_tensors.quantization import QuantizationArgs
from llmcompressor.observers import Observer

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from llmcompressor.observers import Observer


class QuantizationModifier(nn.Module):
    def __init__(self, module: nn.Module):

        super().__init__()
        self.module = module

        # self.args = QuantizationArgs(num_bits=4) # 
        self.args = QuantizationArgs(num_bits=8) # loss-less
        
        self.observer = Observer.load_from_registry(
            self.args.observer, quantization_args=self.args
        )

        orig_weight = module.weight
        scale, zero_point = self.observer(orig_weight)
        q_tensor = quantize(
            x=orig_weight, scale=scale, zero_point=zero_point, args=self.args
        )
        qdq_weight = dequantize(
            x_q=q_tensor, scale=scale, zero_point=zero_point, args=self.args
        )

        # update weight
        module.weight.data.copy_(qdq_weight.to(orig_weight.device))

    @property
    def weight(self):
        return self.module.weight

    def forward(self, *args, **kwargs):
        # print("wrapped module inference")

        # breakpoint()
        # print(self.module.weight)
        # """
        # (Pdb) self.module.weight
        # Parameter containing:
        # tensor([[-0.0000, -0.0000, -0.0125,  ...,  0.0000, -0.0000, -0.0125],
        #         [ 0.0000,  0.0000, -0.0125,  ...,  0.0000,  0.0000,  0.0125],
        #         [-0.0000,  0.0000, -0.0249,  ...,  0.0125,  0.0000, -0.0125],
        #         ...,
        #         [ 0.0125, -0.0000,  0.0125,  ..., -0.0000,  0.0249, -0.0125],
        #         [-0.0125, -0.0000, -0.0125,  ...,  0.0000, -0.0125,  0.0125],
        #         [-0.0125, -0.0000, -0.0125,  ...,  0.0000, -0.0125,  0.0125]],
        #     device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
        # """
        return self.module(*args, **kwargs)


class QuantizedTinyLlamaModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._replace_q_proj_layers()

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

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

                # overwrite weight with qdq weight
                quant_linear = QuantizationModifier(orig_linear)
                setattr(parent, child_name, quant_linear)

    def _get_parent_module(self, module_name: str) -> Optional[nn.Module]:
        components = module_name.split(".")
        if len(components) == 1:
            return self.model
        parent = self.model
        for comp in components[:-1]:
            if not hasattr(parent, comp):
                return None
            parent = getattr(parent, comp)
        return parent


stub = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(stub)
model = AutoModelForCausalLM.from_pretrained(
    stub,
    device_map="auto",
    torch_dtype="auto",
)

"""debugging
original_model.model.layers[0].self_attn.q_proj.weight
wrapped_model.model.model.layers[0].self_attn.q_proj.weight
"""

print("Before wrapping:", model.model.layers[0].self_attn.q_proj.weight.shape)

# wrap the model. pass by ref, model is also wrapped
wrapped_model = QuantizedTinyLlamaModel(model)
print(
    "After wrapping:", wrapped_model.model.model.layers[0].self_attn.q_proj.weight.shape
)

original_model = AutoModelForCausalLM.from_pretrained(
    stub,
    device_map="auto",
    torch_dtype="auto",
)

print("original weight: ", original_model.model.layers[0].self_attn.q_proj.weight)
print("wrapped weight: ", wrapped_model.model.model.layers[0].self_attn.q_proj.weight)

"""
tensor([[-0.0015, -0.0024, -0.0074,  ...,  0.0055, -0.0011, -0.0137],
        [ 0.0027,  0.0060, -0.0178,  ...,  0.0008,  0.0006,  0.0105],
        [-0.0003,  0.0019, -0.0187,  ...,  0.0073,  0.0018, -0.0109],
        ...,
        [ 0.0148, -0.0015,  0.0109,  ..., -0.0060,  0.0193, -0.0144],
        [-0.0165, -0.0023, -0.0067,  ...,  0.0037, -0.0181,  0.0139],
        [-0.0161, -0.0019, -0.0074,  ...,  0.0039, -0.0179,  0.0140]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
wrapped weight:  Parameter containing:
tensor([[-0.0000, -0.0000, -0.0125,  ...,  0.0000, -0.0000, -0.0125],
        [ 0.0000,  0.0000, -0.0125,  ...,  0.0000,  0.0000,  0.0125],
        [-0.0000,  0.0000, -0.0249,  ...,  0.0125,  0.0000, -0.0125],
        ...,
        [ 0.0125, -0.0000,  0.0125,  ..., -0.0000,  0.0249, -0.0125],
        [-0.0125, -0.0000, -0.0125,  ...,  0.0000, -0.0125,  0.0125],
        [-0.0125, -0.0000, -0.0125,  ...,  0.0000, -0.0125,  0.0125]],
       device='cuda:0', dtype=torch.bfloat16, requires_grad=True)

"""

input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    wrapped_output_ids = wrapped_model.generate(input_ids, max_length=150)
    original_output_ids = original_model.generate(input_ids, max_length=150)

wrapped_decoded_output = tokenizer.decode(wrapped_output_ids[0])
original_decoded_output = tokenizer.decode(original_output_ids[0])

print("Wrapped Decoded Output:\n", wrapped_decoded_output)
print("\nOriginal Decoded Output:\n", original_decoded_output)


""" num_bits=8 -> loss-less quantization

Wrapped Decoded Output:
 <s> Hello my name is John Smith and I am a software engineer. I have been working in the software industry for the past 5 years and have experience in developing web applications using various technologies such as Java, JavaScript, and HTML. I am

Original Decoded Output:
 <s> Hello my name is John Smith and I am a software engineer. I have been working in the software industry for the past 5 years and have experience in developing web applications using various technologies such as Java, JavaScript, and HTML. I am

"""

""" num_bits=4 -> cannot recapitulate the sample distribution
Wrapped Decoded Output:
 <s> Hello my name is John Doe.

2. A man and a woman are walking down the street when they see a man with a dog.

3. A man and a woman are walking down the street when they see a man with a dog.

4. A man and a woman are walking down the street when they see a man with a dog.

5. A man and a woman are walking down the street when they see a man with a dog.

6. A man and a woman are walking down the street when they see a man with a dog.

7. A man and a woman are walking down the street when they see a man with a dog.

8

Original Decoded Output:
 <s> Hello my name is John Smith and I am a software engineer. I have been working in the software industry for the past 5 years and have experience in developing web applications using various technologies such as Java, JavaScript, and HTML. I am proficient in using tools such as Git, Jira, and Slack to manage projects and communicate with team members. I am also skilled in designing and implementing user-friendly interfaces using CSS and HTML. In my free time, I enjoy playing video games, reading books, and spending time with my family and friends. I am passionate about learning new technologies and staying up-to-date with the latest trends in the industry. I am looking forward to
"""
