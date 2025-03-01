from datasets import load_dataset
from .quantize import GPTQ, QuantizationConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quant_config = QuantizationConfig(bits=4, group_size=128)

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

model = GPTQ.from_pretrained(model_id, quant_config)

model.quantize(calibration_dataset, batch_size=2)

save_path = model_id.replace("/")[-1] + "-gptq"
model.save(save_path)

model = GPTQ.load(save_path)
result = model.generate("The capital of Japan is ")[0] 
print(model.tokenizer.decode(result)) 