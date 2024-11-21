from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def run_llm_inference(model, tokenizer, dataset):
    for example in dataset:
        input_text = example["question"]
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs)
        print(f"Q: {input_text}\nA: {tokenizer.decode(outputs[0])}\n")


from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
from PIL import Image

def run_vlm_inference(model, processor, dataset):
    for image_path in dataset:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs)
        print(f"Image: {image_path}\nCaption: {processor.decode(outputs[0])}\n")
