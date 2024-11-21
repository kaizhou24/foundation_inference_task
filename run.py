from data_loader import DataLoaderGSM8K, DataLoaderMME
from inference import run_llm_inference, run_vlm_inference
from transformers import AutoModelForCausalLM, AutoTokenizer, VisionEncoderDecoderModel, AutoProcessor

# Load models and tokenizers
llm_model = AutoModelForCausalLM.from_pretrained("path/to/qwen2-1.5b").to("cuda")
llm_tokenizer = AutoTokenizer.from_pretrained("path/to/qwen2-1.5b")

vlm_model = VisionEncoderDecoderModel.from_pretrained("path/to/minigpt-4").to("cuda")
vlm_processor = AutoProcessor.from_pretrained("path/to/minigpt-4")

# Load data
gsm8k_loader = DataLoaderGSM8K()
mme_loader = DataLoaderMME("path/to/mme/images")

# Run inference
llm_data = gsm8k_loader.get_data()
run_llm_inference(llm_model, llm_tokenizer, llm_data)

vlm_data = mme_loader.get_data()
run_vlm_inference(vlm_model, vlm_processor, vlm_data)
