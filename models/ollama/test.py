# Load model directly
from transformers import AutoProcessor, AutoModelForPreTraining

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-90B-Vision")
model = AutoModelForPreTraining.from_pretrained("meta-llama/Llama-3.2-90B-Vision")

g = 1