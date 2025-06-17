from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# 🔧 Verzeichnisse (anpassen, falls du andere Namen verwendest)
base_model_path = "../base_model/Llama-3.2-3B-Instruct"
lora_path = "../lora_adapter"
output_path = "../output/merged_model"

# Modell laden
print("🔄 Lade Basis-Modell...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA-Adapter laden
print("🔌 Lade LoRA-Adapter...")
model = PeftModel.from_pretrained(model, lora_path)

# Merge durchführen
print("🧠 Merge LoRA → Base...")
model = model.merge_and_unload()

# Speichern
print("💾 Speichere das gemergte Modell...")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print(f"✅ Merge abgeschlossen! Gespeichert in: {output_path}")
