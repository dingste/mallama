# Call: source /home/dieter/Entwicklung/HysteroGrad/venv/bin/activate && PYTHONPATH=/home/dieter/Entwicklung/HysteroGrad python gguf_schaerfen.py

# Ein modus-sensitives Sprachinstrument, bei dem Rollen nicht erklärt, sondern angespielt werden.
# Überzeichnung → Sampling + Stil / Konsistenz → früher Pfad + early-exit

import os
import sys

# Add HysteroGrad to path (sibling directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
hysterograd_path = os.path.join(os.path.dirname(project_root), "HysteroGrad")
if os.path.exists(hysterograd_path):
    sys.path.append(hysterograd_path)

import torch
import numpy as np
from llama_cpp import Llama
from hysterograd import HIOptimizer
from gguf import GGUFWriter
import argparse
import json

# --- Konfiguration ---
# Pfade relativ zum Skript (das in scripts/ liegt)
BASE_DIR = project_root
model_name = "mistral-7B_q8"
my_lib_path = os.path.join(BASE_DIR, "bin/libllama.so")
mod_path = os.path.join(BASE_DIR, "export/models/base/mistral-7B_q8.gguf")

# Argument Parser Setup
parser = argparse.ArgumentParser(description='Process Persona JSON and create LoRA.')
parser.add_argument('--config', type=str, required=True, help='Path to the persona configuration JSON file')
args = parser.parse_args()

# Load Configuration
try:
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file not found at {args.config}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {args.config}")
    sys.exit(1)

if "llm_meta" not in config_data:
    print("Error: 'llm_meta' section missing in configuration file.")
    sys.exit(1)

personas = config_data["llm_meta"]

# 1. GGUF Modell laden (nur einmal nötig)
print(f"Lade Basis-Modell: {mod_path}")
llm = Llama(model_path=mod_path, n_gpu_layers=-1, lib_path=my_lib_path, embedding=True, verbose=False)
base_dim = llm.n_embd()
n_layers = 32 
rank = 8
EARLY_LAYERS = 6

# 2. Hilfsklasse für einen einzelnen LoRA-Block
class LoRALayer(torch.nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        # A wird klein initialisiert, B mit Nullen (LoRA-Standard)
        self.lora_a = torch.nn.Parameter(torch.randn(rank, dim) * 0.005)
        self.lora_b = torch.nn.Parameter(torch.zeros(dim, rank))

    def forward(self, x):
        return (x @ self.lora_a.T) @ self.lora_b.T

# 3. Das Gesamtmodell (angepasst für Personas)
class PersonaAdapter(torch.nn.Module):
    def __init__(self, n_layers, dim, rank):
        super().__init__()
        # PyTorch erlaubt keine Punkte in Namen, daher blk_{i} statt blk.{i}
        self.layers = torch.nn.ModuleDict({
            f"blk_{i}": LoRALayer(dim, rank) for i in range(n_layers)
        })

    def forward(self, x):
        correction = torch.zeros_like(x)
        for name, layer in self.layers.items():
            correction += layer(x)
        return x + correction

# --- Hilfsfunktion für saubere Embeddings ---
def get_clean_emb(text):
    res = llm.create_embedding(text)['data'][0]['embedding']
    tensor = torch.tensor(res, dtype=torch.float32)
    # Falls das Resultat 2D ist (n_tokens, dim), mitteln wir über die Token
    if tensor.ndim > 1:
        tensor = torch.mean(tensor, dim=0)
    return tensor

# --- Hilfsfunktion für Orthogonalisierung ---
def orthogonalize(target_vec, noise_vec):
    # Sicherstellen, dass beide Vektoren 1D sind für dot product
    t = target_vec.flatten()
    n = noise_vec.flatten()
    
    # Einheitsvektor des Rauschens berechnen
    norm_n = torch.norm(n)
    if norm_n < 1e-10:
        return t
    
    unit_noise = n / (norm_n + 1e-8)
    
    # Projektion von t auf n berechnen
    projection = torch.dot(t, unit_noise) * unit_noise
    
    # Das Ergebnis ist der Teil von t, der senkrecht auf n steht
    return t - projection

# --- Optimierter Training Loop ---
for p_id, p_data in personas.items():
    if "trigger" not in p_data:
        print(f"Skipping non-persona entry: {p_id}")
        continue
    print(f"\n--- Training Expert: {p_id} ---")
    adapter = PersonaAdapter(n_layers, base_dim, rank)
    optimizer = HIOptimizer(adapter.parameters(), lr=4e-4)

    # 1. Antagonist Embedding als "Verbots-Zone"
    anti_emb = get_clean_emb(p_data['antagonist'])

    for epoch in range(15): # Etwas mehr Zeit für die Orthogonalisierung
        epoch_loss = 0
        for text in [p_data['trigger'], p_data['style']]:
            optimizer.zero_grad()
            raw_emb = get_clean_emb(text)
            output = adapter(raw_emb)
            ortho_base = orthogonalize(raw_emb, anti_emb)
            norm_ortho = torch.norm(ortho_base)
            early_boost = torch.tanh(torch.norm(raw_emb) / 8.0)
            target = ortho_base + early_boost * (ortho_base / (norm_ortho + 1e-6)) * p_data['margin']
            loss = torch.nn.functional.mse_loss(output, target)
            stability_loss = torch.norm(output - raw_emb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Loss {epoch_loss/2:.8f}")

    # 4. GGUF Export für diese Persona
    output_dir = os.path.join(BASE_DIR, "export/models/sts")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, p_data['output_file'])
    print(f"Exportiere {p_id} nach {out_path}...")
    
    writer = GGUFWriter(out_path, "llama")
    writer.add_string("general.type", "adapter")
    writer.add_string("general.name", f"HIO-{p_id}")
    writer.add_string("adapter.type", "lora") 
    writer.add_string("adapter.base_model.name", model_name)
    writer.add_uint32("adapter.lora.r", rank)
    writer.add_float32("adapter.lora.alpha", 16.0)

    for i in range(n_layers):
        if i >= EARLY_LAYERS:
            continue

        for target in ["attn_q", "attn_output"]:
            layer_key = f"blk_{i}"
            layer = adapter.layers[layer_key]
            base_name = f"blk.{i}.{target}.weight"
        
            wa = layer.lora_a.detach().numpy().astype(np.float32)
            wb = layer.lora_b.detach().numpy().astype(np.float32)
        
            writer.add_tensor(f"{base_name}.lora_a", wa)
            writer.add_tensor(f"{base_name}.lora_b", wb)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Fertig mit {p_id}.")

print(f"\nAlle Personas trainiert und exportiert.")
