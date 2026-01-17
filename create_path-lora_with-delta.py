import os
import sys
import torch
import numpy as np
import argparse
import json
import subprocess
from llama_cpp import Llama
from gguf import GGUFWriter

# --- Pfad-Logik für HysteroGrad ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# HysteroGrad Pfad (angepasst an deine Struktur)
sys.path.append(os.path.join(os.path.dirname(project_root), "HysteroGrad"))

try:
    from hysterograd import HIOptimizer
except ImportError:
    print("Warnung: HIOptimizer nicht gefunden. Fallback auf Adam.")
    HIOptimizer = torch.optim.Adam

# --- CLI Setup ---
parser = argparse.ArgumentParser(description='MGR Core: Persona LoRA Creator & Base Quantizer')
parser.add_argument('--base_model', type=str, required=True, help='Pfad zum FP16/Q8 Basis GGUF')
parser.add_argument('--persona_json', type=str, required=True, help='Pfad zur Persona Konfiguration')
parser.add_argument('--quantize', action='store_true', help='Erstelle zusätzlich ein Q4_K_M Basismodell')
args = parser.parse_args()

# --- Konfiguration laden ---
with open(args.persona_json, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
personas = config_data["llm_meta"]

# Verzeichnisse vorbereiten
output_dir = os.path.dirname(os.path.abspath(args.base_model))
model_name_base = os.path.basename(args.base_model).replace(".gguf", "")

# --- 1. Optionales MGR-Basis-Quantizing (Q4_K_M) ---
if args.quantize:
    q4_path = os.path.join(output_dir, f"{model_name_base}_MGR_Q4.gguf")
    print(f"\n--- MGR: Erzeuge niedrigquantisierte Basis (Q4_K_M) ---")
    if not os.path.exists(q4_path):
        # Wir rufen llama-quantize auf (muss im PATH sein oder im build/bin liegen)
        try:
            subprocess.run(["llama-quantize", args.base_model, q4_path, "Q4_K_M"], check=True)
            print(f"Basis erstellt: {q4_path}")
        except Exception as e:
            print(f"Quantisierung fehlgeschlagen (Ist llama-quantize im PATH?): {e}")
    else:
        print(f"Q4-Basis existiert bereits: {q4_path}")

# --- 2. Basis-Modell für Embedding-Extraktion laden ---
print(f"\nLade Basis für Embedding-Extraktion: {args.base_model}")
llm = Llama(model_path=args.base_model, embedding=True, verbose=False)
base_dim = llm.n_embd()
n_layers = 32 
rank = 8
EARLY_LAYERS = 6 # MGR Fokus auf frühe kausale Layer

# --- LoRA & Adapter Klassen ---
class LoRALayer(torch.nn.Module):
    def __init__(self, dim, rank=8):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.randn(rank, dim) * 0.005)
        self.lora_b = torch.nn.Parameter(torch.zeros(dim, rank))
    def forward(self, x):
        return (x @ self.lora_a.T) @ self.lora_b.T

class PersonaAdapter(torch.nn.Module):
    def __init__(self, n_layers, dim, rank):
        super().__init__()
        self.layers = torch.nn.ModuleDict({
            f"blk_{i}": LoRALayer(dim, rank) for i in range(n_layers)
        })
    def forward(self, x):
        correction = torch.zeros_like(x)
        for name, layer in self.layers.items():
            correction += layer(x)
        return x + correction

def get_clean_emb(text):
    res = llm.create_embedding(text)['data'][0]['embedding']
    return torch.tensor(res, dtype=torch.float32)

def orthogonalize(target_vec, noise_vec):
    t, n = target_vec.flatten(), noise_vec.flatten()
    norm_n = torch.norm(n)
    if norm_n < 1e-10: return t
    unit_noise = n / (norm_n + 1e-8)
    return t - torch.dot(t, unit_noise) * unit_noise

# --- Training & Export Loop ---
for p_id, p_data in personas.items():
    if "trigger" not in p_data: continue
    print(f"\n--- MGR-Training: {p_id} (Kausale Schärfung) ---")
    
    adapter = PersonaAdapter(n_layers, base_dim, rank)
    optimizer = HIOptimizer(adapter.parameters(), lr=4e-4)
    anti_emb = get_clean_emb(p_data['antagonist'])

    for epoch in range(15):
        optimizer.zero_grad()
        raw_emb = get_clean_emb(p_data['trigger'])
        output = adapter(raw_emb)
        
        ortho_base = orthogonalize(raw_emb, anti_emb)
        norm_ortho = torch.norm(ortho_base)
        early_boost = torch.tanh(torch.norm(raw_emb) / 8.0)
        
        target = ortho_base + early_boost * (ortho_base / (norm_ortho + 1e-6)) * p_data['margin']
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0: print(f"  Epoch {epoch}: Loss {loss.item():.8f}")

    # --- Export ---
    out_path = os.path.join(output_dir, f"MGR_Adapter_{p_id}.gguf")
    writer = GGUFWriter(out_path, "llama")
    writer.add_string("general.type", "adapter")
    writer.add_string("general.name", f"MGR-{p_id}")
    writer.add_string("adapter.type", "lora") 
    writer.add_string("adapter.base_model.name", model_name_base)
    writer.add_uint32("adapter.lora.r", rank)
    writer.add_float32("adapter.lora.alpha", 16.0)

    # Fokus auf kausale Layer für Early-Exit Strategie
    for i in range(n_layers):
        if i >= EARLY_LAYERS: continue
        for target in ["attn_q", "attn_output"]:
            layer = adapter.layers[f"blk_{i}"]
            base_name = f"blk.{i}.{target}.weight"
            writer.add_tensor(f"{base_name}.lora_a", layer.lora_a.detach().numpy().astype(np.float32))
            writer.add_tensor(f"{base_name}.lora_b", layer.lora_b.detach().numpy().astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"MGR-LoRA exportiert: {out_path}")

print(f"\nProzess abgeschlossen. Alle Dateien im Verzeichnis: {output_dir}")
