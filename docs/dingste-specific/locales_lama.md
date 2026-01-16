# llama

## bau von llama.cpp
hängt stark von HW ab!!!! da keine gute gfx im laptop:
cmake -B build -DGGML_HIPBLAS=0 \
cmake --build build --config Release -- -j 16

## ausführung
### als server (127.0.0.1:8080):
build/bin/llama-server -m ../localllama/mistralai/Mistral-7B-v0.1/Mistral-7B-v0.1-F16.gguf
### CLI
llama.cpp/build/bin/llama-cli -m models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q8_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e
-n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt


## model herunterladen
(environment nicht vergessen)
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"  --local-dir models/Meta-Llama-3-8B-Instruct

## model konvertieren
(environment nicht vergessen)
f16 ist schon?
./convert_hf_to_gguf.py ../localmodels/models/Meta-Llama-3-8B-Instruct --outtype {f32,f16,bf16,q8_0,auto}


# auf ki knecht
ROCR_VISIBLE_DEVICES=0 llama-server -m models/Meta-Llama-3-8B-Instruct-Q8_0.gguf -ngl 100 --no-warmup --host 0.0.0.0


## server mit early-exit patch und 2 trained persona-paths
llama.cpp/bin/llama-server \
  -m models/mistral-7B_q8.gguf \
  --lora models/hio_p_usch.gguf \
  --lora models/hio_p_prof.gguf \
  --early-exit --early-exit-gap 16.0 \
  --embeddings --pooling cls \
  --port 8080 --mlock

### Fehler: 
* Meta ausgabe unabhängig von response (macht das regex/concat der response schwierig)
* 'get_embeddings_ith: invalid embeddings id 0, reason: no embeddings'
* per llama-completion gibt es andere response: > llama.cpp/bin/llama-completion -m models/mistral-7B_q8.gguf --lora models/hio_p_prof.gguf -p "Professor, was wissen sie über quantenphysik?" --early-exit --early-exit-gap 8.0 --embeddings

## python entwicklung
mit >source ../HysteroGrad/venv/bin/activate && PYTHONPATH=../HysteroGrad && python ./*.py

