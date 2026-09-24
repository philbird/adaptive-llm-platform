# Synthetic base models

The `tiny_base` session fixture generates `generated/tiny/seed-17/` at test time using
`adaptive_llm.training.lora.generate_tiny_base`. It contains a fixed-seed, random-init
two-layer Llama (32 hidden units), a 259-token byte tokenizer, a chat template, CC0 licence
and a complete SHA-256 inventory. No download or pretrained weights are used.
Generated files are ignored by Git and copied into each test's isolated data directory.
