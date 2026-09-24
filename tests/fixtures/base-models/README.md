# Synthetic base models

The `tiny_base` session fixture generates `generated/tiny/seed-17/` at test time using
`adaptive_llm.training.lora.generate_tiny_base`. It contains a fixed-seed, random-init
two-layer Llama (32 hidden units), a 259-token byte tokenizer, a chat template, CC0 licence
and a complete SHA-256 inventory. No download or pretrained weights are used.
Generated files are ignored by Git and copied into each test's isolated data directory.

Slice 5a's `student_base` fixture also generates `generated/tiny-student/seed-17/` with
`generate_tiny_base(..., student=True)`: one layer and 16 hidden units, using the same byte
vocabulary/template and fixed seed. The trainer records the verified student architecture and
parameter count and, for real teachers, requires fewer parameters than the teacher's verified base
inventory. Tests also generate different geometries; admission does not require this fixture's shape.
