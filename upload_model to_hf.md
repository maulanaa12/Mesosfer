# Upload Checkpoint ke HuggingFace Hub

Script untuk mengupload checkpoint training (model, optimizer state, dan metadata) ke HuggingFace Hub secara interaktif maupun via CLI.

---

## Prasyarat

### 1. Install dependensi

```bash
pip install huggingface_hub
```

### 2. Login ke HuggingFace

```bash
hf auth login
```

Masukkan token HF kamu (bisa dibuat di https://huggingface.co/settings/tokens dengan permission `write`).

---

## File yang Diupload

Setiap checkpoint terdiri dari 3 file:

| File | Deskripsi |
|------|-----------|
| `model_XXXXXX.pt` | Bobot model (base model weights) |
| `meta_XXXXXX.json` | Metadata training (step, val_bpb, loss, dll) |
| `optim_XXXXXX_rank0.pt` | Optimizer state (untuk resume training) |

> Gunakan flag `--model-only` untuk skip optimizer state (upload lebih cepat, tapi tidak bisa resume training).

---

## Cara Pakai

### Mode Interaktif (direkomendasikan)

Jalankan tanpa argumen, akan muncul menu pilihan:

```bash
python scripts/upload_checkpoint_to_hf.py
```

```
=============================================
  Upload Checkpoint ke HuggingFace Hub
=============================================
  Checkpoint dir: ~/.cache/mesosfer/base_checkpoints/d24

  Checkpoint tersedia: 5 (2,000 – 10,000)

  Pilih mode upload:
  [1] Save Latest     — upload checkpoint step terbaru
  [2] Best Checkpoint — upload checkpoint val_bpb terbaik
  [3] Lihat semua checkpoint
  [q] Keluar

  Pilihan (1/2/3/q):
```

---

### Mode CLI (non-interaktif)

#### Upload checkpoint terbaru (step tertinggi)

```bash
python scripts/upload_checkpoint_to_hf.py --latest
```

#### Upload checkpoint terbaik (val_bpb terendah)

```bash
python scripts/upload_checkpoint_to_hf.py --best
```

#### Upload step tertentu

```bash
python scripts/upload_checkpoint_to_hf.py --step 8000
```

#### Lihat semua checkpoint yang tersedia

```bash
python scripts/upload_checkpoint_to_hf.py --list
```

Output contoh:
```
Step       val_bpb      Status
-----------------------------------
2000       1.234567
4000       1.198432
6000       1.187654     ← BEST
8000       1.201234
10000      1.195678
```

---

## Opsi Tambahan

| Flag | Default | Deskripsi |
|------|---------|-----------|
| `--depth` | `d24` | Tag depth model (folder di dalam repo HF) |
| `--repo` | `Dummy9898/mesosfer-checkpoints` | HuggingFace repo ID tujuan |
| `--model-only` | `false` | Skip optimizer state, hanya upload model + meta |
| `--base-dir` | `~/.cache/mesosfer` | Override lokasi checkpoint dir |

### Contoh dengan opsi custom

```bash
# Upload best checkpoint ke repo lain, skip optimizer
python scripts/upload_checkpoint_to_hf.py --best \
    --repo username/my-model \
    --depth d12 \
    --model-only

# Upload dari direktori custom
python scripts/upload_checkpoint_to_hf.py --latest \
    --base-dir /mnt/storage/mesosfer
```

---

## Struktur Repo HuggingFace

File diupload ke dalam subfolder berdasarkan `--depth`:

```
User/mesosfer-checkpoints/
└── d24/
    ├── model_008000.pt
    ├── meta_008000.json
    └── optim_008000_rank0.pt
```

---

## Troubleshooting

**`ERROR: Checkpoint dir tidak ditemukan`**
- Pastikan training sudah berjalan dan menyimpan checkpoint
- Cek path: `~/.cache/mesosfer/base_checkpoints/d24/`
- Gunakan `--base-dir` jika checkpoint disimpan di lokasi lain

**`ERROR: Tidak bisa login ke HuggingFace`**
- Jalankan `hf auth login` dan masukkan token yang valid
- Pastikan token punya permission `write`

**`SKIP: optim_XXXXXX_rank0.pt tidak ditemukan`**
- File optimizer tidak ada di checkpoint dir
- Gunakan `--model-only` untuk skip file ini
- Atau cek apakah nama file optimizer berbeda (multi-rank training)
