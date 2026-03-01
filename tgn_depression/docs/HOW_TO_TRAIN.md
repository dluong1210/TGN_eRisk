# Hướng dẫn chạy train

## Cách 1: Pipeline TGN (pretrain link prediction → supervised)

Đúng với TGN gốc: pretrain encoder bằng link prediction, sau đó freeze encoder và chỉ train decoder.

### Bước 1 – Pretrain (link prediction)

Chỉ dùng **cạnh có ít nhất 1 node là target user** (user có label trong dataset).

```bash
cd tgn_depression   # hoặc từ repo root: python -m tgn_depression.train_pretrain_link_prediction

python -m tgn_depression.train_pretrain_link_prediction \
  --data_dir /path/to/parquet_folders \
  --epochs 10 \
  --batch_size 200 \
  --lr 1e-4 \
  --save_dir ./saved_models_pretrain \
  --gpu 0
```

- Mặc định: `--only_target_user_edges` (chỉ cạnh có ≥1 target user). Tắt bằng `--no_only_target_user_edges` nếu muốn dùng mọi cạnh.
- Kết quả: `./saved_models_pretrain/encoder_link_pred.pth`

### Bước 2 – Supervised (freeze encoder, train decoder)

```bash
python -m tgn_depression.train_supervised_tgn_style \
  --data_dir /path/to/parquet_folders \
  --encoder_checkpoint ./saved_models_pretrain/encoder_link_pred.pth \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --save_dir ./saved_models_supervised \
  --gpu 0
```

- Decoder tốt nhất: `./saved_models_supervised/decoder_best.pth`
- Kết quả: `./saved_models_supervised/results_supervised.json`

---

## Cách 2: End-to-end (encoder + decoder cùng train)

Một script, không pretrain:

```bash
# 1 GPU
python -m tgn_depression.train \
  --data_dir /path/to/parquet_folders \
  --epochs 50 \
  --lr 1e-4 \
  --accumulation_steps 4 \
  --save_dir ./saved_models \
  --gpu 0

# Nhiều GPU (DDP)
torchrun --nproc_per_node=6 -m tgn_depression.train \
  --data_dir /path/to/parquet_folders \
  --epochs 50 \
  --accumulation_steps 4 \
  --num_workers 2 \
  --save_dir ./saved_models
```

---

## Data

- `--data_dir`: thư mục chứa `neg/` và `pos/`, mỗi folder có các file `.parquet` (tên file = target user id).
- Dummy data: `--use_dummy_data` (bỏ qua `--data_dir`).
