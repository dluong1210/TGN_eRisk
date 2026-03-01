# Phân tích luồng Model và Training – TGN eRisk

## 1. Tổng quan kiến trúc

```
UserData (window) → [Conv1, Conv2, ..., ConvK]
                         ↓
    Mỗi Conv: events (src, dst, ts, post_id) → Message → Memory Update (GRU) → Embedding
                         ↓
    Cuối mỗi Conv: free_nodes_except(target_user)  [cắt graph giữa các conv]
                         ↓
    Sau Conv cuối: embedding = _get_target_embedding(target_user)
                         ↓
    logits = classifier(embedding)
                         ↓
    loss = CrossEntropy(logits, label)
```

---

## 2. Các thành phần chính của Model

### 2.1 Memory (`modules/memory.py`)

- **Buffer**: `memory_buffer` [n_nodes, memory_dim] – tensor đầy đủ, gradient có thể chảy qua.
- **`get_memory(node_idxs)`**: Trả về slice `memory_buffer[idx]` – không copy, giữ computation graph.
- **`set_memory(node_idxs, values)`**: Cập nhật có đạo hàm:
  ```python
  old = memory_buffer.detach().clone()
  scatter[idx] = values   # values từ GRU
  memory_buffer = old * (1 - mask) + scatter
  ```
  Truncated BPTT: `old` bị detach, gradient chỉ chảy qua `values` (output của GRU).
- **`get_full_memory_tensor()`**: Trả về `memory_buffer` trực tiếp – dùng cho embedding module.
- **`free_nodes_except(keep_node)`**: Giữ memory của `keep_node`, zero + detach các node khác. **Cắt graph giữa các conversation** – gradient không chảy qua biên conversation.

### 2.2 Message Function (`modules/message_function.py`)

- **Input**: `[src_memory || dst_memory || edge_feats || time_enc]`
- **Identity**: Trả về raw message không đổi.
- **MLP**: Biến đổi qua linear layers.
- Có tham số (nếu dùng MLP) → gradient chảy qua.

### 2.3 Message Aggregator (`modules/message_aggregator.py`)

- **Last**: Lấy message mới nhất cho mỗi node.
- **Mean**: Trung bình các message.
- Không có tham số – chỉ stack/mean, gradient vẫn chảy qua.

### 2.4 Memory Updater – GRU (`modules/memory_updater.py`)

- **GRUCell**: `updated_memory = GRU(message, current_memory)`
- `current_memory = memory.get_memory(nodes)` → slice từ buffer.
- `set_memory(nodes, updated_memory)` → ghi lại buffer.
- Gradient: `loss → ... → updated_memory → GRU(message, current_memory)` → cập nhật trọng số GRU.

### 2.5 Time Encoder (`modules/embedding_module.py`)

- **TimeEncode**: `cos(w * t)` với `w` learnable.
- Dùng trong `_get_raw_messages` (time delta) và embedding module.
- Có tham số → gradient chảy qua.

### 2.6 Embedding Module (`modules/embedding_module.py`)

- **Identity**: `z = memory[target_user]` – không thêm tham số.
- **GraphAttention**: Multi-head attention trên temporal neighbors.
- **GraphSum**: Tổng có trọng số qua linear layers.
- Cả hai đều có tham số → gradient chảy qua.

### 2.7 Classification Head (`utils/utils.py`)

- MLP: `input_dim → 128 → 64 → num_classes`
- Có tham số → gradient chảy qua.

---

## 3. Luồng Forward chi tiết

### 3.1 Một window (1 sample)

1. **reset_state()**: `memory_buffer = zeros`, xóa messages/last_update.
2. **Với mỗi conversation**:
   - Lấy ego subgraph: `(sources, dests, post_ids, timestamps)`.
   - **Với mỗi event** (src, dst, ts):
     - `_get_raw_messages`: `[src_mem, dst_mem, edge_feats, time_enc]` → lưu messages.
     - `_update_memory`: aggregate → message_function → GRU → `set_memory`.
   - Lấy embedding: `_get_target_embedding(target_user, conv_context)`.
   - `free_nodes_except(target_user)`: detach buffer, giữ target_user.
3. **Sau conv cuối**: `result[-1]` = embedding của target_user.
4. **Classifier**: `logits = classifier(result[-1].unsqueeze(0))`.

### 3.2 Gradient path (chỉ trong conversation cuối)

```
loss
  → classifier (Linear layers)
  → embedding = result[-1]
  → _get_target_embedding(target_user, conv_context)
       ├─ identity: memory.get_memory([target_user]) = memory_buffer[target_user]
       └─ graph_attention: embedding_module(full_memory) → dùng memory_buffer
  → memory_buffer[target_user] (và neighbors nếu graph_attention)
  → set_memory(..., updated_memory)  [scatter part]
  → updated_memory = GRU(message, current_memory)
  → GRU params, message
  → message = message_function(raw_message)
  → raw_message = [src_mem, dst_mem, edge_feats, time_enc]
  → time_encoder, edge_features (buffer), memory (cho src/dst)
```

**Truncated BPTT**:

- `free_nodes_except` detach sau mỗi conv → gradient không chảy qua các conv trước.
- Trong `set_memory`, `old` bị detach → gradient không chảy qua memory cũ, chỉ qua GRU output.

---

## 4. Luồng Training (`train.py`)

### 4.1 Một epoch

```python
for window_user_data, label in dataloader:
    raw_model.reset_state()
    logits = model.forward(window_user_data, return_logits=True)
    loss = criterion(logits, label_t)
    loss.backward()
    optimizer.step()
```

- Mỗi sample = 1 window (đã flatten từ `build_flat_window_samples`).
- Mỗi window: reset memory → 1 forward → 1 backward → 1 step.
- Không concat nhiều window → không backward qua nhiều window.

### 4.2 Data flow

- **Positive user**: Sliding windows → nhiều samples (label=1).
- **Negative user**: 1 window = toàn bộ conversations → 1 sample (label=0).
- `FlatWindowDataset` + `DistributedSampler` → mỗi rank nhận ~cùng số samples.

---

## 5. Gradient có chạy vào các thành phần không?

| Thành phần               | Có tham số | Gradient flow | Ghi chú                                            |
| ------------------------ | ---------- | ------------- | -------------------------------------------------- |
| **Memory**               | Không      | Qua buffer    | Buffer được cập nhật bởi GRU, gradient qua scatter |
| **Message Function**     | MLP: có    | Có            | Identity: không tham số                            |
| **Message Aggregator**   | Không      | Có            | Chỉ stack/mean                                     |
| **Memory Updater (GRU)** | Có         | Có            | `updated_memory` nối với loss                      |
| **Time Encoder**         | Có         | Có            | Dùng trong messages và embedding                   |
| **Embedding Module**     | Có         | Có            | GraphAttention/GraphSum có linear/attention        |
| **Classifier**           | Có         | Có            | Trực tiếp: loss → logits → classifier              |

### 5.1 Kiểm tra gradient

Chạy với `--check_gradients` để in grad norm:

```bash
python tgn_depression/train.py ... --check_gradients
```

Kỳ vọng: `grad_norm/memory_updater`, `grad_norm/embedding_module`, `grad_norm/classifier` > 0.

---

## 6. Điểm cần lưu ý

### 6.1 Truncated BPTT

- Gradient chỉ chảy trong **conversation cuối** của mỗi window.
- Các conv trước chỉ cung cấp memory cho conv sau qua `free_nodes_except(keep_val)` (giá trị, không graph).
- Đây là thiết kế có chủ đích để tránh BPTT dài và OOM.

### 6.2 Window với 1 conversation

- Negative user: 1 window = 1 conv.
- Positive user (window nhỏ): có thể 1 window = 1 conv.
- Luồng vẫn đúng: embedding từ conv đó, gradient chảy qua toàn bộ events trong conv.

### 6.3 `free_nodes_except` và gradient

- Gọi **sau** khi lấy embedding → embedding vẫn nối với memory của conv hiện tại.
- **Đã sửa (để model học được):** Không còn detach buffer; dùng **mask nhân** (zero các node khác, giữ `keep_node`). Gradient chảy qua toàn bộ chuỗi conversations thay vì bị cắt sau mỗi conv.

---

## 7. Kết luận

- Luồng model và training **đúng** với thiết kế TGN + truncated BPTT.
- Gradient chảy tới: **GRU**, **Time Encoder**, **Embedding Module**, **Classifier** (và Message Function nếu dùng MLP).
- Train loss giảm phù hợp với gradient flow đã mô tả.
- Để xác nhận: chạy `--check_gradients` và kiểm tra grad norm > 0 cho các module có tham số.

---

## 8. Tối ưu performance (train chậm)

- **Gradient accumulation**: `--accumulation_steps 4` (mặc định) — gộp gradient 4 sample rồi mới `optimizer.step()`, giảm số lần step/sync.
- **Cache / GC**: Không gọi `torch.cuda.empty_cache()` và `gc.collect()` mỗi step; chỉ gọi định kỳ (mỗi 200 / 500 step) để giảm overhead.
- **DataLoader**: `--num_workers 2` (mặc định) — prefetch data; `persistent_workers=True` để tránh tạo process mỗi epoch.
- **Forward**: Trong vòng lặp event dùng slice `sources[i:i+1]` thay vì `np.array([src])` để giảm allocation.
- **Lưu ý**: Không gộp toàn bộ events trong một conv thành một bước cập nhật memory (message i+1 phụ thuộc memory sau event i), nên vòng lặp theo từng event vẫn cần thiết cho đúng với TGN.
