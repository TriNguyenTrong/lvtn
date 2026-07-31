# Kế hoạch kỹ thuật — thay SRT ground-truth bằng quỹ đạo bút thật ở nhánh online

> Soạn 2026-07-28 theo yêu cầu của thầy hướng dẫn: bỏ đầu vào oracle để kết quả so sánh được với các phương pháp đã công bố.
> **Trạng thái: BẢN THẢO CHỜ DUYỆT — chưa đụng vào code.**

---

## 1. Vấn đề đang có

Nhánh online hiện đọc `crohme_all.txt`, mỗi dòng là một chuỗi SRT tuyến tính hóa lấy từ phần **annotation** của file InkML:

```
crohme_all/UN19_1033_em_466.inkml	2 Right \pi Right ( Right \sin Right \theta Sub 1 NoRel + ...
```

Chuỗi này chứa sẵn nhãn ký hiệu (`\pi`, `\sin`, `2`) và quan hệ không gian (`Right`, `Sub`, `Above`) — tức là gần trọn cây biểu thức mà mô hình phải sinh ra. Bộ mã hóa chuỗi nhúng chúng qua `nn.Embedding` trên vocab 108 token, độ dài trung bình 19 token/mẫu.

Hệ quả: mọi con số hiện tại (ExpRate 77,06 / 72,62 / 73,29; micro 74,17) là **cận trên trong điều kiện oracle**, đúng như Mục 4.4 đã ghi nhận, và không đặt cạnh TAP, WAP, BTTR hay bất kỳ hệ nào khác được.

Cần thay đầu vào nhánh online bằng thứ thiết bị thật sự thu được: **dãy toạ độ bút theo thời gian**.

---

## 2. Dữ liệu cần bổ sung

Repo hiện chỉ có `data.zip` (ảnh `.bmp` đã render) và `crohme_all.txt`. **Không có file InkML gốc nào** — đây là việc phải làm trước tiên.

Nguồn (đã xác minh 2026-07-28): gói **ICDAR2019-CROHME-TDF** trên kho TC-11 (IAPR), gộp dữ liệu CROHME 2011–2019, ground-truth ở định dạng InkML kèm SLG và symbol layout tree.

- File: `TC11_package_CROHME2019.zip`, **364 MB**, giấy phép **CC BY-NC-SA 3.0** (dùng cho nghiên cứu, phi thương mại — hợp với luận văn).
- Link tải trực tiếp: `http://tc11.cvc.uab.es/index.php?com=upload&action=file_down&section=dataset&section_id=270&file=237`
- Trang mô tả: https://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1
- Bản dự phòng: gói CROHME 2023 (https://crohme2023.ltu-ai.dev/data-tools/) — bao trọn dữ liệu CROHME cũ, thêm ảnh render tương ứng cho từng inkml; và bản mirror trên Kaggle `ntcuong2103/crohme2019`.

Số liệu khớp chính xác với dữ liệu đang dùng, nên gần như chắc chắn đây đúng là nguồn gốc của `data.zip`:

| Tập | Ảnh trong data.zip | Có SRT | Số công bố của CROHME |
|---|---|---|---|
| train | 8.835 | 8.834 | 8.836 |
| 2014 | 986 | 986 | 986 |
| 2016 | 1.147 | 1.147 | 1.147 |
| 2019 | 1.199 | 1.198 | 1.199 |

Tên file trong `crohme_all.txt` giữ nguyên quy ước gốc của CROHME (`formulaire017-equation012.inkml`, `MfrDB####.inkml`, `RIT_2014_###.inkml`, `UN19_1033_em_466.inkml`…), nên map InkML ↔ ảnh chỉ cần so basename, không cần bảng đối chiếu thủ công.

**Việc cần làm:** tải gói TC-11, giải nén, chạy script kiểm tra độ phủ (mục 7, bước 0). Nếu tỉ lệ khớp < 99% thì dừng lại báo cáo trước khi làm tiếp.

---

## 3. Biểu diễn đầu vào online

Theo TAP [33] — công trình chuẩn cho nhánh online HMER và đã có trong kho `references/` — mỗi điểm quỹ đạo được mô tả bằng vector 8 chiều:

```
[ x_i , y_i , Δx_i , Δy_i , Δ'x_i , Δ'y_i , δ(s_i = s_{i+1}) , δ(s_i ≠ s_{i+1}) ]
```

với `Δx_i = x_{i+1} − x_i`, `Δ'x_i = x_{i+2} − x_i` (tương tự cho y), và hai bit cuối là trạng thái bút: `[1,0]` = pen-down (điểm kế tiếp cùng nét), `[0,1]` = pen-up (kết thúc nét). Dùng đúng công thức này cho phép trích dẫn [33] trong luận văn và giúp người đọc đối chiếu trực tiếp.

Tiền xử lý trước khi trích đặc trưng, theo thứ tự:

1. **Chỉ đọc các phần tử `<trace>`** trong InkML, theo đúng thứ tự xuất hiện (thứ tự thời gian viết). **Tuyệt đối không đọc `<traceGroup>`, `<annotation>`, `<annotationXML>`** — đó là phần nhãn, đọc vào là tái lập oracle dưới hình thức khác. Đây là ranh giới quan trọng nhất của cả kế hoạch, nên viết thành một hàm parse riêng và ghi chú rõ trong docstring.
2. **Loại điểm trùng lặp** liên tiếp (thiết bị lấy mẫu dày, nhiều điểm trùng toạ độ).
3. **Chuẩn hóa**: dịch về gốc, chia theo chiều cao của toàn biểu thức để `y ∈ [0,1]`, giữ nguyên tỉ lệ khung (x có thể > 1). Chuẩn hóa theo chiều cao chứ không theo từng nét, để giữ thông tin vị trí tương đối giữa các ký hiệu — đây chính là thứ nhánh online đóng góp cho cấu trúc.
4. **Resample theo khoảng cách đều** (khoảng cách cố định giữa hai điểm liên tiếp), khử khác biệt tốc độ viết giữa các người viết và giữa các thiết bị.
5. Trích vector 8 chiều, lưu `float16`.

Sản phẩm: một file `online.npz`/thư mục `.npy` song song với `data.zip`, cùng một `index.txt` ghi độ dài từng mẫu để tính batch.

**Cần đo trước khi chốt tham số:** phân bố độ dài chuỗi sau resample (trung bình / p95 / max). Con số này quyết định bước 4 của mục kế tiếp.

---

## 4. Thay đổi code — cụ thể từng file

Nguyên tắc: **không đụng vào bộ giải mã.** Toàn bộ đóng góp kiến trúc của luận văn (dual cross-attention chia sẻ trọng số, huấn luyện hai chiều, cross rescoring) giữ nguyên không sửa một dòng. Chỉ thay đường ống dữ liệu và tầng nhúng đầu vào của bộ mã hóa chuỗi.

### 4.1 `bttr/model/encoder_seq.py`

| Hiện tại | Đổi thành |
|---|---|
| `self.word_embed = nn.Sequential(nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model))` | `self.point_proj = nn.Sequential(nn.Linear(8, d_model), nn.LayerNorm(d_model))` |
| tham số `vocab_size` | bỏ; thêm `in_dim: int = 8` |
| — | thêm khối giảm chiều dài trước Transformer |

Khối giảm chiều dài là bắt buộc chứ không phải tùy chọn: chuỗi SRT hiện dài trung bình 19 token, còn chuỗi điểm sau resample thường vài trăm đến hơn một nghìn. Attention bậc hai theo độ dài nên nếu đưa thẳng vào sẽ vừa tràn bộ nhớ vừa chậm gấp hàng chục lần.

```python
self.downsample = nn.Sequential(
    nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2), nn.ReLU(),
    nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2), nn.ReLU(),
)   # rút ngắn 4 lần
```

Trong `forward`: `point_proj` → `rearrange b l d -> b d l` → `downsample` → `b d l -> b l d` → `pos_enc` → Transformer như cũ. **Mask phải rút ngắn đồng bộ** với hệ số 4 (dùng `mask[:, ::4]` sau khi đã cắt đúng độ dài đầu ra của conv, hoặc `F.max_pool1d` trên mask ép kiểu float rồi ép lại bool). Sai chỗ này là lỗi im lặng — mô hình vẫn chạy, vẫn hội tụ, nhưng attention nhìn vào vùng padding và kết quả tụt mà không rõ lý do.

### 4.2 `bttr/datamodule/datamodule.py`

- `Data`: phần tử thứ ba đổi từ `List[str]` (token SRT) sang `np.ndarray [n, 8]`.
- `build_dataset`: bỏ `vocab_enc.words2indices(seq_dict[fname].split())`, thay bằng nạp mảng quỹ đạo đã tiền xử lý.
- `data_iterator`: hiện gom lô chỉ theo diện tích ảnh (`MAX_SIZE = 32e4`). Phải thêm ràng buộc theo độ dài quỹ đạo, vì một mẫu ảnh nhỏ vẫn có thể có quỹ đạo rất dài. Thêm `maxlen_traj` (đặt theo p99 đo ở mục 3) và điều kiện tổng `max_traj_len × (i+1)` không vượt ngưỡng.
- `Batch`: `seq_indices: List[List[int]]` → `traj: FloatTensor [b, L, 8]` và `traj_mask: BoolTensor [b, L]`; cập nhật `to(device)` để chuyển cả hai lên GPU (hiện `seq_indices` là list Python nên `to()` bỏ qua nó).
- `collate_fn`: thêm phần pad quỹ đạo về `max_L` trong lô và dựng mask. Việc pad hiện đang do `to_src` làm ở tầng `lit_bttr`; chuyển hẳn xuống đây cho gọn.
- Bỏ tham số `vocab_enc` khỏi `CROHMEDatamodule` (hoặc giữ chữ ký nhưng không dùng — tôi nghiêng về bỏ hẳn để không ai vô tình nạp lại SRT).

### 4.3 `bttr/lit_bttr.py`

- `training_step` / `validation_step` / `test_step`: bỏ `seq, seq_mask = to_src(batch.seq_indices, self.device)`, dùng thẳng `batch.traj`, `batch.traj_mask`.
- Dòng 113 trong hàm `beam_search` công khai: `seq_mask = torch.zeros_like(sequence_feature, dtype=torch.bool)` — chỗ này vốn đã sai chiều (mask cùng shape với feature thay vì `[b, L]`), với đầu vào float 3 chiều thì càng sai. Sửa thành `torch.zeros(sequence_feature.shape[:2], dtype=torch.bool, device=...)`.
- Bỏ `vocab_enc` khỏi `hparams`. Lưu ý: đổi `hparams` làm **checkpoint cũ không load lại được** — đây là lý do phải huấn luyện lại toàn bộ chứ không fine-tune từ `best.ckpt`.

### 4.4 `bttr/model/bttr.py`

Chỉ đổi chữ ký: bỏ `vocab_size_enc`, đổi tên tham số `sequence_feature` cho đúng nghĩa (`traj`). Phần `fusion == "concat"` nối theo trục token vẫn hoạt động nguyên vẹn vì sau downsample hai memory đều có dạng `[b, ·, d]`.

### 4.5 `custom_train.py` / `config.yaml`

`batch_size=32` nhiều khả năng phải giảm xuống 16 hoặc 8 do chuỗi dài hơn. Nếu đổi thì **phải cập nhật Bảng 2 của luận văn** và ghi lại lý do.

### 4.6 `bttr/utils.py`

`to_src` chỉ phục vụ nhánh SRT; sau khi chuyển thì không còn ai gọi. Giữ nguyên hàm (vô hại) nhưng thêm ghi chú "chỉ dùng cho biến thể oracle-SRT" để phiên sau không nhầm.

---

## 5. Quy trình kiểm chứng trước khi chạy dài

Bốn cửa ải, qua cửa trước mới sang cửa sau. Bỏ qua bước nào là mất vài ngày huấn luyện vô ích:

1. **Dựng hình lại.** Lấy 10 mẫu ngẫu nhiên, vẽ quỹ đạo từ tensor đã tiền xử lý ra ảnh và đặt cạnh file `.bmp` tương ứng trong `data.zip`. Hai hình phải là cùng một biểu thức. Bước này bắt được lỗi map tên file, lỗi lật trục y (InkML gốc toạ độ y hướng xuống), lỗi chuẩn hóa.
2. **`fast_dev_run=True`** — chỉ để chắc shape và mask không nổ.
3. **`overfit_batches=1`, ~200 bước.** Loss phải xuống gần 0. Không xuống được nghĩa là đường dẫn gradient qua nhánh mới bị đứt.
4. **Chạy cấu hình `fusion="online"` trước tiên** (chỉ nhánh quỹ đạo, không có ảnh). Đây là phép thử quyết định: TAP đơn mô hình đạt **50,41% ExpRate trên CROHME 2014** [33]. Nếu ta ra khoảng 40–50% thì tiền xử lý đúng và có thể chạy tiếp. Nếu ra dưới 15% thì gần như chắc chắn lỗi tiền xử lý hoặc mask, đừng chạy các cấu hình còn lại.

---

## 6. Kế hoạch thí nghiệm

Giữ nguyên cấu trúc ablation của Chương 4 để không phải viết lại bố cục, chỉ thay số. Sáu lần chạy, cùng `seed_everything(7)`, thư mục thêm hậu tố `_traj` để không đè checkpoint cũ:

| Cấu hình | Mục đích |
|---|---|
| `offline` | nhánh ảnh đơn lẻ — số này không đổi so với hiện tại, dùng làm mốc kiểm tra tính nhất quán |
| `online` | nhánh quỹ đạo đơn lẻ — đối chiếu TAP [33] |
| `dual_shared` | cấu hình chính của luận văn |
| `concat` | trục thiết kế hợp nhất |
| `cascaded` | trục thiết kế hợp nhất |
| `dual_shared_uni` | trục hướng huấn luyện (một chiều) |

Mỗi lần dưới hai giờ trên RTX 4080 theo ghi nhận ở Mục 3.10, cộng thêm phần chuỗi dài hơn thì ước lượng 2–4 giờ. Tổng khối lượng máy khoảng một đến hai ngày, chạy nối tiếp được qua đêm.

**Nên cân nhắc chạy thêm 2 seed** cho `dual_shared` và `concat`: chênh lệch giữa hai thiết kế hợp nhất hiện chỉ 1,7 điểm, sát biên nhiễu, và hội đồng đã từng nêu điểm này.

---

## 7. Các bước theo thứ tự

| # | Việc | Kết quả kiểm chứng được |
|---|---|---|
| 0 | ~~Tải gói InkML từ TC-11, kiểm độ phủ~~ **✅ XONG 2026-07-28** | `inkml.zip` (12.166 file), độ phủ 99,99%; biên bản: `ghi_chu/BOC_TACH_INKML.md` |
| 1 | ~~Viết `tools/prep_online.py`~~ **✅ XONG 2026-07-28** | `online/{train,2014,2016,2019}.npz`; 6/6 hình dựng lại khớp `.bmp` (`online/preview_train_6.png`); độ dài 4 tập đồng đều 312–330 |
| 2 | ~~Sửa `encoder_seq.py`~~ **✅ XONG 2026-07-28** | nhánh git `online-traj`; test ở `test_component.py` (chờ chạy trong env `bttr`) |
| 3 | ~~Sửa `datamodule.py`, `bttr.py`, `lit_bttr.py`~~ **✅ XONG 2026-07-28** | thêm `test_all.py`, `predict_test.py`, `config.yaml`; chi tiết ở mục 11 |
| 3b | ~~Chạy `test_component.py`~~ **✅ PASS TOÀN BỘ 2026-07-28** | 9/9 nhóm kiểm PASS; bất biến padding 9,54e-07 |
| 4 | Overfit 1 batch | loss → ~0 |
| 5 | Chạy `fusion="online"` | ExpRate 2014 trong vùng 40–50% |
| 6 | Chạy 5 cấu hình còn lại | 24 file kết quả mới trong `results/` |
| 7 | Cập nhật luận văn (mục 8) | 2 docx + 2 PDF |

Ước lượng: bước 0–1 khoảng một đến hai ngày, bước 2–4 khoảng một ngày, bước 5–6 chủ yếu là thời gian máy.

---

## 8. Ảnh hưởng tới luận văn

> **CẬP NHẬT 2026-07-29 — thầy chốt GIỮ CẢ HAI bộ số.** Mục 8 bên dưới viết theo giả định thay thế hoàn toàn; phần còn hiệu lực là danh sách các mục phải sửa. Cấu trúc trình bày mới nằm ở **mục 12**.

Đây là phần cần nói thẳng với thầy: **con số headline sẽ tụt mạnh.** Bỏ oracle đi thì mất phần thông tin nhãn được cho không, kỳ vọng rơi về vùng 55–62% ExpRate thay vì 74,17% — nhưng đó là con số đặt cạnh BTTR offline (53,96%), TAP (50,41%) và các hệ đa phương thức khác được, tức là đúng thứ thầy đang yêu cầu.

Luận điểm của luận văn vẫn đứng vững nếu cấu hình hợp nhất vượt **cả hai** nhánh đơn lẻ dưới cùng điều kiện đầu vào — mà đó là phép so nội bộ, không phụ thuộc oracle.

Các mục phải sửa:

- **3.2–3.3**: mô tả đầu vào nhánh online (chuỗi điểm 8 chiều thay chuỗi token SRT), thêm phần tiền xử lý và khối giảm chiều dài.
- **Hình 3, Hình 4** (`thesis/figures/`): nhãn hiện là "Online symbol-relation sequence · linearized SRT · token ids [b, l1]" → đổi sang "Online trajectory · point features [b, l₁, 8]". Sửa SVG cả EN lẫn VI, render, duyệt, rồi mới nhúng.
- **Bảng 2**: bỏ dòng vocab bộ mã hóa chuỗi (111), thêm khoảng cách resample, hệ số giảm chiều dài, độ dài quỹ đạo tối đa; cập nhật batch size nếu đổi.
- **Bảng 3/4/5**: toàn bộ số mới.
- **Mục 4.4**: caveat oracle-SRT không còn cần thiết ở dạng hiện tại — viết lại thành một đoạn so sánh với các hệ đã công bố (thứ trước đây không làm được), và giữ số oracle cũ như một dòng "cận trên" trong bảng ablation kèm giải thích.
- **Abstract, Mục 1, Mục 5.1**: cập nhật con số và câu trả lời cho câu hỏi nghiên cứu.
- **Bảng 1**: giờ mới thực sự có nghĩa — cột kết quả của "Ours" đặt cạnh các dòng khác được.
- **`bao_ve/`**: slide và kịch bản phải làm lại số.

Khối lượng sửa luận văn không nhỏ nhưng cấu trúc giữ nguyên; phần lớn là thay số và viết lại 4.4.

---

## 9. Phương án dự phòng

Nếu không lấy được InkML: huấn luyện một bộ nhận dạng online riêng (phân đoạn nét → phân loại ký hiệu → quan hệ) để sinh SRT dự đoán, rồi giữ nguyên đường ống hiện tại. Về mặt công sức thì nặng hơn phương án chính, lỗi lại cộng dồn qua hai tầng, và vẫn không phải cách các công trình khác định nghĩa "online". Chỉ dùng nếu bước 0 thất bại.

Cách chỉ thêm nhiễu mô phỏng vào SRT ground-truth **không** đáp ứng yêu cầu của thầy — nó đo được độ nhạy của mô hình với chất lượng đầu vào, hữu ích như một ablation phụ, nhưng không tạo ra con số so sánh được với ai.

---

## 10. Câu hỏi còn mở, cần chốt trước khi bắt tay

1. Thầy có yêu cầu giữ lại kết quả oracle như một phần của luận văn không, hay muốn bỏ hẳn? (Tôi nghiêng về giữ, trình bày như cận trên — nó cho thấy nhánh online còn dư địa bao nhiêu.)
2. Có ràng buộc thời gian bảo vệ không? Toàn bộ kế hoạch này ước tính 4–6 ngày làm việc cộng thời gian sửa luận văn.
3. Khoảng cách resample và hệ số giảm chiều dài (4×) là hai tham số tôi đề xuất theo TAP; sau khi đo phân bố độ dài thực tế có thể phải điều chỉnh — sẽ báo lại trước khi chốt.

---

## 11. Nhật ký sửa code — bước 2 và 3 (2026-07-28)

Toàn bộ nằm trên nhánh git **`online-traj`**, chưa commit. Bộ giải mã không bị đụng một dòng nào: `decoder.py`, `encoder_img.py`, `pos_enc.py` nguyên vẹn.

| File | Thay đổi |
|---|---|
| `bttr/model/encoder_seq.py` | `nn.Embedding(vocab, d)` → `nn.Linear(8, d)`; thêm khối `reduce` gồm 2 lớp `Conv1d(k=5, stride=2)` giảm 4×; thêm `_shrink_mask` tính lại mask theo độ dài hợp lệ; `forward` nay trả về **cặp** `(memory, mask)` |
| `bttr/model/bttr.py` | bỏ `vocab_size_enc`, thêm `traj_dim=8` / `traj_downsample=4`; đổi tên `sequence_feature*` → `traj`/`traj_mask`; nhận mask mới từ encoder chuỗi |
| `bttr/datamodule/datamodule.py` | thêm `load_online()` đọc `.npz`; `Batch.seq_indices` → `traj` + `traj_mask` (cả hai lên GPU trong `to()`); `collate_fn` pad quỹ đạo; `data_iterator` thêm hạn mức `MAX_TRAJ_SIZE=25600` và `MAX_TRAJ_LEN=3000`; datamodule đổi `seq_annotation`/`vocab_enc` thành `online_dir` |
| `bttr/lit_bttr.py` | bỏ `to_src` và `vocab_enc`; ba `*_step` dùng thẳng `batch.traj`; sửa lỗi mask sai chiều ở `beam_search` công khai (dòng 113 cũ) |
| `bttr/utils.py` | `to_src` giữ nguyên, chỉ thêm ghi chú là di sản của biến thể oracle-SRT |
| `test_all.py`, `predict_test.py` | lấy quỹ đạo từ `batch`/`online/*.npz` thay vì dựng chuỗi SRT |
| `config.yaml` | thêm `traj_dim`, `traj_downsample`, `online_dir` |
| `test_component.py` | viết lại thành bộ smoke test cho nhánh mới |
| `.gitignore` | bỏ qua `TC11_package_CROHME2019.zip` và `online/*.npz` (sinh lại trong 10 giây) |

**Hai chi tiết dễ sai đã xử lý:**

1. `forward` của `SeqEncoder` trả về cả mask vì tích chập stride làm chuỗi ngắn lại; nếu vẫn dùng mask cũ thì attention sẽ đọc vào vùng đệm. Mask mới tính lại từ độ dài hợp lệ (`ceil(len/4)`) chứ không cắt lát — đã kiểm bằng số học trên toàn dải độ dài 1–3000 cho cả ba hệ số 2/4/8, khớp đúng chiều dài đầu ra của tích chập, 0 sai lệch.
2. Dòng `seq_mask = torch.zeros_like(sequence_feature, dtype=torch.bool)` trong `lit_bttr.beam_search` vốn đã sai chiều từ trước (mask cùng shape với feature). Với đầu vào float ba chiều thì sai hẳn, nên đã sửa thành `torch.zeros(traj.shape[:2])`.

**🐞 Lỗi tìm được khi user chạy test lần 1 (2026-07-28) — ĐÃ SỬA:** `WordPosEnc` mặc định `max_len=500`, trong khi mẫu dài nhất tập train là 2.038 điểm → 510 sau khi giảm 4×. Bảng mã vị trí thiếu chỗ, `x + emb` vỡ chiều. Đây là lỗi của huấn luyện thật chứ không riêng bài test — mẫu dài sẽ làm sập giữa chừng epoch đầu. Đã thêm tham số `max_len` cho `SeqEncoder`, mặc định 3000 khớp `MAX_TRAJ_LEN`, phủ luôn trường hợp `downsample=1`; bảng tốn khoảng 3 MB. Sửa kèm: nới điều kiện "vùng đệm bằng 0" trong `test_datamodule_batch` cho lô một mẫu (không có đệm nên tensor rỗng, `.max()` sẽ lỗi).

**🐞 Lỗi tìm được khi user chạy test lần 2 (2026-07-28) — ĐÃ SỬA:** phép chiếu điểm kết thúc bằng `LayerNorm`, nên một điểm đệm toàn số 0 **không** ở lại bằng 0 mà thành `LayerNorm(bias)` — một vector hằng khác 0. Nhân chập sau đó trộn vector này vào vị trí thật cuối cùng của mỗi mẫu, và tự-chú-ý lan sai lệch đó ra toàn chuỗi: cùng một mẫu sẽ cho kết quả khác nhau tùy nó nằm chung lô với mẫu dài bao nhiêu. Sửa bằng một `masked_fill` ngay sau `point_proj`, đưa vùng đệm về đúng 0 để nhân chập nhìn thấy cùng thứ số 0 mà nó tự đệm. Bài kiểm cũng viết lại cho chặt hơn: so ba lần chạy (không đệm, đệm 600, đệm 901) và đòi trùng khớp trên **toàn bộ** vùng hợp lệ với sai số 1e-5, thay vì bỏ qua hai vị trí biên như trước.

**🐞 Lỗi tìm được khi user chạy test lần 3 (2026-07-28) — ĐÃ SỬA:** che một lần trước nhân chập là chưa đủ. Mỗi nhân chập (k=5) vươn ngược vào dữ liệu thật, nên vị trí đệm **đầu tiên** ở đầu ra của mỗi tầng vẫn khác 0 dù đầu vào đã sạch; tầng sau lại trộn giá trị đó ngược vào vị trí thật cuối cùng. Sửa: đổi `reduce` từ `Sequential` sang `ModuleList` và che lại sau **từng** tầng, dùng mask đã thu nhỏ tương ứng (hệ số 2 mỗi tầng, ceil hai lần = ceil bốn — đã kiểm số học trên toàn dải 1–3000 cho cả ba hệ số, 0 sai lệch). Sau sửa, một mẫu cho kết quả trùng khớp tuyệt đối dù đệm tới độ dài nào.

**Việc còn lại của bước 2–4 (cần máy có GPU/torch):**

```
conda run -n bttr --no-capture-output python test_component.py     # bước 2
```

**✅ ĐÃ CHẠY, PASS TOÀN BỘ 2026-07-28** (user chạy trong env `bttr`): decoder forward · chiều ra + chiều mask của `SeqEncoder` (4 hệ số giảm × 4 độ dài) · **bất biến với padding, sai số 9,54e-07** (so ba lần chạy: không đệm, đệm 600, đệm 901) · `forward` của `BTTR` cả 5 chế độ hợp nhất · gradient về tới `point_proj` và cả 2 tầng tích chập · một lô thật từ datamodule (`(1, 59, 8)`).

Hai quan sát từ log lô thật, cần nhớ khi viết lại Chương 4:

- `sentence 1 length bigger than 200 ignore` — mẫu `505_em_51` của CROHME 2014 có nhãn > 200 token, bị `data_iterator` loại. **Không phải lỗi mới:** đối chiếu `results/` thì mọi số cũ đều tính trên mẫu số 985 (77,06% = 759/985), đường ống mới tái lập đúng tập đánh giá cũ.
- **Mẫu số tập 2019 đổi 1198 → 1199.** Mẫu `UN19_1001_em_0` trước bị loại vì không có dòng SRT, nay có quỹ đạo nên được nhận vào. Chênh lệch ExpRate cỡ 0,06 điểm, nhưng số mẫu ghi trong bảng phải sửa theo.

⚠️ **`git` trong repo đang kẹt `.git/index.lock`** (mount của tôi không xóa được file). Anh xóa tay trên Windows rồi mới thao tác git được:

```
del .git\index.lock
git reset          # bỏ staging mà tôi lỡ tạo bằng `git add -A`
```

---

## 12. Khung Chương 4 khi trình bày song song hai bộ số (soạn 2026-07-29)

Nguyên tắc sắp xếp: đọc từ trên xuống, người đọc gặp số thật trước, hiểu nó so được với ai, rồi mới gặp số oracle như một phép đo dư địa. Không bao giờ ngược lại.

| Mục | Nội dung | Bộ số |
|---|---|---|
| 4.1–4.3 | Dữ liệu, thiết lập, độ đo | chung; **bổ sung** mô tả tiền xử lý quỹ đạo và mẫu số từng tập |
| 4.4 **(mới)** | **Kết quả chính** trên CROHME 2014/2016/2019 | quỹ đạo thật |
| 4.5 **(mới)** | **So với các phương pháp đã công bố** — Bảng 1 | quỹ đạo thật, **chỉ một dòng "Ours"** |
| 4.6 | Ablation thiết kế hợp nhất (offline / online / dual_shared / concat / cascaded) | quỹ đạo thật |
| 4.7 | Ablation hướng huấn luyện (hai chiều vs một chiều) | quỹ đạo thật |
| 4.8 **(mới)** | **Cận trên với đầu vào oracle** — tiêu đề nói thẳng; bảng cũ chuyển nguyên vào đây | oracle SRT |
| 4.9 **(mới)** | **Khoảng cách oracle ↔ quỹ đạo thật** — phần có giá trị khoa học nhất của việc giữ cả hai | so hai bộ |
| 4.10 | Phân tích lỗi | quỹ đạo thật |

Mục 4.9 là chỗ biện minh cho quyết định giữ cả hai, nên phải viết cho ra tấm ra món chứ không phải một đoạn cho có. Ba ý cần có: (a) chênh lệch tuyệt đối theo từng tập kiểm thử; (b) diễn giải chênh lệch đó là dư địa của nhánh online — nếu một bộ nhận dạng ký hiệu online đạt độ chính xác X thì hệ thống ước lượng thu về bao nhiêu phần của khoảng cách; (c) chỉ rõ giới hạn của phép đo: oracle không chỉ cho nhãn đúng mà còn cho **quan hệ không gian** đúng, nên nó là cận trên rộng rãi chứ không phải mục tiêu khả thi.

Bảng 2 (siêu tham số) nên gộp làm một, thêm cột hoặc dòng đánh dấu tham số nào chỉ áp cho biến thể nào — tách hai bảng sẽ khiến người đọc phải lật đi lật lại.

Câu hỏi nghiên cứu ở Mục 1 và câu trả lời ở Mục 5.1 chỉ được căn theo số quỹ đạo thật. Số oracle xuất hiện ở Chương 5 nhiều nhất là một câu, dưới dạng hướng phát triển ("cải thiện nhánh online có thể thu về tới N điểm").

---

**Nguồn tham chiếu**

- [33] J. Zhang, J. Du, L. Dai, "Track, Attend, and Parse (TAP): An end-to-end framework for online handwritten mathematical expression recognition" — công thức đặc trưng 8 chiều (eq. 2), ExpRate 50,41% trên CROHME 2014 (hệ P4). File: `references/33.pdf`.
- [22] Stroke constrained attention network — tham khảo cách xử lý cấp nét. File: `references/22.pdf`.
- [39]/[40]/[41] các báo cáo CROHME 2014/2016/2019 — mô tả định dạng InkML và bộ dữ liệu.
- Gói dữ liệu: ICDAR2019-CROHME-TDF, kho TC-11 IAPR.
