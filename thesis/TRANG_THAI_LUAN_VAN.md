# TRẠNG THÁI LUẬN VĂN — Multi-modal HMER (nguồn sự thật DUY NHẤT)

> File này là nơi lưu chuẩn của toàn bộ trạng thái, quyết định và việc còn mở của luận văn.
> Chuyển từ memory của Claude Code vào repo ngày 2026-07-02 để Cowork/Claude Code cùng đọc được.
> **Mọi phiên làm việc (Cowork hay Claude Code): ĐỌC file này trước khi sửa luận văn, và CẬP NHẬT nó sau khi có quyết định mới.**


Trạng thái luận văn thạc sĩ "Multi-modal HMER" (dual-modality BTTR) tính đến 2026-07-02. Xem kiến trúc ở [[project-dual-modality-bttr]], build docx ở [[setup-build-docx-windows]].

## 🔴 ĐỔI HƯỚNG LỚN 2026-07-28 — BỎ ORACLE SRT, NHÁNH ONLINE DÙNG QUỸ ĐẠO BÚT THẬT

> **ĐỌC MỤC NÀY TRƯỚC MỌI THỨ KHÁC.** Thầy hướng dẫn yêu cầu thay đầu vào ground-truth của bộ mã hóa online bằng dữ liệu thật, để kết quả so sánh được với các phương pháp đã công bố. Toàn bộ số liệu Bảng 3/4/5 hiện hành (77,06 / 72,62 / 73,29, micro 74,17) **sẽ bị thay** sau khi huấn luyện lại. Chưa sửa một chữ nào trong 2 docx.

**Kế hoạch đầy đủ + nhật ký: `KE_HOACH_ONLINE_TRAJECTORY.md` ở gốc dự án. Biên bản dữ liệu: `ghi_chu/BOC_TACH_INKML.md`.**

Tóm tắt tình hình:

- **Vấn đề:** nhánh online đọc `crohme_all.txt` — chuỗi SRT lấy từ annotation InkML, chứa sẵn nhãn ký hiệu và quan hệ không gian. Mọi con số vì thế là cận trên trong điều kiện oracle (đã ghi ở Mục 4.4) và không đặt cạnh TAP/WAP/BTTR được.
- **Hướng giải quyết:** thay đầu vào bằng quỹ đạo bút thô trích từ InkML gốc, đặc trưng 8 chiều theo TAP [33]. Bộ giải mã dual cross-attention, huấn luyện hai chiều, cross rescoring **giữ nguyên không sửa** — đóng góp kiến trúc của luận văn không đổi.
### Đã xong (2026-07-28, phiên Cowork)

① **Dữ liệu.** Tải + bóc `TC11_package_CROHME2019.zip` (kho TC-11, CC BY-NC-SA 3.0) → **`inkml.zip`** ở gốc dự án: 12.166 file, thư mục `train/ 2014/ 2016/ 2019/` khớp cấu trúc `data.zip`. Độ phủ **99,99%** (12.166/12.167); mẫu thiếu duy nhất `MfrDB0104` vốn đã bị loại từ trước vì không có SRT. Train lấy `Train_2014` (8.834) chứ không lấy trọn train CROHME2019 (9.993) để giữ nguyên tập huấn luyện cũ; 2016/2019 lấy bản **noGT** (0 file chứa `<traceGroup>`).

② **Tiền xử lý.** `tools/prep_online.py` → **`online/{train,2014,2016,2019}.npz`** (42 MB, dựng lại trong 10 giây). Đặc trưng 8 chiều theo TAP [33] eq.(2), resample theo khoảng cách đều `--step 0.03`. Kiểm hình **6/6 khớp `.bmp`** (`online/preview_train_6.png`). Độ dài sau xử lý: 312/313/321/330 điểm — trước đó tập 2014 lệch 73% so với train (706 vs 408), resample theo khoảng cách đã khử hết.

③ **Code.** Nhánh git **`online-traj`**, 9 file: `encoder_seq.py`, `bttr.py`, `datamodule.py`, `lit_bttr.py`, `utils.py`, `test_all.py`, `predict_test.py`, `config.yaml`, `test_component.py`. **Bộ giải mã KHÔNG bị đụng** — `decoder.py`, `encoder_img.py`, `pos_enc.py` nguyên vẹn, nên đóng góp kiến trúc của luận văn không đổi.

④ **`test_component.py` PASS TOÀN BỘ** (user chạy 2026-07-28 trong env `bttr`): decoder forward · chiều ra + chiều mask của SeqEncoder (4 hệ số × 4 độ dài) · **bất biến với padding sai số 9,54e-07** · BTTR forward cả 5 chế độ hợp nhất · gradient về tới `point_proj` và 2 tầng conv · lô thật từ datamodule.

**3 lỗi tìm được và đã sửa trong quá trình test** (nguyên văn ở mục 11 của `KE_HOACH_ONLINE_TRAJECTORY.md` — đọc nếu cần sửa tiếp nhánh online):
- `WordPosEnc` mặc định `max_len=500` trong khi mẫu train dài nhất là 510 sau giảm 4× → thêm tham số `max_len`, mặc định 3000 khớp `MAX_TRAJ_LEN`. **Đây là lỗi của huấn luyện thật**, không riêng bài test.
- `point_proj` kết thúc bằng `LayerNorm` nên điểm đệm toàn 0 hóa thành `LayerNorm(bias)` khác 0 → thêm `masked_fill`.
- Che một lần là chưa đủ: nhân chập k=5 vươn ngược vào dữ liệu thật nên vị trí đệm đầu tiên ở đầu ra mỗi tầng vẫn khác 0 → `reduce` đổi sang `ModuleList`, che lại sau **từng** tầng.

### 🟢 QUYẾT ĐỊNH CỦA THẦY 2026-07-29 — GIỮ CẢ HAI BỘ SỐ, KHÔNG BỎ BẢN ORACLE

Thầy hướng dẫn chốt: **không bỏ nội dung cũ, chỉ bổ sung phần quỹ đạo thật.** Luận văn sẽ trình bày song song hai bộ kết quả.

Lý do việc này tốt hơn phương án thay thế: cùng mô hình, cùng seed 7, cùng dữ liệu, cùng siêu tham số — chỉ khác đúng một biến là chất lượng đầu vào nhánh online. Khoảng cách giữa hai bộ số vì thế là một **thí nghiệm có kiểm soát**, trả lời được câu hỏi "nhánh online còn dư địa bao nhiêu nếu nhận dạng ký hiệu online tốt lên" — điều mà bản chỉ-có-quỹ-đạo không trả lời được.

**Ba ràng buộc BẮT BUỘC về cách trình bày** (nếu vi phạm thì coi như chưa đáp ứng yêu cầu của thầy):

1. **Số quỹ đạo thật dẫn dắt, số oracle phụ thuộc.** Abstract, Bảng 1, Mục 5.1 chỉ nêu số quỹ đạo thật. Số oracle nằm trong một mục riêng của Chương 4, tiêu đề nói thẳng là cận trên.
2. **TUYỆT ĐỐI không để số oracle đứng chung bảng với các phương pháp đã công bố.** Bảng 1 chỉ có một dòng "Ours" = quỹ đạo thật. Đây chính là vấn đề thầy yêu cầu sửa.
3. **Tự nêu hạn chế của số oracle trước, dứt khoát**, ngay khi giới thiệu nó — để hội đồng không đọc ra sự luyến tiếc một con số đẹp.

### ⚠️ HAI VIỆC CHỜ USER CHỐT (nêu 2026-07-29, chưa có quyết định)

**(a) Mẫu số khi đánh giá.** Bộ số hiện tại tính trên **985** mẫu CROHME 2014 chứ không phải 986 — mẫu `505_em_51` bị `data_iterator` loại vì nhãn > 200 token (`maxlen=200`). Trước đây không quan trọng vì caveat oracle khiến việc so sánh là vô nghĩa; **giờ thì quan trọng**, vì đặt cạnh BTTR 53,96% vốn tính trên đủ 986 mẫu thì mẫu số lệch là sai lệch giao thức. Chênh lệch: 759/985 = 77,06% so với 759/986 = 76,98%. Ba lựa chọn: ① sửa `test_all.py` tính trên đủ 986/1147/1199, mẫu bị loại tính là sai (**khuyến nghị** — đúng giao thức CROHME); ② giữ 985 và chú thích; ③ bảng chính dùng 986, chú thích chân trang ghi con số trên 985. **Chốt trước khi chạy `test_all.py`, không cần chốt trước khi huấn luyện.**

**(b) Tái lập số oracle.** Code nhánh online đã đổi nên `results/` cũ không sinh lại được bằng code hiện tại. Ba lựa chọn: ① **gắn git tag cho commit trước khi đổi** (khuyến nghị — đơn giản, luận văn dẫn tag); ② thêm công tắc `online_input: traj | srt` để một codebase chạy được cả hai (đẹp cho luận văn nhưng phải giữ đường `nn.Embedding` cũ và test thêm); ③ chỉ giữ `results/` làm bằng chứng.

### 🏃 NHẬT KÝ HUẤN LUYỆN QUỸ ĐẠO THẬT (phiên Claude Code, bắt đầu 2026-07-29)

Nhánh git `online-traj`, `seed_everything(7)`, mọi thư mục checkpoint có hậu tố `_traj`.
Bộ `results/` cũ và `lightning_logs/abl_*` (không hậu tố) **không bị đụng**.

| Bước | Ngày | Kết quả |
|---|---|---|
| 1. `fast_dev_run` | 2026-07-29 | ✅ **PASS**. Chạy trọn train step + val step trên RTX 4080; loss 5,62 / val_loss 6,75. Không lỗi shape hay mask. Nạp dữ liệu đúng: train 8.834 quỹ đạo → 1.139 lô (2 ảnh bị loại vì diện tích > 32e4, đúng như trước), 2014 986 quỹ đạo → 985 mẫu (`505_em_51` nhãn > 200 token). |
| 2. Overfit 1 lô | 2026-07-29 | ✅ **PASS**. Loss 7,8 → **0,000273**. Đường gradient qua nhánh quỹ đạo mới liền mạch. ⚠️ Lưu ý cho phiên sau: `max_epochs=30` như kế hoạch ghi là **không đủ** — 30 epoch × 1 lô = 30 bước optimizer, loss mới xuống 6,05 và EarlyStopping cắt ở epoch 21, dễ đọc nhầm thành "gradient đứt". Adadelta lr=1,0 cần ~200–300 bước; phải chạy `max_epochs=300`, tắt EarlyStopping, `num_workers=0` (tránh 10 s/epoch respawn worker). `custom_train.py` đã hoàn nguyên sau khi chạy. |
| 3. `fusion="online"` | 2026-07-29 | ⛔ **KHÔNG QUA CỔNG: ExpRate 2014 = 0,00% (0/985).** Lần 1 dừng sớm ở epoch 7 (EarlyStopping patience mặc định); lần 2 với `patience=15` chạy đủ 50 epoch, val_loss tốt nhất 2,5392 @ epoch 27. Nguyên nhân đã truy ra: **bộ lập lịch LR cắt learning rate giữa lúc đang học** (mục 🔻 dưới). Kết quả: `results/traj_abl_online_2014_results.txt`; checkpoint `lightning_logs/abl_online_traj/lightning_logs/version_1/checkpoints/epoch=27-step=31892-val_loss=2.5392.ckpt`. **CHƯA chạy 5 cấu hình còn lại.** |

#### 🔻 CỔNG QUYẾT ĐỊNH KHÔNG QUA — nguyên nhân là lịch learning rate, KHÔNG phải tiền xử lý (2026-07-29)

Kế hoạch dự đoán "dưới 15% ⇒ gần như chắc chắn lỗi tiền xử lý hoặc mask". **Dự đoán đó sai** — đã loại trừ cả hai bằng bằng chứng cứng (xem 4 phép kiểm ở khối chẩn đoán bên dưới, cộng 2 phép kiểm suy luận sau). Nguyên nhân nằm ở tối ưu hóa.

**Đã loại trừ ở đường suy luận:**
- `fusion` nạp lại từ checkpoint đúng là `online` ở cả `LitBTTR.hparams` lẫn từng lớp decoder → không phải nhầm cấu hình khi load.
- **Tráo quỹ đạo giữa các mẫu làm đổi dự đoán ở 6/6 mẫu** → quỹ đạo CÓ tới được bộ giải mã trong beam search, không phải lỗi đường suy luận.
- Teacher-forced token accuracy trên 2014 = **43,43%** (1.666/3.836), full-sequence exact 0/300. Mô hình có học tín hiệu thật (ngẫu nhiên ~1%), chỉ là quá yếu để trúng trọn biểu thức.

**Nguyên nhân: `configure_optimizers` trong `lit_bttr.py` dùng `ReduceLROnPlateau(mode="max")` nhưng `monitor="val_loss"`.** Với `mode="max"`, val_loss GIẢM bị đọc là "không cải thiện", nên bộ lập lịch mất hoàn toàn tính thích ứng và trở thành lịch cố định: cứ 10 lần kiểm val (= 20 epoch) cắt LR 10× một lần. Số bước/epoch = 1.139, nên các mốc thực tế:

| | LR 1,0 → 0,1 | LR 0,1 → 0,01 |
|---|---|---|
| oracle `online` | step 27.336 = **epoch 24** | step 52.394 = epoch 46 |
| quỹ đạo `online` | step 29.614 = **epoch 26** | step 54.672 = epoch 48 |

Đây chính là khiếm khuyết đã biết từ 2026-07-02 (khi đó xử lý bằng PA-C: trung tính hóa câu văn ở Mục 3.10, KHÔNG sửa code). Nó vô hại với bài toán oracle vì tới epoch 24 train_loss đã 0,14 — coi như hội tụ xong. Với bài toán quỹ đạo thì nhát cắt rơi đúng giữa lúc mô hình đang học:

| epoch | 8 | 12 | 16 | 20 | 24 | **26 = cắt LR** | 27 | 49 |
|---|---|---|---|---|---|---|---|---|
| train_loss | 1,9906 | 1,8693 | 1,7503 | 1,6777 | 1,6185 | ↓ | 1,3939 | 1,2660 |

Giảm đều không dấu hiệu chững cho tới epoch 24; sau nhát cắt tụt một nhịp rồi **đóng băng** — 22 epoch cuối chỉ đi được 1,39 → 1,27. val_loss tốt nhất rơi đúng epoch 27, ngay sau nhát cắt. Đây là chữ ký kinh điển của "LR bị cắt khi còn đang học", không phải của mô hình đã hết khả năng.

**✅ QUYẾT ĐỊNH CỦA USER 2026-07-29:** sửa `mode="max"` → `mode="min"` trong `lit_bttr.py` (đảo quyết định PA-C ngày 2026-07-02 — khi đó chọn trung tính hóa văn bản thay vì sửa code); giữ `max_epochs=50`; **chạy lại riêng `online` trước** để lặp lại cổng quyết định rồi mới quyết có chạy 5 cấu hình còn lại. Đang chạy → `lightning_logs/abl_online_traj/lightning_logs/version_2`.

Ảnh hưởng tới bộ số oracle: **cùng lập luận như với EarlyStopping** — số oracle không đổi theo nghĩa đen (checkpoint + `results/` cũ, không chạy lại); so sánh nội bộ trong từng bộ không bị ảnh hưởng (mọi cấu hình trong cùng bộ vẫn dùng thiết lập giống hệt nhau); so chéo hai bộ lệch về phía an toàn vì lịch LR đúng chỉ giúp bên quỹ đạo, làm khoảng cách Mục 4.9 **hẹp lại** chứ không phóng đại. **Mục 3.10 nay phải mô tả lịch LR đúng thay vì câu trung tính hóa, và ghi rõ bộ oracle chạy dưới lịch cũ.**

#### 🔴 ĐÍNH CHÍNH 2026-07-29 — SỬA `mode="min"` LÀ SAI, PHẢI HOÀN NGUYÊN VỀ `mode="max"`

Lần chạy đối chứng `fusion="offline"` (bộ giải mã KHÔNG nhìn `memory2`, tức quỹ đạo bị loại hoàn toàn khỏi phương trình) cho **35,63% (351/985)** trên 2014, so với **47,61% (469/985)** của bộ oracle cũ — **tụt 12 điểm**. Vì nhánh này không dùng quỹ đạo, 12 điểm đó hoàn toàn do thay đổi ở phần dùng chung.

**Nguyên nhân: chính bản sửa `mode="max"` → `mode="min"`.** Với `mode="max"` và `monitor="val_loss"`, val_loss giảm không bao giờ tạo kỷ lục mới nên bộ đếm patience tăng đều và LR bị cắt 10× sau mỗi 10 lần kiểm val — tức **epoch ~24 và ~46**. Đó không phải khiếm khuyết vô hại mà là một **lịch giảm LR theo bậc trên thực tế**, và TOÀN BỘ bộ số oracle (77,06 / 72,62 / 73,29) được tạo ra nhờ nó. Đổi sang `mode="min"` làm LR đứng nguyên 1,0 suốt 50 epoch, mất hẳn giai đoạn tinh chỉnh cuối.

Bằng chứng khớp: online v1 (còn nhát cắt LR) train_loss cuối **1,266**; online v2 ("đã sửa") **1,444** — bản chưa sửa fit tốt hơn. Chi tiết này đã xuất hiện trong log lúc đó nhưng bị đọc sai ý nghĩa.

**⇒ Quyết định PA-C ngày 2026-07-02 (không sửa code, chỉ trung tính hóa văn bản Mục 3.10) là ĐÚNG. Phải hoàn nguyên `lit_bttr.py` về `mode="max"`.** Nếu muốn mô tả trung thực trong luận văn thì nói rõ nó vận hành như lịch giảm bậc tại epoch ~24 và ~46, chứ đừng đổi code — đổi là mọi số phải chạy lại từ đầu, cả hai bộ.

**KHÔNG ảnh hưởng kết luận về nhánh online:** v1 chạy dưới lịch LR cũ vẫn cho ExpRate 0,00%.

**✅ ĐÃ HOÀN NGUYÊN 2026-07-29** (user chốt: hoàn nguyên, không chạy lại offline để tiết kiệm 2h). `lit_bttr.py` trở về `mode="max"`, kèm chú thích giải thích đây là lựa chọn có chủ đích chứ không phải lỗi, để phiên sau đừng "sửa" lại lần nữa.

⚠️ **Việc còn treo:** chưa tách được 12 điểm đó là do lịch LR toàn bộ, hay còn phần do cách gom lô mới (`MAX_TRAJ_SIZE`, `MAX_TRAJ_LEN`, tập mẫu loại theo quỹ đạo thay vì theo SRT). **Phải chạy lại `offline` dưới `mode="max"` trước khi đặt bất kỳ số quỹ đạo nào cạnh số oracle** — nếu về ~47,6% thì hai bộ so được; nếu vẫn ~35% thì tầng gom lô cũng đã đổi kết quả và Mục 4.9 phải ghi chú điều đó. Chi phí: ~2h15.

**Kết quả offline đối chứng (chạy dưới `mode="min"`, GIỮ LÀM BẰNG CHỨNG, không dùng cho luận văn):** `results/traj_abl_offline_2014_results.txt` = 35,63%; checkpoint `lightning_logs/abl_offline_traj/lightning_logs/version_0`, val_loss tốt nhất 0,5859 @ epoch 39.

#### 🔧 SỬA KIẾN TRÚC NHÁNH ONLINE — bản `_traj2` (user duyệt 2026-07-29)

User cho phép đề xuất đổi kiến trúc. Ba thay đổi, chọn theo đúng ba nút thắt đo được, chạy một lượt.

**Chẩn đoán làm căn cứ:**

1. **Trường tiếp nhận quá nhỏ.** 2 lớp `Conv1d(k=5, s=2)` cho trường tiếp nhận 13 điểm ≈ **0,39 đơn vị** chiều dài cung (bước resample 0,03, chuẩn hóa theo chiều cao), trong khi một ký hiệu dài cỡ 1,0–1,5. Không vector memory nào từng nhìn trọn một ký hiệu. Hai lớp đó vốn chỉ để rút ngắn chuỗi, không phải trích đặc trưng — đối lập với DenseNet 16 lớp × 3 khối của nhánh ảnh.
2. **Toạ độ tuyệt đối áp đảo.** std từng chiều trên tập train: x 3,0187 · y 0,2455 · dx 0,1071 · dy 0,0694 · d²x 0,1528 · d²y 0,1018. Qua đúng một `Linear(8→256)` thì x lớn hơn dx **28 lần**; các chiều delta (mã hóa hình dáng nét) chỉ còn vài phần trăm biên độ, còn x tuyệt đối là đặc trưng lý tưởng để học thuộc.
3. **Không có tăng cường dữ liệu**, trong khi quỹ đạo nhạy với tốc độ viết, độ nghiêng, kích thước theo cách ảnh không nhạy.

**Đã cài (2026-07-29):**

| # | Thay đổi | File |
|---|---|---|
| A | Chuẩn hóa 6 chiều liên tục bằng mean/std tập train, lưu thành `register_buffer` nên đi theo checkpoint; 2 bit pen giữ nguyên (chuẩn hóa chúng chỉ khuếch đại sự kiện pen-up hiếm) | `encoder_seq.py` |
| B | `_ReduceStage`: mỗi lần giảm 2× nay gồm 2 tích chập stride 1 có kết nối tắt + 1 tích chập stride 2. Tổng 6 lớp thay vì 2, vẫn giảm đúng 4×, **trường tiếp nhận 13 → 37 điểm ≈ 1,1 đơn vị** (trọn một ký hiệu) | `encoder_seq.py` |
| C | `augment_traj`: xoay ±8°, co giãn 0,85–1,15, nghiêng ±0,2, tịnh tiến nhỏ. **Chỉ áp cho tập train** qua `partial(collate_fn, augment=True)` | `datamodule.py` |

**Hai chi tiết kỹ thuật quan trọng:**

- **KHÔNG dùng BatchNorm/GroupNorm trong khối tích chập.** Cả hai gộp thống kê dọc trục thời gian hoặc theo lô nên phá tính bất biến với padding — đúng cái bẫy đã mất 3 vòng sửa ở phiên trước. Dùng `_ChannelNorm` = LayerNorm theo trục kênh tại từng vị trí, không trộn qua thời gian. Che mask lại sau **mỗi** tích chập, không phải mỗi stage.
- **Tăng cường làm thẳng trên đặc trưng 8 chiều đã lưu**, không phải trên điểm thô: delta là hàm tuyến tính của toạ độ nên cùng một ma trận 2×2 áp y hệt cho `(x,y)`, `(dx,dy)`, `(d²x,d²y)`. Không phải sinh lại `.npz`, không đụng `prep_online.py`.

**Kiểm chứng:** `test_component.py` **PASS TOÀN BỘ 10/10**, bất biến padding 1,76e-06. `SeqEncoder` 3,03M → **4,34M** tham số, tổng mô hình ~9,4M → ~10,7M ⇒ **Bảng 2 phải cập nhật số tham số, số lớp tích chập, và ghi thêm tăng cường dữ liệu**. (`test_component.py` sửa 1 chỗ: phép kiểm gradient nay duyệt cả 3 tích chập của mỗi stage thay vì `stage[0]`.)

Checkpoint vào `lightning_logs/abl_online_traj2` — tách khỏi `_traj` để hai kiến trúc không dùng chung thư mục.

#### 📊 BẢNG TỔNG HỢP MỌI LẦN CHẠY QUỸ ĐẠO (cập nhật 2026-07-29)

Tất cả seed 7, 50 epoch, `EarlyStopping patience=15`, mẫu số 2014 = 985.

| # | Cấu hình | Lịch LR | val_loss tốt nhất | token acc train / 2014 | ExpRate 2014 | Thư mục |
|---|---|---|---|---|---|---|
| 1 | online, encoder Transformer | `max` (gốc) | 2,5392 @ e27 | — | **0,00%** | `abl_online_traj/version_1` |
| 2 | online, encoder Transformer | `min` (sai, đã hoàn nguyên) | 2,5169 @ e45 | 73,21 / 42,29 | **0,00%** | `abl_online_traj/version_2` |
| 3 | online, **+conv sâu +chuẩn hóa +tăng cường** | `max` | 2,4736 @ e39 | 76,07 / 46,03 | **0,00%** | `abl_online_traj2/version_0` |
| 4 | **offline** (đối chứng) | `min` (sai) | 0,5859 @ e39 | 96,18 / 84,46 | **35,63%** | `abl_offline_traj/version_0` |
| 5 | online, **+encoder BiGRU theo TAP** | `max` | 2,4060 @ e45 | 76,35 / 47,69 | **0,30%** (3/985) | `abl_online_traj3/version_0` |

## 🤝 BÀN GIAO SANG COWORK — 2026-07-30

> Phiên Claude Code kết thúc tại đây. **Mọi con số dưới đây đọc trực tiếp từ file trên đĩa, không có số nào ước lượng hay nhớ lại.** Đường dẫn ghi tương đối từ gốc dự án `C:\Users\Admin\lv\lvtn`.
> Việc kế tiếp: **viết luận văn**. Chưa sửa một chữ nào trong 2 file docx.

### Trạng thái mã nguồn

- Nhánh git **`online-traj`**, đã commit **`cee0c45`** (nối tiếp `f6eef03`). **KHÔNG push, chưa từng chạy lệnh nào tới `origin`** — user yêu cầu tuyệt đối không push.
- Diff `f6eef03..cee0c45`: 13 file, +1322 −154 dòng. Chi tiết: `datamodule.py` +194, `encoder_seq.py` +331, `lit_bttr.py` +67, `bttr.py` +51, `custom_train.py` +50, `test_component.py` +206, `utils.py` +6, `config.yaml` +4, `predict_test.py` +39, `test_all.py` +12, cộng 3 file mới trong `tools/`.
- **`decoder.py`, `encoder_img.py`, `pos_enc.py` KHÔNG ĐỔI** — kiểm bằng `git diff --stat f6eef03 HEAD --` trên 3 file này, kết quả rỗng. Đóng góp kiến trúc của luận văn nguyên vẹn.
- Không đưa vào commit: 2 docx luận văn, `references/`, `demo/`, `bao_ve/`, `inkml.zip` (29,4 MB, đã thêm vào `.gitignore`), `online/*.npz`.

### Checkpoint của từng cấu hình (đường dẫn đã xác minh tồn tại)

| Cấu hình | val_loss | Checkpoint |
|---|---|---|
| `offline` | 0,4458 | `lightning_logs/abl_offline_traj3/lightning_logs/version_0/checkpoints/epoch=47-step=54672-val_loss=0.4458.ckpt` |
| `online`+phụ | 0,5446 | `lightning_logs/abl_online_aux/.../epoch=49-step=56950-val_loss=0.5446.ckpt` |
| `dual_shared`+phụ | 0,3622 | `lightning_logs/abl_dual_shared_aux/.../epoch=39-step=45560-val_loss=0.3622.ckpt` |
| `dual_shared_uni`+phụ | 0,3748 | `lightning_logs/abl_dual_shared_uni_aux/.../epoch=47-step=54672-val_loss=0.3748.ckpt` |
| `concat`+phụ | 0,3731 | `lightning_logs/abl_concat_aux/.../epoch=29-step=34170-val_loss=0.3731.ckpt` |
| `cascaded`+phụ | 0,3510 | `lightning_logs/abl_cascaded_aux/.../epoch=33-step=38726-val_loss=0.3510.ckpt` |

⚠️ `lightning_logs/` nằm trong `.gitignore` nên checkpoint **chỉ có trên máy này**, không theo git.

### Cách tái lập

```
python tools/prep_online.py                 # inkml.zip -> online/*.npz (~10 giây)
python tools/prep_stroke_labels.py          # -> online/stroke_labels_train.npz
python custom_train.py --fusion dual_shared --aux-stroke-weight 0.5 --suffix aux
python tools/run_ablations.py               # chạy nối tiếp phần còn lại + đo 3 tập
```
`custom_train.py` nay nhận tham số dòng lệnh; mặc định = hệ chính. Mọi lệnh python cần `conda run -n bttr --no-capture-output python ...`.

### Việc CHƯA làm

1. **Chưa viết gì vào luận văn.** Danh sách thay đổi đề xuất ở mục "Kế hoạch viết" bên dưới.
2. **Trần 50 epoch đã cắn** với `online`+phụ (val_loss tốt nhất rơi đúng epoch 49, lần kiểm cuối). Nới ngân sách có thể cải thiện nhưng **phải chạy lại cả 6 cấu hình** (~15 giờ). User chưa quyết.
3. **Mẫu số 2014 vẫn là 985** (mẫu `505_em_51` nhãn > 200 token bị `data_iterator` loại). Nay số liệu đặt cạnh BTTR/TAP nên mẫu số lệch thành sai giao thức. User chưa chốt có sửa `test_all.py` tính trên đủ 986 hay không. **Tôi không tự sửa cách đếm.**
4. **Chưa chạy `offline` kèm giám sát phụ.** Về nguyên tắc không đổi (bộ giải mã chế độ `offline` không dùng `memory2`), nhưng nếu muốn giao thức tuyệt đối đồng nhất thì chạy lại 2,5h.

### Kế hoạch viết (đã trình bày với user, chưa duyệt)

Ba ràng buộc của thầy ngày 2026-07-29 buộc số quỹ đạo thật phải **dẫn dắt**, nên khối lượng lớn hơn "thêm một mục phụ lục":

| Vị trí | Thay đổi |
|---|---|
| Bảng 1 | dòng "Ours" → 52,59 / 48,30 / 51,96 (quỹ đạo thật), đặt cạnh BTTR 53,96 và TAP 50,41 |
| Abstract | thay 77,06/74,17 → 52,59/50,89; ablation: hợp nhất +2,98đ so với chỉ-ảnh, hai chiều +8,98đ |
| Mục 1 | câu trả lời nay khẳng định được dưới đầu vào thật |
| Mục 3 | tiểu mục mới: đọc `<trace>`, đặc trưng 8 chiều TAP, resample 0,03, chuẩn hóa, conv, BiGRU, giám sát phụ. **Phải nêu rõ `traceGroup` chỉ đọc lúc huấn luyện** |
| Bảng 2 | thêm bước resample, hệ số giảm, loại bộ mã hóa, trọng số phụ 0,5, tăng cường, 9,2M tham số; dòng vocab 111 đánh dấu chỉ áp cho biến thể oracle |
| Hình 3, Hình 4 | nhãn "linearized SRT · token ids [b, l1]" → đặc trưng điểm 8 chiều. Sửa SVG EN+VI, render, duyệt, nhúng. **Rủi ro kỹ thuật cao nhất, cắt trước nếu thiếu thời gian** |
| Chương 4 | viết mới phần lớn: kết quả chính, so công bố, 3 trục ablation kèm McNemar, mục cận trên oracle, mục khoảng cách oracle↔thật |
| Mục 5 | kết luận mới; hướng phát triển: trần epoch, hệ hai tầng, hợp nhất mức encoder (MMSCAN-E) |
| `bao_ve/` | slide + kịch bản làm lại số |

**Deadline user nêu: chốt trước thứ Tư 5/8/2026.**

---

#### ✅✅ BỘ SỐ HOÀN CHỈNH VỚI QUỸ ĐẠO THẬT — 2026-07-30 (ĐỌC MỤC NÀY TRƯỚC)

Sáu cấu hình, seed 7, 50 epoch, `mode="max"`, `patience=15`, **không oracle**. Mọi cấu hình dùng bộ mã hóa BiGRU + giám sát phụ mức nét (trừ `offline`, vốn không dùng nhánh online).

| Cấu hình | 2014 | 2016 | 2019 | micro |
|---|---|---|---|---|
| chỉ ảnh (`offline`) | 49,95% | 47,60% | 46,54% | 47,91% |
| chỉ quỹ đạo (`online`) | 40,81% | 32,52% | 36,36% | 36,36% |
| **`dual_shared` (Ours)** | **52,59%** | **48,30%** | **51,96%** | **50,89%** |
| `cascaded` | 53,40% | 48,21% | 51,71% | **51,01%** |
| `concat` | 52,28% | 45,60% | 48,71% | 48,69% |
| `dual_shared_uni` (một chiều) | 45,58% | 39,49% | 41,20% | 41,91% |

Mẫu số 985 / 1147 / 1199 (2019 đổi từ 1198 vì `UN19_1001_em_0` nay có quỹ đạo). Kết quả: `results/traj_abl_*_aux_*.txt` và `results/traj_abl_offline_lrmax_*.txt`.

**Đối chiếu công bố trên CROHME 2014:** BTTR 53,96% · TAP 50,41% · **luận văn 52,59% (`dual_shared`), 53,40% (`cascaded`)**.

##### Kiểm định McNemar (chính xác, hai phía)

| So sánh | 2014 | 2016 | 2019 | Kết luận |
|---|---|---|---|---|
| hợp nhất vs chỉ ảnh | p=0,087 | p=0,640 | **p<0,0001** | hợp nhất vượt, chắc chắn ở 2019 |
| **hai chiều vs một chiều** | **p<0,0001** | **p<0,0001** | **p<0,0001** | **có ý nghĩa CẢ BA TẬP, +8,98đ micro** |
| `dual_shared` vs `cascaded` | p=0,580 | p=1,000 | p=0,877 | **không phân biệt được** |
| `dual_shared` vs `concat` | p=0,876 | **p=0,022** | **p=0,005** | shared-query vượt concat |

##### Ba khẳng định dùng được cho luận văn

1. **Luận điểm trung tâm ĐỨNG VỮNG dưới đầu vào thật:** hợp nhất (50,89%) vượt cả hai nhánh đơn (47,91% và 36,36%). Đây là điều bộ số oracle không chứng minh được một cách thuyết phục vì oracle làm nhánh online mạnh giả tạo.
2. **Huấn luyện hai chiều +8,98 điểm micro, có ý nghĩa trên cả ba tập.** Kết quả chắc chắn nhất của cả đợt. ⚠️ Lần chạy này suýt bị bỏ — user từng quyết bỏ vì cho rằng nhánh online yếu thì trục này cũng vô nghĩa; cơ chế hai chiều nằm ở phía bộ giải mã nên độc lập với chất lượng nhánh online.
3. **Trục thiết kế hợp nhất ĐẢO THỨ TỰ so với oracle.** Oracle: concat 75,89 > shared 74,17 > cascaded 73,51. Quỹ đạo thật: cascaded 51,01 ≈ shared 50,89 > concat 48,69. Shared-query và cascaded **không phân biệt được về mặt thống kê**; cả hai vượt concat. ⚠️ Vẫn giữ ràng buộc cũ: **KHÔNG tuyên bố shared-query là tốt nhất** — nói đúng là "ngang cascaded, vượt concat".

##### Việc còn mở

- **Trần 50 epoch nay đã cắn** với cấu hình `online`+phụ (val_loss tốt nhất rơi đúng epoch 49). Nới ngân sách có thể cải thiện thêm nhưng **phải chạy lại toàn bộ** để giữ tính so sánh. Chưa làm, chờ user quyết.
- Mẫu số 2014 vẫn là 985 (mẫu `505_em_51` nhãn > 200 token). User chưa chốt có sửa `test_all.py` tính trên đủ 986 hay không.
- Chưa chạy `offline` kèm giám sát phụ — về nguyên tắc không đổi vì bộ giải mã ở chế độ `offline` không dùng `memory2`, nhưng nếu muốn tuyệt đối đồng nhất giao thức thì chạy lại 2,5h.

#### 🟩 [LỊCH SỬ] ĐẢO CHIỀU 2026-07-30 — HỆ HỢP NHẤT ĐẠT 48,02% VỚI QUỸ ĐẠO THẬT (bản CHƯA có giám sát phụ)

**Kết luận "dừng thí nghiệm, viết kết quả âm tính" ở khối bên dưới ĐÃ BỊ VƯỢT QUA.** Sai lầm phương pháp của tôi: suốt hai ngày chỉ đo các cấu hình **đơn lẻ** rồi suy ra cả hệ, trong khi cấu hình hợp nhất — thứ luận văn thật sự nói tới — chưa hề được đo.

| Cấu hình, đầu vào quỹ đạo thật, seed 7, 50 epoch | val_loss | token acc train/2014 | **ExpRate 2014** |
|---|---|---|---|
| chỉ quỹ đạo (`online`, BiGRU) | 2,4060 | 76,35 / 47,69 | 0,30% (3/985) |
| chỉ ảnh (`offline`) ⚠️ lịch LR sai | 0,5859 | 96,18 / 84,46 | 35,63% (351/985) |
| **hợp nhất (`dual_shared`, BiGRU)** | **0,4506 @ e47** | — | **48,02% (473/985)** |

Checkpoint: `lightning_logs/abl_dual_shared_traj3/lightning_logs/version_0/checkpoints/epoch=47-step=54672-val_loss=0.4506.ckpt`. Kết quả: `results/traj_abl_dual_shared_2014_results.txt`.

**48,02% là con số KHÔNG có oracle, đặt cạnh TAP (50,41%) và BTTR (53,96%) được** — đúng yêu cầu ban đầu của thầy. Dự đoán mẫu cho thấy nhận dạng thật sự hoạt động (`- 7`, `k N`, `[ b ]` đúng; lỗi là lỗi hợp lý như `1 T` thay `1 7`), khác hẳn bản chỉ-quỹ-đạo vốn sinh LaTeX không liên quan.

**⚠️ CHƯA kết luận được nhánh online có đóng góp hay không.** Con số chỉ-ảnh 35,63% chạy dưới `mode="min"` (lịch LR sai, đã biết làm mất ~12 điểm). Bộ oracle cũ cho chỉ-ảnh 47,61%. Nếu chạy lại đúng giao thức mà ra ~47–48% thì hợp nhất gần như không thêm gì và luận điểm trung tâm yếu; nếu thấp hơn rõ thì hợp nhất có tác dụng thật. **Đây là thí nghiệm quan trọng nhất còn lại** — đang chạy → `lightning_logs/abl_offline_traj3`.

#### 🚀 GIÁM SÁT PHỤ MỨC NÉT TỪ `traceGroup` — BƯỚC ĐỘT PHÁ (2026-07-30)

Ý tưởng: gắn một đầu phân loại lên bộ mã hóa online, bắt nó đoán **ký hiệu của từng nét bút**, cộng loss đó vào loss chính với trọng số 0,5.

**Vì sao KHÔNG phải oracle** (điểm sống còn, phải nêu rõ trong luận văn): `<traceGroup>` chỉ được đọc ở **tập train**. Đầu phụ không tham gia suy luận, không nhãn nào đi vào đầu vào mô hình, và tập 2016/2019 là bản noGT nên không có `traceGroup` kể cả muốn đọc. Khác hoàn toàn biến thể SRT cũ, nơi nhãn được **đưa thẳng vào làm đầu vào bộ mã hóa**.

**Dữ liệu:** `tools/prep_stroke_labels.py` → `online/stroke_labels_train.npz`. 8.834 biểu thức, **121.161/121.306 nét có nhãn (99,88%)**, 0 sai lệch số nét so với `.npz` đặc trưng. Cần ánh xạ `\lt`→`<`, `\gt`→`>` (CROHME ghi khác từ điển decoder). Lượng giám sát tăng **13,7 lần** so với 8.834 chuỗi LaTeX.

**Hiệu quả, đo trên `fusion="online"` để cô lập bộ mã hóa:**

| epoch | BiGRU không giám sát phụ | BiGRU + giám sát phụ |
|---|---|---|
| 1 | 3,336 | 3,102 |
| 3 | 3,406 | 2,865 |
| 5 | 3,345 | 2,705 |
| 7 | 2,965 | 2,114 |
| 9 | 2,973 | **1,439** |
| 17 | ~2,85 | **1,150** |

Bản không giám sát phụ có val_loss **tốt nhất của cả 50 epoch là 2,4060**; bản có giám sát phụ vượt qua mốc đó ngay từ epoch 9. `train_aux` giảm 4,023 → 1,420 sau 7 epoch, tức bộ mã hóa thật sự học nhận ra ký hiệu từ nét.

**Hệ quả:** giám sát phụ trở thành một phần của hệ thống chính, nên **mọi cấu hình ablation phải chạy kèm nó** mới so sánh được. Bộ số `dual_shared` 48,02% ở trên là bản KHÔNG có giám sát phụ và sẽ thành dòng "trước cải tiến".

**Code:** `encoder_seq.py` (`aux_head`, `stroke_logits`, tách `_stroke_ids`/`_pool_by_id`), `bttr.py` (`traj_aux_classes`, `forward(..., return_aux=)`), `lit_bttr.py` (`aux_stroke_weight`), `datamodule.py` (`load_stroke_labels`, `Batch.stroke_labels`, `data_iterator` mang phần tử thứ 5), `tools/prep_stroke_labels.py`. Mặc định `aux_stroke_weight=0` nên checkpoint cũ vẫn nạp đúng. Kiểm: căn chỉnh nét↔nhãn khớp từng mẫu, loss phụ khởi điểm 4,7201 ≈ ln(113), gradient tới `aux_head` và `point_proj`, `test_component.py` 11/11 PASS.

**Hạ tầng:** `custom_train.py` nay nhận tham số dòng lệnh (mặc định = hệ chính, hành vi không đổi); `tools/run_ablations.py` chạy nối tiếp huấn luyện → chọn checkpoint tốt nhất → đo 3 tập cho từng cấu hình.

#### 🔬 Kết quả biến thể mức nét SCAN [22] — KÉM HƠN mức điểm

| Bộ mã hóa online | token acc train/2014 | val_loss |
|---|---|---|
| Transformer, kiến trúc gốc | 73,21 / 42,29 | 2,5169 |
| Transformer + conv sâu + chuẩn hóa + tăng cường | 76,07 / 46,03 | 2,4736 |
| **BiGRU theo TAP [33]** (dùng cho các cấu hình hợp nhất) | **76,35 / 47,69** | **2,4060** |
| BiGRU + gộp mức nét theo SCAN [22] | 72,99 / 43,08 | 2,6277 |

Bốn thiết kế khác nhau về bản chất đều rơi trong dải 42–48% token accuracy. Giả thuyết cho việc mức nét kém hơn: gộp trung bình ~20 điểm thành một vector xóa mất quỹ đạo hình dáng bên trong nét, vốn là tín hiệu phân biệt ký hiệu. Công tắc `traj_stroke_pooling` giữ lại trong code (mặc định `False`) nếu sau này muốn thử cách tổng hợp tinh hơn.

**Xác minh đường ống dữ liệu (600 file train):** `<trace>` trong InkML = 15,89/file, số nét trong `.npz` = 15,89/file, **0 sai lệch**. Ký hiệu (`traceGroup`) = 11,12/file, tức 1,43 nét/ký hiệu — đúng như kỳ vọng. Không mất nét ở khâu resample.

#### 🛑 [ĐÃ VƯỢT QUA] CHỐT 2026-07-30 ~00:30 — DỪNG THÍ NGHIỆM, CHUYỂN SANG VIẾT

**Ràng buộc mới, user cho biết 2026-07-29: chỉ còn 1 tuần, phải chốt trước thứ Tư 5/8/2026.**

Lần chạy #5 (BiGRU theo TAP) cho **ExpRate 0,30% (3/985)** — lần đầu khác 0 nhưng vô nghĩa về thực tiễn. Chuỗi ba lần cải tiến kiến trúc cho thấy lợi tức giảm dần rõ rệt:

| Biến thể nhánh online | token acc 2014 | ExpRate |
|---|---|---|
| Transformer, kiến trúc gốc | 42,29% | 0,00% |
| + conv sâu + chuẩn hóa đặc trưng + tăng cường | 46,03% | 0,00% |
| + bộ mã hóa BiGRU theo TAP [33] | 47,69% | 0,30% |
| **mốc cần đạt** (suy từ offline: 84,46% token → 35,63% ExpRate) | **85–90%** | 40–50% |

Mỗi can thiệp được 2–4 điểm token accuracy; cần thêm ~37 điểm nữa. **Vá kiến trúc tiếp sẽ không đổi cục diện**, và mỗi lần thử tốn 3 giờ trong ngân sách 7 ngày.

**Ước lượng đường "chạy tiếp cho đủ bộ số":** 5 cấu hình × 2,5h + 6 cấu hình × 3 tập × 15' ≈ **17 giờ máy chạy liên tục không hỏng**, tức 2,5 ngày, còn lại 4,5 ngày cho toàn bộ việc thay số Bảng 3/4/5, viết lại 4.4–4.10, sửa Hình 3/4 cả EN lẫn VI, Bảng 2, Abstract, Mục 1, 5.1, Bảng 1, slide, xuất 2 PDF. **Không kịp** với quy trình soạn old→new + duyệt của dự án.

**⇒ Phương án đã chọn: giữ bộ oracle làm kết quả chính (thầy đã chốt giữ cả hai bộ), thêm một mục Chương 4 trình bày trung thực kết quả âm tính của nhánh quỹ đạo.** Ưu điểm về khối lượng: **KHÔNG phải đụng Bảng 3/4/5, KHÔNG phải sửa Hình 3/4**, vì hệ thống chính vẫn là hệ SRT.

⚠️ **Phải nói rõ với user và thầy:** phương án này **không** giao được con số quỹ đạo đặt cạnh TAP/BTTR như thầy yêu cầu ban đầu. Cần báo thầy **sớm, kèm bằng chứng chẩn đoán**, để cho thấy đây là kết luận có cơ sở chứ không phải bỏ cuộc.

**Việc chưa làm (user hủy lúc 00:30 để tắt máy): chạy lại `offline` dưới `mode="max"`.** Vẫn cần trước khi đặt số quỹ đạo cạnh số oracle — số #4 hiện có (35,63%) chạy dưới lịch LR sai, chưa biết 12 điểm chênh so với 47,61% là do lịch LR toàn bộ hay còn do cách gom lô mới. ~2h15.

**Lập luận cho mục mới của Chương 4 (đã có đủ bằng chứng, chỉ cần viết):**
- Đường ống quỹ đạo thật đã dựng xong và kiểm chứng: phân bố đặc trưng train↔test khớp, ghép cặp quỹ đạo↔ảnh đúng 6/6 trên tập 2014, bất biến padding 1,2–1,8e-06, gradient thông suốt, overfit 1 lô xuống 3e-4.
- Thất bại khu trú ở nhánh quỹ đạo: cùng đường ống, nhánh ảnh đạt token 96,18/84,46 và ExpRate 35,63%.
- Nhánh quỹ đạo vừa thiếu khớp vừa quá khớp (76,35/47,69, khoảng cách 28,66 so với 11,72 của nhánh ảnh) — bộ mã hóa không trích được đặc trưng khái quát hóa được.
- Ba biến thể kiến trúc, gồm cả bộ mã hóa BiGRU đúng theo TAP, đều không thoát khỏi vùng này.
- Đường học tách nhau tại epoch 3 giữa bản oracle và bản quỹ đạo (oracle sụp xuống 1,43 khi phát hiện có thể chép nhãn từ SRT; quỹ đạo tiếp tục giảm đều) — **bằng chứng định lượng trực tiếp cho luận điểm "oracle cho không phần lớn bài toán"**, dùng cho Mục 4.9.

File kết quả: `results/traj_abl_online_2014_results.txt` (#1), `..._online_lrfix_...` (#2), `..._online_arch2_...` (#3), `..._offline_...` (#4). **`results/` cũ (bộ oracle) nguyên vẹn, không đụng.**

#### 🔬 KẾT LUẬN CHẨN ĐOÁN — nhánh online vừa THIẾU KHỚP vừa QUÁ KHỚP

Đo cùng một chỉ số trên hai nhánh (teacher-forced, eval mode):

| | token acc train | token acc 2014 | khoảng cách | ExpRate |
|---|---|---|---|---|
| online (quỹ đạo, bản #3) | 76,07% | 46,03% | **30,03 đ** | 0,00% |
| offline (ảnh, bản #4) | 96,18% | 84,46% | **11,72 đ** | 35,63% |

Nhánh online thua ở **cả hai trục**: quá khớp gấp 2,6 lần nhánh ảnh, **đồng thời** fit chính tập train kém hơn 20 điểm, và train_loss vẫn đang giảm ở epoch 49 (chưa hội tụ). Hai triệu chứng cùng một gốc: bộ mã hóa không trích được đặc trưng khái quát hóa được nên mô hình xoay sang ghi nhớ dấu hiệu riêng từng mẫu. **Vì vậy chỉ thêm chính quy hóa hoặc tăng cường dữ liệu là không đủ — chúng chữa quá khớp nhưng làm thiếu khớp nặng thêm.**

**Mốc hiệu chỉnh quan trọng:** nhánh offline 84,46% token → 35,63% ExpRate. Suy ra để ExpRate vào vùng 40–50% thì **token accuracy phải đạt khoảng 85–90%**. Hiện ở 46%. Đây là khoảng cách về phương pháp, không phải về tinh chỉnh tham số.

**Dữ kiện định hướng:** TAP [33] đạt 50,41% ExpRate CROHME 2014 trên **đúng bộ dữ liệu này, đúng số mẫu này, học đầu-cuối từ quỹ đạo, KHÔNG dùng traceGroup**. Nên dữ liệu là đủ; khoảng cách nằm ở phương pháp. Khác biệt lớn nhất: TAP dùng GRU hai chiều nhiều tầng, ta dùng Transformer — với 8.834 mẫu thì Transformer thiếu thiên kiến quy nạp và đói dữ liệu, đúng chế độ mà RNN thắng. Đây là căn cứ cho lần chạy #5.

#### 🔧 BẢN `_traj3` — ĐỔI BỘ MÃ HÓA ONLINE SANG BiGRU THEO TAP (user duyệt 2026-07-29)

| Thay đổi | File |
|---|---|
| Thêm công tắc `traj_encoder: "transformer" \| "gru"`, **mặc định `"transformer"`** để checkpoint `_traj2` vẫn nạp đúng kiến trúc chúng được huấn luyện (mặc định `"gru"` sẽ khiến `strict=False` nạp nhầm trong im lặng) | `encoder_seq.py`, `bttr.py`, `lit_bttr.py` |
| Đường `gru`: `nn.GRU` 3 tầng hai chiều, `hidden = d_model/2` mỗi chiều nên đầu ra vẫn 256; **bỏ mã hóa vị trí** ở nhánh này (hồi quy đã mang thứ tự) | `encoder_seq.py` |
| `custom_train.py` truyền `traj_encoder="gru"`, `out_dir` hậu tố `_traj3` | `custom_train.py` |

**Chỗ dễ hỏng nhất, đã xử lý:** phải `pack_padded_sequence` trước RNN, nếu không chiều lùi sẽ bắt đầu bằng việc đọc vùng đệm và mã hóa của một mẫu phụ thuộc vào mẫu nào nằm chung lô — đúng loại lỗi im lặng đã mất 3 vòng sửa ở nhánh tích chập. `test_component.py` nay kiểm bất biến padding cho **cả hai** đường: transformer 1,76e-06, gru 1,19e-06. **11/11 PASS.**

Tham số: 10,7M (Transformer) → **9,2M** (GRU) — GRU nhẹ hơn thứ nó thay thế.

Việc này cũng tạo một trục ablation sạch cho Chương 4 nếu cần: cùng dữ liệu, cùng bộ giải mã, cùng khối tích chập, chỉ khác bộ mã hóa chuỗi.

#### ⏭️ BÀN GIAO — VIỆC CẦN LÀM TIẾP (2026-07-29)

**Trạng thái code:** nhánh git `online-traj`, **chưa commit**. `decoder.py`, `encoder_img.py`, `pos_enc.py` vẫn nguyên vẹn — đóng góp kiến trúc của luận văn không đổi. `thesis/` không bị đụng ngoài chính file này.

⚠️ **`ModelCheckpoint` đặt `save_weights_only=True` nên checkpoint KHÔNG chứa trạng thái optimizer** — không nối tiếp huấn luyện giữa chừng được, dừng là phải chạy lại từ đầu. Nhớ điều này trước khi ngắt một lần chạy.

**Việc còn mở, theo thứ tự:**

1. Đọc kết quả lần chạy #5 (BiGRU). Nếu token accuracy 2014 lên vùng 80%+ thì hướng đúng, chạy tiếp 5 cấu hình còn lại. Nếu vẫn ~46% thì hướng vá kiến trúc đã hết dư địa — chuyển sang phương án viết kết quả âm tính (mục 3 dưới).
2. **Chạy lại `offline` dưới `mode="max"`** (~2h15) — bắt buộc trước khi đặt bất kỳ số quỹ đạo nào cạnh số oracle. Số #4 hiện có chạy dưới lịch LR sai. Cần biết 12 điểm chênh (47,61 → 35,63) là do lịch LR toàn bộ, hay còn phần do cách gom lô mới.
3. Nếu nhánh online không đạt: viết trung thực. Thầy đã chốt giữ cả hai bộ số nên bộ oracle vẫn là kết quả chính; Chương 4 thêm mục nói thẳng rằng với quỹ đạo thật, trong giới hạn kiến trúc và ngân sách của luận văn, nhánh online không đạt độ chính xác dùng được — và khoảng cách đó định lượng chính xác phần mà oracle cho không. Hợp lệ về khoa học, đúng tinh thần Mục 4.9, nhưng **không** đáp ứng yêu cầu ban đầu của thầy về con số so sánh được.
4. **Câu hỏi chưa ai trả lời, giờ đã thành then chốt:** có ràng buộc thời gian bảo vệ không? (mục 10 kế hoạch)

**Nếu chọn đi tiếp sau này, Bảng 2 phải cập nhật:** số tham số (9,4M → 9,2M hoặc 10,7M tùy bộ mã hóa), số lớp tích chập nhánh online (2 → 6), dòng tăng cường dữ liệu, chuẩn hóa đặc trưng, loại bộ mã hóa chuỗi. Mục 3.10 mô tả lịch LR đúng như nó vận hành (giảm bậc ở epoch ~24 và ~46).

**⚠️ Chưa chứng minh được:** sửa lịch LR thì có đạt vùng 40–50% như kế hoạch kỳ vọng hay không. Mốc TAP 50,41% là của kiến trúc GRU chuyên cho online, huấn luyện dài hơn nhiều. Việc cần làm là bỏ nút thắt tối ưu hóa rồi mới biết trần thật của đường ống này ở đâu.

#### ⛔ Chẩn đoán lần chạy `online` quỹ đạo dừng sớm (2026-07-29)

`EarlyStopping(monitor="val_loss", mode="min")` ở `custom_train.py` **không truyền `patience`** nên lấy mặc định **3**. Với `check_val_every_n_epoch=2`, chỉ 6 epoch không cải thiện là cắt. Các lần chạy oracle không bao giờ chạm callback này (val_loss lao dốc từ epoch 3, chạy tới epoch 45–47) nên khiếm khuyết chưa từng lộ.

Bốn phép kiểm đã loại trừ mọi nguyên nhân dữ liệu/kiến trúc:

1. **Phân bố đặc trưng train↔test khớp** (đọc thẳng `online/*.npz`): y mean 0,49–0,51; dx/dy/d²x/d²y sai lệch < 5%; pen_up rate 0,0437–0,0452; độ dài p50 263–301. Không lệch tiền xử lý giữa các tập.
2. **Ghép cặp quỹ đạo ↔ ảnh ĐÚNG trên tập 2014** — dựng hình từ `online/2014.npz` đặt cạnh `.bmp` của `data.zip`, 6/6 khớp từng nét (vd. `RIT_2014_180` = `∫√a ∫√-a = ∫√-a² = j√a²`). ⚠️ Đây là phép kiểm mà `tools/prep_online.py --preview` **không làm được** vì hardcode `train/`; trước đó chỉ tập train từng được kiểm.
3. **Không có khoảng cách train↔val bất thường.** Đo lại checkpoint epoch 7 ở eval mode: loss trên train 3,85, trên val 4,57, mốc khởi tạo ngẫu nhiên 5,47. Con số "loss=1,71" trên thanh tqdm là trung bình trượt, không phải `train_loss` ghi log (thực tế 2,0646).
4. **Đường học đúng như dự đoán lý thuyết.** train_loss hai lần chạy trùng nhau 3 epoch đầu (3,30/2,65/2,48 so với 3,23/2,66/2,40) — cả hai đang học mô hình ngôn ngữ LaTeX. Epoch 3 bản oracle sụp xuống 1,43 vì chép được nhãn từ SRT; bản quỹ đạo tiếp tục giảm đều 2,37 → 2,06, **không có dấu hiệu chững**. Đây là bằng chứng định lượng cho luận điểm "oracle cho không phần lớn bài toán", dùng được cho Mục 4.9.

**Cách dừng thực tế của 5 lần chạy oracle** (đọc từ `metrics.csv`, kiểm 2026-07-29 — EarlyStopping CÓ kích hoạt ở 3/5, trái với ghi nhận sơ bộ ban đầu):

| Lần chạy oracle | Kết thúc | val_loss tốt nhất | Cách dừng |
|---|---|---|---|
| `cascaded` | epoch 49 | 0,1557 @ 49 | chạy hết 50 |
| `concat` | epoch 49 | 0,1511 @ 47 | chạy hết 50 |
| `dual_shared` | epoch 45 | 0,1566 @ 39 | EarlyStopping |
| `dual_shared_uni` | epoch 47 | 0,1559 @ 41 | EarlyStopping |
| `online` | epoch 47 | 0,2046 @ 41 | EarlyStopping |

Ba lần bị cắt đều đã hội tụ: checkpoint tốt nhất nằm trước điểm cắt 6 epoch, val_loss đứng ở chữ số thập phân thứ tư (0,1566 → 0,1601 → 0,1616 → 0,1618). Mất 2–4 epoch cuối không đổi kết quả.

**✅ QUYẾT ĐỊNH CỦA USER 2026-07-29: nới `patience=15`, giữ `max_epochs=50`.** Áp cho cả 6 cấu hình quỹ đạo.

Ảnh hưởng tới bộ số oracle, ba tầng:
- **Số oracle không đổi theo nghĩa đen** — là checkpoint trên đĩa + 24 file `results/`, không chạy lại (và không chạy lại được bằng code hiện tại, xem việc (b) chờ chốt).
- **So sánh nội bộ từng bộ không bị ảnh hưởng** — mọi khẳng định chính của luận văn là so giữa các cấu hình trong cùng một bộ, mà trong mỗi bộ 6 cấu hình vẫn dùng thiết lập giống hệt nhau.
- **So chéo hai bộ (Mục 4.9) lệch nhẹ về phía an toàn** — bên quỹ đạo được trọn 50 epoch, bên oracle thực nhận 45–49, nên sai lệch chỉ có thể làm khoảng cách oracle↔quỹ đạo **hẹp lại**, không phóng đại. Với luận điểm "oracle là cận trên rộng rãi" thì đây là sai lệch bảo thủ. **Bảng 2 (hoặc Mục 4.3) phải ghi tiêu chí dừng của từng bộ.**

⚠️ **Rủi ro còn lại, cần theo dõi sau lần chạy tới: trần 50 epoch.** `cascaded` oracle đạt val_loss tốt nhất ngay tại lần kiểm cuối (epoch 49) — có thể vẫn đang cải thiện khi hết ngân sách. Bài toán quỹ đạo khó hơn hẳn nên khả năng chạm trần còn cao hơn. Nếu bản quỹ đạo bị huấn luyện thiếu thì thiệt hại rơi đúng hai chỗ cần nhất: số đặt cạnh TAP/BTTR thấp giả tạo, và khoảng cách Mục 4.9 bị thổi phồng theo hướng tâng bốc số oracle. **Kiểm ngay sau khi chạy xong: checkpoint tốt nhất rơi vào epoch nào? Nếu 47–49 thì trần đang cắn, phải bàn lại ngân sách epoch trước khi chạy 5 cấu hình còn lại.**

**Lỗi môi trường đã gặp (Windows):** script chạy trực tiếp qua `python -c` hoặc file ngoài repo bị `BrokenPipeError` khi `num_workers>0` vì thiếu `if __name__ == "__main__"` (Windows dùng spawn). `custom_train.py` vốn đã có guard nên không ảnh hưởng đường huấn luyện chính thức.

### Việc kế tiếp (user chuyển sang Claude Code, 2026-07-28)

Chạy tuần tự, dừng lại ngay khi một bước không đạt:

1. **`fast_dev_run`** — chỉ để chắc shape và mask không nổ trên GPU thật:
   ```
   conda run -n bttr --no-capture-output python -c "from pytorch_lightning import Trainer, seed_everything; from bttr.datamodule import CROHMEDatamodule; from bttr.lit_bttr import LitBTTR; seed_everything(7); m=LitBTTR(d_model=256,growth_rate=24,num_layers=16,nhead=8,dim_feedforward=1024,dropout=0.3,num_encoder_layers=3,num_decoder_layers=3,beam_size=10,max_len=200,alpha=1.0,learning_rate=1.0,patience=20); Trainer(gpus=1,fast_dev_run=True).fit(m, CROHMEDatamodule(batch_size=32,num_workers=5))"
   ```
2. **Overfit 1 batch** — thêm tạm `overfit_batches=1, max_epochs=30` vào `Trainer` trong `custom_train.py`. Loss phải xuống gần 0; không xuống nghĩa là đường gradient qua nhánh mới có chỗ đứt.
3. **Chạy `FUSION="online"` trước tiên** (chỉ nhánh quỹ đạo). Đây là phép thử quyết định: TAP đơn mô hình đạt **50,41% ExpRate CROHME 2014** [33]. Ra vùng 40–50% là tiền xử lý đúng, chạy tiếp. Dưới 15% thì gần như chắc chắn lỗi tiền xử lý hoặc mask — dừng lại sửa, đừng chạy các cấu hình còn lại.
4. **5 cấu hình còn lại**, cùng `seed_everything(7)`: `offline`, `dual_shared`, `concat`, `cascaded`, `dual_shared_uni`. Thư mục nên thêm hậu tố `_traj` để không đè checkpoint cũ. Ước lượng 2–4 giờ/lần trên RTX 4080.
5. Cân nhắc chạy thêm 2 seed cho `dual_shared` và `concat` — chênh lệch giữa hai thiết kế hợp nhất chỉ 1,7 điểm, sát biên nhiễu, hội đồng đã từng nêu.
6. `test_all.py` sinh `results/` mới; **giữ nguyên bộ `results/` cũ** (bằng chứng cho số oracle, sẽ thành dòng "cận trên").

### Lưu ý khi có số mới

- **Mẫu số tập 2019 đổi 1198 → 1199.** Mẫu `UN19_1001_em_0` trước bị loại vì không có SRT, nay có quỹ đạo nên được nhận. Ảnh hưởng ExpRate cỡ 0,06 điểm nhưng số mẫu trong bảng phải sửa theo.
- Mẫu số 2014 vẫn là **985** (mẫu `505_em_51` có nhãn > 200 token, bị `data_iterator` loại — đúng như các lần chạy cũ, 77,06% = 759/985). Không phải lỗi mới.
- **Kỳ vọng:** tụt về vùng 55–62% ExpRate thay vì 74,17. Luận điểm vẫn đứng nếu cấu hình hợp nhất vượt **cả hai** nhánh đơn dưới cùng điều kiện đầu vào — phép so nội bộ, không phụ thuộc oracle.
- **Phần luận văn sẽ phải sửa:** 3.2–3.3 (mô tả đầu vào online + tiền xử lý + khối giảm chiều dài), Hình 3 + Hình 4 (nhãn "linearized SRT · token ids [b, l1]" → "point features [b, l₁, 8]", sửa SVG cả EN lẫn VI rồi render + duyệt trước khi nhúng), Bảng 2 (bỏ dòng vocab bộ mã hóa chuỗi 111; thêm bước resample 0,03, hệ số giảm 4×, `MAX_TRAJ_LEN`; cập nhật batch size nếu phải giảm), Bảng 3/4/5, Mục 4.4 (caveat oracle viết lại thành so sánh với hệ đã công bố), Abstract, Mục 1, Mục 5.1, Bảng 1, và bộ `bao_ve/`.

⚠️ **`.git/index.lock` đang kẹt** — Cowork không xóa được file trên mount. Trên Windows chạy trước khi làm gì với git:
```
del .git\index.lock
git reset
```

## 📌 TÓM TẮT BÀN GIAO (2026-07-02 — user chuyển sang Cowork, kết nối CÙNG thư mục C:\Users\Admin\lv\lvtn)
- **⚡ CẬP NHẬT 2026-07-13 — 15 VIỆC ĐỢT ĐÁNH GIÁ HỘI ĐỒNG (A1–A4, B1–B4, C1–C3, D1, E1–E5) ĐÃ ÁP CHÍNH THỨC vào cả 2 docx + xuất lại 2 PDF (xem khối ✅ 2026-07-13 ngay dưới). Refs: user ĐÃ tải đủ 7 PDF mới, kho khớp [1]–[41]. **✅ Việc F1 (nhúng lại Hình 3 EN sửa `cross rescoring`) 2026-07-13 + đồng bộ ký hiệu KIEN_TRUC 2026-07-13 + Việc G1/G2 (reword attention 3.7 hai bản; "crucial"→"essential" EN) 2026-07-14 ĐỀU ĐÃ ÁP — hiện KHÔNG còn việc mở.** Lịch sử (đã xong):** User yêu cầu đánh giá kiểu hội đồng → `thesis/DANH_GIA_HOI_DONG.md`. Từ đó soạn **15 việc sửa (A1–A4, B1–B4, C1–C3, D1, E1–E5), user đã DUYỆT TOÀN BỘ, CHƯA ÁP** — nguyên văn old→new nằm ở mục "⏳ CHỜ ÁP MỘT LƯỢT — ĐỢT SỬA THEO ĐÁNH GIÁ HỘI ĐỒNG" + "📊 GÓI CHƯƠNG 4" bên dưới. Chờ user ra lệnh "sửa chính thức" mới áp. **Thứ tự áp khuyến nghị:** backup 2 docx → các phép text (A/B/D/E; BỎ QUA A3#7 vì D1a hấp thụ; C2 áp TRƯỚC C1 để công thức (19) là bản mới) → bảng (B3 xóa dòng GETD tables[0]; E2b thêm 2 cột tables[2], font 9,5pt) → C1 đánh số 21 công thức → kiểm docPr id duy nhất → References + references_ieee.txt + INDEX.txt → xuất 2 PDF (mẹo soffice outdir rỗng + cp -f). User cần tải 4 PDF refs mới ([5][6][8][9], link trong Việc A4) và tùy chọn abstract [39]–[41]. Bằng chứng thống kê gói E: `results/analysis_significance_editdist.{py,txt}`.
- **File chuẩn DUY NHẤT ở `thesis\`**: 2 docx + 2 pdf (bản mới nhất 2026-07-02 18:49). KHÔNG sync Downloads (bản Downloads là snapshot cũ). `CLAUDE.md` ở gốc dự án (tạo 2026-07-02) tóm tắt cấu trúc + quy tắc + gotchas cho phiên mới.
- **Luận văn hoàn chỉnh Phần 1–5, EN+VI đồng bộ**: Phần 2 có mục 2.1–2.7 (2.4 = Đa phương thức); refs [1]–[38] đều đã xác minh; 6 hình vector EN/VI; Bảng 1: 12 dòng, ô GETD = "—" (số cũ chưa truy được nguồn); Bảng 2: 14 dòng (đã có dòng "Số lớp bộ mã hóa chuỗi = 3"); câu scheduler 3.10 đã trung tính hóa (PA-C, không sửa code).
- **Việc MỞ (tùy chọn, không gấp)**: (1) có PDF gốc [11] BPD / [17] GETD thì đối chiếu Bảng 1, điền lại số GETD thay "—"; (2) chạy thêm seed xác nhận chênh lệch fusion; (3) review style References.
- **Quy tắc user đặt ra (BẮT BUỘC)**: không bịa — mọi trích dẫn có nguồn xác minh; quy trình: soạn old→new → user duyệt → user ra lệnh "sửa chính thức" mới ghi file; mỗi phép sửa docx khớp ĐÚNG 1 vị trí (không lưu nếu lệch); không rõ thì hỏi lại user. **Bổ sung 2026-07-02:** (a) khi tra cứu/trích dẫn: ưu tiên kho `references/` có sẵn TRƯỚC, hết mới tìm internet (vẫn nguồn chính thống); (b) giọng văn học thuật, chuyên nghiệp, tự nhiên như người viết — hòa giọng hiện có của bài, TRÁNH khuôn mẫu AI (sáo ngữ "Ngoài ra/Hơn nữa/Tóm lại" dồn dập, liệt kê thay văn xuôi, cấu trúc lặp, EN tránh "delve/crucial/It is worth noting").

## ✅ ĐÃ ÁP CHÍNH THỨC 2026-07-13 (user lệnh "sửa chính thức") — ĐỢT ĐÁNH GIÁ HỘI ĐỒNG (A1–A4, B1–B4, C1–C3, D1, E1–E5)
Đã áp trọn 15 việc vào CẢ 2 docx trong `thesis/` + xuất lại 2 PDF. Quy trình: backup → dry-run (mọi anchor khớp ĐÚNG 1 vị trí, 0 fail cả 2 bản) → text (A/B/D/E, **C2 trước C1** để công thức (19) là bản mới) → bảng (B3, E2b) → C1 đánh số 21 công thức → References. Backup: `thesis/_backup_apply_20260713_093625/`. Script áp + log: `outputs/apply_edits.py` (hàm run-preserving thay run không gộp).
- **Text (0 gộp run):** EN 22 phép + VI 16 phép + B2 thay-cả-đoạn (2 bản) + E1/E4 chèn đoạn mới sau caption Bảng 5/Bảng 4 (2 bản, template = đoạn văn xuôi liền trước, đã strip bookmark). A3 chỉ EN, **BỎ A3#7** (D1a hấp thụ "Chapter 1"→"Section 1"). E5 chỉ VI ("chương này"→"phần này").
- **Bảng:** B3 xóa dòng GETD [17] khỏi Bảng 1 (tables[0]) → còn **11 hàng** (1 header + 10 dữ liệu). E2b thêm 2 cột "≤1 (%)"/"≤2 (%)" vào Bảng 3 (tables[2]) → **6 cột**, font 9,5pt khớp; số: 2014 84,26/87,21 · 2016 79,16/83,96 · 2019 80,13/85,31 · micro 81,02/85,41 (VI dấu phẩy).
- **C1:** 21 công thức (idx căn giữa) đánh số (1)–(21) bằng tab-stop **center 234pt + right 468pt** (alignment LEFT, text = "\t"+công thức+"\t(n)"). Soi PDF: công thức vẫn căn giữa, số hiệu sát lề phải; (19)=rescoring mới sau C2, (20)/(21) đúng.
- **References:** thay in-place [5]=Elman 1990, [6]=Hochreiter&Schmidhuber 1997, [8]=Pascanu 2013, [9]=Cho 2014; thêm [39]=CROHME2014, [40]=CROHME2016, [41]=CROHME2019 cuối danh mục — trong 2 docx (nháy cong, en-dash, không DOI theo style docx) + `references_ieee.txt` (nháy thẳng, gạch nối, KÈM DOI đã xác minh, trừ [8] PMLR không DOI). Header txt → [1]–[41]. `references/INDEX.txt`: đổi 4 tiêu đề + thêm 3 dòng [39–41], marker **TODO** = chưa có file.
- **✅ PDF references ĐÃ ĐỦ (user tải xong 2026-07-13):** cả 7 file mới ([5][6][8][9] full PDF + [39][40][41] PDF gốc IEEE) đã vào `references/`, nội dung ĐÃ XÁC MINH đúng bài (soi trang 1 từng file); bản cũ lưu ở `thesis/_backup_apply_20260713_093625/refs_old/`; `INDEX.txt` marker PDF đủ [1]–[41]. Kho references KHỚP HOÀN TOÀN danh mục.
- **🖼️ REVIEW HÌNH + BẢNG 2026-07-13 (phiên Cowork đánh giá hội đồng):** soi cả 6 hình EN (PNG chuẩn trong figures/) đối chiếu văn bản sau đợt sửa: Hình 1/2/4/5/6 khớp hoàn toàn (Hình 5 khớp cả mô tả 3.7 mới, Hình 6 khớp 5 cấu hình 4.6, Hình 4 khớp từng con số 3.4); bảng đã kiểm qua đợt áp. **1 LỖI TÌM THẤY — ✅ Việc F1 ĐÃ ÁP 2026-07-13:** Hình 3 EN ghi `cross rate-scoring` trong khi văn bản dùng `cross rescoring` (3.1 + 3.9 ×2; nguồn gốc lỗi = tên hàm `_cross_rate_score` trong decoder.py); bản VI đã đúng ("chấm điểm chéo") — KHÔNG đụng docx VI, KHÔNG đụng PDF VI.
  **Trạng thái đã làm xong (2026-07-13):** (a) `figures/arch_diagram_en.svg` đã sửa rate-scoring→rescoring; (b) đã render (cairosvg scale 3, nền trắng — hình không có ký tự Hy Lạp nên cairosvg an toàn) và **PNG chuẩn MỚI đã nằm sẵn tại `thesis/figures/arch_diagram_en.png`** (2280×2472, đúng bằng kích thước bản cũ); bản cũ backup ở `thesis/_backup_apply_20260713_093625/arch_diagram_en_old.png`.
  **✅ ĐÃ THỰC HIỆN XONG 2026-07-13 (phiên Cowork):** backup docx EN → `thesis/_backup_f1_20260713_233431/`; nhúng PNG mới vào Hình 3 = inline_shapes[2] (chỉ thay blob: md5 a6e46343…→f3160765…, giữ nguyên rId/kích thước hiển thị/docPr); verify: inline_shapes=6, docPr id {1..6} duy nhất, text docx không đổi; xuất lại CHỈ PDF EN (47 trang, 6 hình). Soi PDF trang 18: ô cuối Hình 3 nay ghi đúng "cross rescoring → LaTeX string". PDF VI + docx VI KHÔNG đụng. Các bước gốc (đã hoàn tất):
  **Các bước tham chiếu:**
  1. Backup `thesis/Multi-modal_HMER_Sections1-5_full.docx` vào thư mục backup mới (`thesis/_backup_f1_<timestamp>/`).
  2. Nhúng: mở docx EN bằng python-docx, xác định **Hình 3 = `inline_shapes[2]`** (mapping cố định inline_shapes[0..5] = Hình 1..6), lấy rId của blip → **chỉ thay bytes của image part** bằng nội dung `thesis/figures/arch_diagram_en.png` (image_part._blob / write bytes). TUYỆT ĐỐI không tạo shape mới, không đổi kích thước hiển thị, **không đụng `wp:docPr` id** (id trùng làm xuất PDF treo).
  3. Verify sau lưu: `len(inline_shapes)==6`; blob của inline_shapes[2] == bytes PNG mới (so md5); docPr id vẫn là {1,2,3,4,5,6} không trùng; text docx không đổi (đếm nhanh vài anchor bất kỳ).
  4. Xuất lại **CHỈ PDF EN** vào `thesis/` (gotchas: kill soffice trước; `soffice --headless --convert-to pdf --outdir <THƯ MỤC RỖNG>` + UserInstallation profile mới, rồi `cp -f` đè về `thesis/` — rm/rename trên mount bị chặn; mỗi lần 1 file). Soi PDF: trang chứa Hình 3 phải đọc được "cross rescoring → LaTeX string"; đủ 47 trang (±1 nếu phân trang lệch nhẹ do LibreOffice); 6 hình nguyên.
  5. Ghi kết quả (ngày, verify) vào file này: chuyển F1 sang ✅ ĐÃ ÁP.
- **✅ Việc G1 + G2 — 2 sửa theo báo cáo đạo văn/AI (user lệnh "SỬA CHÍNH THỨC" 2026-07-14, ĐÃ ÁP):** backup `_backup_g1g2_20260714_125000/`; dry-run mỗi anchor ×1 (G1-EN, G2-EN, G1-VI), thay giữ run; verify old mất/new có ở cả 2 bản; docPr 1–6 duy nhất, 6 hình nguyên; xuất lại 2 PDF (EN 47 tr, VI 45 tr; soi EN text "produces the output"/"input are essential", VI render tr.25 câu 3.7 mới). Nguyên văn old→new (giữ làm nhật ký):
  **G1 — reword câu định nghĩa attention 3.7** (phá cụm 11 từ trùng [14]; cụm còn lại dài nhất sau sửa = 7 từ "queries Q keys K and values V" < ngưỡng 8):
  · **EN old:** `Scaled dot-product attention maps a set of queries Q, keys K, and values V to an output:` → **EN new:** `Given queries Q, keys K, and values V, scaled dot-product attention produces the output:`
  · **VI old:** `Chú ý tích vô hướng có tỉ lệ ánh xạ một tập truy vấn Q, khóa K và giá trị V thành đầu ra:` → **VI new:** `Cho truy vấn Q, khóa K và giá trị V, chú ý tích vô hướng có tỉ lệ sinh đầu ra:`
  **G2 — thay "crucial" ×1 (Mục 1, EN-only; VI "then chốt" giữ nguyên vì vẫn khớp nghĩa):**
  · **EN old:** `in which the order and context of the input are crucial.` → **EN new:** `in which the order and context of the input are essential.`
  (Đã áp 2026-07-14: 2 docx (G1) + docx EN (G2), giữ run, xuất lại 2 PDF.)
- **🎤 BỘ TÀI LIỆU BẢO VỆ 2026-07-15 (user yêu cầu; v2 theo góp ý user):** thư mục `bao_ve/`: **`SLIDE_BAO_VE.pptx`** (20 slide 16:9: 17 chính + 3 phụ lục Q&A [A DenseNet, B kết quả từng năm + giải thích khoảng cách BTTR, C định vị related-works kèm caveat oracle]; chữ slide tiếng Việt nhưng **hình + biểu đồ dùng bản EN** trong thesis/figures/ theo yêu cầu user; **trang bìa nền trắng, tiêu đề EN cỡ lớn 41pt**, phụ đề VI; bảng native + bar chart EN; speaker notes từng slide; validate PASS + QA 20/20 render; **v3 2026-07-15: tăng cỡ chữ nội dung theo yêu cầu user** — body 12–14 → 13,5–16pt, bullet 15–16,5pt, bảng 13–14,5pt kèm nới rowH/thẻ, QA lại 20/20 không tràn; script gốc: /tmp/make_deck.js — /tmp không bền, muốn sửa deck thì đọc lịch sử phiên hoặc unpack pptx) và **`KICH_BAN_THUYET_TRINH.docx`** (kịch bản nói từng slide có căn giờ ~17', bảng phân bổ thời gian, 5 nguyên tắc Q&A). ⚠️ User cần điền vào slide 1: tên GVHD, [Trường/Khoa], [Tháng, 2026]. Mọi số liệu khớp bản luận văn hiện hành. Lưu ý nhỏ: chữ "LUẬN VĂN THẠC SĨ" trên bìa có charSpacing — LibreOffice render dấu hơi mờ nhưng PowerPoint thật (Calibri) hiển thị đúng.
- **🏷️ DỌN METADATA DOCX 2026-07-14 (user yêu cầu, chỉ sửa-cho-đúng, KHÔNG ngụy tạo):** thay "Un-named" (rác python-docx) → author/lastModifiedBy = "Nguyen Tri" cả 2 bản; điền title = tên đề tài EN/VI; EN modified 19/06 (stale, sai) → 14/07 12:50 (thời điểm áp G1/G2 thật); GIỮ NGUYÊN created 19/06, VI modified/revision/TotalTime (đúng sẵn từ lần user lưu Word). Từ chối có chủ đích việc chỉnh created/TotalTime làm đẹp lịch sử (đã giải thích cho user — ngụy tạo hồ sơ). Cách làm: phẫu thuật zip chỉ thay docProps/core.xml, document.xml verify nguyên vẹn byte-một cả 2 bản; PDF không cần xuất lại. Backup: `thesis/_backup_meta_20260714/`. ⚠️ Còn 1 file rác `Multi-modal_HMER_Sections1-5_full.docx.tmp_new` trong thesis/ (staging kẹt do mount chặn xóa; user từ chối cấp quyền xóa — vô hại, user có thể xóa tay).
- **🧪 KIỂM ĐẠO VĂN + DẤU HIỆU AI 2026-07-14 (user yêu cầu, trọng tâm bản EN):** 3 lớp — (1) so 8-gram thân bài (14.272 từ) với CẢ 41 PDF trong references/: chỉ 15 span trùng ≥8 từ (<1% độ phủ), toàn bộ là số liệu trích dẫn/tên riêng/boilerplate; duy nhất 1 chỗ đáng để mắt = câu định nghĩa attention trùng 11 từ với [14] (diễn đạt chuẩn ngành gốc Vaswani — tùy chọn reword); (2) dò web 6 câu đặc trưng: 0/6 trùng; (3) stylometry: 0 delve/pivotal/landscape/…, std độ dài câu 15,2 (burstiness cao kiểu người viết), không chuỗi Moreover/Furthermore. **KHÔNG phát hiện đạo văn; dấu hiệu AI rất thấp.** Báo cáo đầy đủ: **`thesis/KIEM_TRA_DAOVAN_AI_20260714.md`**. Việc bắt buộc còn lại thuộc user: chạy Turnitin/iThenticate chính thức của trường + nắm quy định trường về công cụ hỗ trợ viết.
- **🔍 AUDIT ĐỘC LẬP PHẦN 3 ↔ CODE 2026-07-13 (user yêu cầu):** đọc toàn bộ `bttr/` + `custom_train.py` + vocab + hparams.yaml seed-7, kiểm 9 cụm khẳng định của Phần 3 → **9/9 PASS, không có sai lệch văn bản↔code**. Biên bản đầy đủ kèm dẫn chứng file/dòng: **`thesis/AUDIT_PHAN3_CODE_20260713.md`**. Điểm đáng giá: xác nhận chung-trọng-số 2 cross-attention ở mức code (`_mha_block` dùng chung) + bằng đếm tham số checkpoint (3,35M chỉ khớp khi chung); công thức rescoring (19) mới khớp từng dòng `_cross_rate_score`; tìm ra nguồn gốc lỗi Hình 3 = tên hàm `_cross_rate_score`. 3 ghi chú bên lề (scheduler mode="max" — đã trung tính hóa từ 02/07, không phải lỗi văn bản; encoder post-norm nhưng bài chỉ claim pre-norm cho decoder — chính xác). Không mâu thuẫn với biên bản 2026-07-04.
- **✔️ KIỂM TRA ĐỘC LẬP 2026-07-13 (phiên Cowork đánh giá hội đồng):** rà lại toàn bộ 15 việc bằng script riêng trên 2 docx sau áp — 60+ phép kiểm PASS hết: mọi old mất/new có đúng nguyên văn cả 2 bản; A3 sạch "paper/chapter" (0 sót); chùm "[5], [6], [9]" cũ hết, câu mới gắn ref từng thực thể đúng; Bảng 1 hết GETD, Bảng 3 đủ 6 cột đúng số; công thức (1)–(21) đủ và đúng thứ tự ở cả 2 bản, số hiệu render ra PDF (soi (1)/(8)/(19)/(21)); C2/C3/D1/E1–E5 khớp từng chữ; docPr 1–6 duy nhất, 6 hình nguyên. Lưu ý nhỏ chấp nhận được: entry References trong docx dùng nháy cong + en-dash (đúng house style docx), `references_ieee.txt` kèm DOI (nhiều thông tin hơn bản duyệt, không sai lệch).
- **Verify PASS:** mọi anchor old (không phải phép nối-thêm) biến mất, mọi new có mặt ở cả 2 bản; docPr id **1–6 duy nhất** (không đụng 6 hình), inline_shapes=6; Bảng 1=11 hàng, Bảng 3=6 cột; công thức (1)–(21) đủ. PDF: **EN 47 trang, VI 45 trang**, tiếng Việt đủ dấu, 6 hình nguyên (soi trực tiếp trang 32/37 EN + 31/36 VI).
- **User đã thả 7 PDF references mới vào `references/` (2026-07-13):** 5/6/8/9/39/40/41.pdf — đã kiểm nội dung khớp đúng paper (Elman/LSTM/Pascanu/Cho + 3 CROHME). INDEX.txt marker → PDF; kho đủ **36 PDF + 5 ABS, thiếu 0**. File cũ đã xoá, backup ở `_backup_apply_20260713_093625/refs_cu_5689/`.
- **🐞 SỬA BORDER BẢNG 2 (2026-07-13, user báo):** 3 dòng thêm ở phiên 2026-07-04 (Batching threshold / Max samples per batch / Vocabulary sizes = tables[1] r11–13) **thiếu `w:tcBorders` riêng** nên kế thừa border mặc định (đậm/đen), lệch với các dòng khác dùng xám `BBBBBB`. Đã gán tcBorders xám (clone từ ô r1) cho cả 6 ô (3 dòng × 2 cột) ở CẢ 2 docx → xuất lại 2 PDF. Verify: 0 dòng thiếu border; soi PDF Bảng 2 kẻ xám đồng đều. Backup trước sửa: `_backup_apply_20260713_093625/{EN,VI}_before_bang2border.docx`.

## 🔧 REVIEW PHẦN 3 vs CODE (2026-07-04) — TEXT ĐÃ ÁP DỤNG, CÒN HÌNH + PDF

> **✅ ĐÃ ÁP DỤNG "sửa chính thức" 2026-07-04 (text):** Toàn bộ gói ký hiệu (E1–E14 EN / V1–V14 VI, BỎ E4/V4) + đoạn "Ký hiệu." (V3, in đậm, nằm đúng giữa đoạn <eos> và heading 3.3) + 3.3 (E15–E17/V15–V17) + R3.4, R3.6, R3.7(A+B+C), R3.9, R3.10, R3.11, R4.2(A+B) + R-Bảng2 đã ghi vào **cả 2 docx** trong thesis\. Cách làm: script python-docx dry-run đếm mọi anchor==1 rồi mới lưu (24 EN + 37 VI body anchor đều ==1; bảng: heads/dropout/maxlen mỗi cái ==1). Verify sau khi lưu: old mất/new có ở cả 2 bản; Bảng 2 14→17 dòng (Dropout tách 0,3/0,2; heads→n_h; +3 dòng ngưỡng gom lô/cap 32/vocab 111·113); docPr vẫn duy nhất 1–6 (PDF không treo). Backup 2 docx gốc ở thesis\_backup_20260704\.
> **✅ HÌNH ĐÃ NHÚNG 2026-07-04:** 4 SVG (arch_en/vi = Hình 3, densenet_en/vi = Hình 4) đã sửa + render (cairosvg, scale 3, nền trắng) + user duyệt + **nhúng vào 2 docx** (thay blob inline_shapes[2],[3], giữ docPr 1–6; verify blob==png, 6 hình). Thay đổi hình: [b,t,d]→[b,m,d] (cả "Flatten h·w→t"→"h·w→m" ở densenet); arch [2b,l2]→[2b,L] & [2b,l2,|V_dec|]→[2b,L,|V_dec|]; đồng bộ thuật ngữ VI (output→đầu ra, tầng→lớp, "nối output với input"→"nối đầu ra với đầu vào"); sửa tràn khung densenet_en (font 12.5→11); **nhãn online**: "symbol-relation seq (SRT)" → tiêu đề "Online symbol-relation sequence" + phụ "linearized SRT · token ids [b, l1]" (VI: "Chuỗi quan hệ ký hiệu online" + "SRT tuyến tính hóa · …"). PNG chuẩn trong figures/ đã cập nhật. Backup docx sau-text: thesis/_backup_20260704/*_after_text.docx. Ghi chú kiến thức "linearized SRT" cho user: ghi_chu/linearized_SRT.md.
> **✅ PDF ĐÃ XUẤT 2026-07-04 (user đổi ý, yêu cầu xuất luôn):** dùng **LibreOffice** (Word COM không có trong Cowork). 2 PDF trong thesis\: EN 46 trang, VI 43 trang; tiếng Việt render đủ dấu; 4 hình + Bảng 2 (17 dòng) hiển thị đúng. ⚠️ Caveat LibreOffice: KHÔNG sinh bookmark theo heading như Word COM, phân trang có thể lệch nhẹ so với Word — nếu cần bản chuẩn bookmark thì xuất lại bằng Word COM trên Windows. Backup PDF cũ: thesis\_backup_20260704\{EN,VI}_old.pdf.
> **🐞 2 LỖI PHÁT HIỆN & SỬA khi soi PDF (2026-07-04):** (1) **3.1-a bị SÓT** khỏi script áp text lần đầu (bản VI vẫn "Hai mũi tên **gộp** đi vào cạnh trên") → đã áp bổ sung: "Hai mũi tên đi vào cạnh trên" (verify ×1). (2) **Bảng 2: 3 dòng mới bị font mặc định (to hơn)** do dùng `t.add_row()` (không kế thừa định dạng) → đã set lại **9,5pt** cho khớp các dòng cũ, cả EN+VI.
> **🔧 MẸO XUẤT PDF trong Cowork (mount chặn xóa/rename):** soffice ghi PDF kiểu temp+rename → fail `Io Abort Code:27` khi file đích ĐÃ tồn tại (kể cả ở outputs). Cách chạy được: `soffice --headless --convert-to pdf --outdir <THƯ MỤC MỚI/RỖNG>` rồi `cp -f` đè vào thesis\ (cp ghi-đè-tại-chỗ được, `rm`/rename thì "Operation not permitted"). Kill soffice + UserInstallation profile mới mỗi lần.
> → **Phần 3 HOÀN TẤT trọn vẹn:** text + hình + Bảng 2 + 2 PDF. Sẵn sàng chuyển Phần 4.

> ## ✅ ĐÃ ÁP CHÍNH THỨC 2026-07-11 (user lệnh "sửa chính thức")
> Đã áp **85 phép prose** (41 việc EN+VI, dry-run mỗi anchor khớp đúng 1 vị trí, 0 gộp run → giữ định dạng) + **nhúng Hình 1** mới (2003×878, có π; blob rId7, docPr id 1–6 không trùng) + **11 sửa TLTK** vào `references_ieee.txt` và References của 2 docx. Xuất lại 2 PDF vào `thesis/` (EN 46 trang, VI 44 trang, LibreOffice). Backup trước khi sửa: `_backup_apply_20260711_020754/`. Kiểm chứng: mọi old biến mất, new có mặt; "Formally" đã hết; TLTK verify khớp. Hai mục ⏳ bên dưới GIỮ LÀM NHẬT KÝ (đã hoàn tất).

## 📝 ĐÁNH GIÁ PHẢN BIỆN MÔ PHỎNG (2026-07-12)
> User yêu cầu đóng vai GS hội đồng, đánh giá toàn luận văn theo 5 tiêu chí học thuật → đã tạo **`thesis/DANH_GIA_HOI_DONG.md`** (đánh giá từng chương + tổng, dựa trên bản EN 2026-07-11, số liệu đối chiếu results/). Kết luận: đạt bảo vệ sau minor revisions. **Các phát hiện chính chưa xử lý (CHƯA áp sửa gì vào docx):** (1) 5.1 trả lời "yes" cho cả vế "vượt multi-modal RNN" — không kiểm chứng được dưới oracle, mâu thuẫn 4.4; (2) EN còn 6 lần "this paper/the paper" (5 đoạn: Mục 1 ×2, 2.1 ×2, 3.13) + "Chapter 1" (5.1) + "this chapter" (4.8) — VI đã nhất quán "luận văn"; (3) offline-only ablation 47,61 vs BTTR công bố 53,96 chưa được giải thích trong 4.6; (4) thiếu McNemar/bootstrap CI (chênh fusion 1,7–2,4đ ≈ biên nhiễu ±1,5đ micro) + thiếu ExpRate ≤1/≤2 lỗi + 4.7 chưa định lượng 3 loại lỗi — đều tính được từ results/, không cần huấn luyện lại; (5) refs nền tảng [5][6][8][9] ngoại vi, benchmark 2014/16/19 chỉ trích [29] (bài 2023); (6) công thức Phần 3 chưa đánh số. Mọi sửa chữa theo quy trình old→new + duyệt như thường lệ.

## ✅ ĐÃ ÁP 2026-07-13 (giữ làm nhật ký old→new) — ĐỢT SỬA THEO ĐÁNH GIÁ HỘI ĐỒNG (soạn 2026-07-12, áp một lượt 2026-07-13)
> User quyết định: duyệt từng phần theo `DANH_GIA_HOI_DONG.md`, ghi lại đây, **CHƯA áp** — áp một lượt khi duyệt xong (chờ lệnh "sửa chính thức"), sau đó xuất lại 2 PDF.
**Việc A1 — Abstract, câu ablation (user DUYỆT 2026-07-12).** Làm rõ 8,5đ (trục fusion) và 10,7đ (trục hướng huấn luyện) là hai phép so độc lập, không cộng dồn. Anchor đã verify ×1 ở cả 2 docx (2026-07-12).
- **EN old:** `and bidirectional training adds a further 10.7 points`
- **EN new:** `and, independently, bidirectional training accounts for a 10.7-point gain over a unidirectional counterpart`
- **VI old:** `và huấn luyện hai chiều bổ sung thêm khoảng 10,7 điểm`
- **VI new:** `và, xét riêng, huấn luyện hai chiều mang lại mức tăng khoảng 10,7 điểm so với biến thể một chiều`
**Việc A2 — Câu hỏi nghiên cứu Mục 1 (user DUYỆT 2026-07-12).** Thu hẹp về vế kiểm chứng được; vế "vượt multi-modal RNN" → khoảng trống. Anchor ×1 cả 2 bản (2026-07-12).
- **EN old:** `to surpass both single-modality recognizers and RNN-based multi-modal recognizers`
- **EN new:** `to surpass both of its single-modality counterparts under a controlled input setting, thereby addressing the gap left by RNN-based multi-modal recognizers`
- **VI old:** `để vượt cả các bộ nhận dạng đơn-phương-thức lẫn các bộ nhận dạng đa phương thức dựa trên RNN hay không`
- **VI new:** `để vượt cả hai biến thể đơn-phương-thức của chính nó trong một thiết lập đầu vào có kiểm soát hay không — qua đó giải quyết khoảng trống mà các bộ nhận dạng đa phương thức dựa trên RNN để lại`
**Việc A3 — Gói thuật ngữ EN-only, 8 phép (user DUYỆT 2026-07-12).** VI đã nhất quán. Mỗi anchor ×1 (2026-07-12): (1) `In this paper we focus on a specialized subdomain`→this thesis; (2) `The remainder of this paper is organized as follows`→this thesis; (3) `concludes the paper and discusses future work`→the thesis; (4) `Throughout this paper we evaluate methods`→this thesis; (5) `The method proposed in this paper follows the string-decoding paradigm`→this thesis; (6) `the central contribution of this paper`→this thesis; (7) `The question raised in Chapter 1`→Section 1; (8) `The experiments in this chapter`→this section. Sau đó EN sạch hoàn toàn paper/chapter (đã đếm 5+1+1+1). **Cách áp cho model khác:** trong mỗi anchor, thế ĐÚNG cụm con: "this paper"→"this thesis", "the paper"→"the thesis", "Chapter 1"→"Section 1", "this chapter"→"this section"; phần còn lại của anchor giữ nguyên từng ký tự.
**Việc A4 — Thay in-place refs [5][6][8][9] (user DUYỆT trọn gói 2026-07-12).** Cả 4 ref chỉ được trích ở đúng 1 đoạn (đoạn RNN Mục 1). Slot mới cho thứ tự trích tăng dần: **[5]=Elman 1990 (RNN), [6]=Hochreiter&Schmidhuber 1997 (LSTM), [8]=Pascanu et al. 2013 (vanishing/exploding), [9]=Cho et al. 2014 (GRU)**. ĐÃ XÁC MINH: [5] Crossref 10.1207/s15516709cog1402_1 (Cogn. Sci. 14(2):179–211); [6] Crossref 10.1162/neco.1997.9.8.1735 (Neural Comput. 9(8):1735–1780); [8] PMLR v28(3):1310–1318; [9] Crossref 10.3115/v1/D14-1179 (EMNLP 2014, pp. 1724–1734).
- Câu văn (gắn ref theo từng thực thể): **EN old:** `Recurrent Neural Networks (RNNs) and their variants—including Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) [5], [6], [9]—are well suited` → **EN new:** `Recurrent Neural Networks (RNNs) [5] and their variants—including Long Short-Term Memory (LSTM) networks [6] and Gated Recurrent Units (GRUs) [9]—are well suited`; **VI old:** `Mạng nơ-ron hồi quy (RNN) và các biến thể — gồm bộ nhớ ngắn-dài hạn (LSTM) và đơn vị hồi tiếp có cổng (GRU) [5], [6], [9] — rất phù hợp` → **VI new:** `Mạng nơ-ron hồi quy (RNN) [5] và các biến thể — gồm bộ nhớ ngắn-dài hạn (LSTM) [6] và đơn vị hồi tiếp có cổng (GRU) [9] — rất phù hợp`. Câu gradient [8] giữ nguyên văn bản.
- 4 entry References mới (áp cho cả 2 docx + references_ieee.txt): [5] `J. L. Elman, "Finding structure in time," Cognitive Science, vol. 14, no. 2, pp. 179–211, 1990.`; [6] `S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.`; [8] `R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training recurrent neural networks," in Proc. ICML, PMLR, vol. 28, no. 3, 2013, pp. 1310–1318.`; [9] `K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, "Learning phrase representations using RNN encoder–decoder for statistical machine translation," in Proc. EMNLP, 2014, pp. 1724–1734.`
- Khi áp: backup 5.pdf/[6]_ABSTRACT.txt/8.pdf/9.pdf cũ → thay PDF mới (user tải: [5] doi.org/10.1207/s15516709cog1402_1; [6] bioinf.jku.at/publications/older/2604.pdf hoặc MIT Press; [8] proceedings.mlr.press/v28/pascanu13.pdf; [9] aclanthology.org/D14-1179.pdf — Cowork không tải được file nhị phân) → cập nhật INDEX.txt 4 dòng tương ứng.
**Việc B1 — 3 mệnh đề định vị cuối đoạn ở 2.2 (user DUYỆT 2026-07-12, bản B1b đã-chỉnh; anchor ×1 cả 2 bản).** Chèn nối tiếp cuối đoạn (giữ nguyên câu cũ, thêm câu mới sau dấu chấm):
- B1a cuối đoạn BPD — **EN sau** `and is especially effective for complex and deeply nested expressions.` **thêm:** ` For our purposes, BPD matters less for its parallel decoding than as evidence that Transformer decoders cope well with deeply nested structure—the property our fusion design relies on.` / **VI sau** `đặc biệt hiệu quả với các biểu thức phức tạp và lồng sâu.` **thêm:** ` Với luận văn này, ý nghĩa của BPD không nằm ở việc giải mã song song mà ở chỗ nó cho thấy bộ giải mã Transformer xử lý tốt cấu trúc lồng sâu — đúng tính chất mà thiết kế hợp nhất của chúng tôi dựa vào.`
- B1b cuối đoạn Truong — **EN sau** `reaching 64.60% and 66.08% on CROHME 2014 and 2016 and 58.72% on the CROHME 2019 test set.` **thêm:** ` Both augmentation lines act on the training data rather than on the model itself; their gains are therefore orthogonal to the architectural direction pursued in this thesis and could be combined with it.` / **VI sau** `đạt 64,60% và 66,08% trên CROHME 2014 và 2016 và 58,72% trên tập kiểm thử CROHME 2019.` **thêm:** ` Cả hai hướng sinh dữ liệu này tác động vào dữ liệu huấn luyện thay vì bản thân mô hình, vì vậy phần cải thiện của chúng độc lập với hướng kiến trúc mà luận văn theo đuổi và có thể kết hợp cùng nó.` (Đã sửa 2026-07-12 so với nháp đầu để tránh lặp ý "target/shortage of training data" với câu mở đoạn.)
- B1c cuối đoạn VLPG — **EN sau** `improving accuracy from 53.57%/54.40%/56.99% to 60.41%/60.51%/62.34% on the CROHME sets.` **thêm:** ` Viewed from the perspective of this thesis, GETD and VLPG enrich the encoder side of the pipeline, while leaving open the question of how a decoder should combine complementary sources of information—precisely the question our work addresses.` / **VI sau** `nâng độ chính xác từ 53,57%/54,40%/56,99% lên 60,41%/60,51%/62,34% trên các tập CROHME.` **thêm:** ` Nhìn từ góc độ của luận văn, GETD và VLPG làm giàu phía bộ mã hóa của pipeline, trong khi vẫn để ngỏ câu hỏi bộ giải mã nên kết hợp các nguồn thông tin bổ trợ ra sao — đúng câu hỏi mà công trình này giải quyết.`
**Việc B2 — Rút gọn đoạn Ung [20] ở 2.3 (user DUYỆT 2026-07-12).** Thay CẢ đoạn (anchor = đoạn bắt đầu `Beyond recognition itself,` / `Ngoài bản thân việc nhận dạng,` — mỗi bản ×1).
- **EN new (cả đoạn):** `Beyond recognition itself, online representations enable downstream applications: Ung et al. [20] address semi-automatic grading by clustering online handwritten expressions with a bag-of-features model that combines six levels of visual, symbolic, and structural features, reaching a purity of up to 0.99 when the number of clusters matches the number of true classes. The example illustrates how the structural richness of online data benefits tasks beyond plain transcription.`
- **VI new (cả đoạn):** `Ngoài bản thân việc nhận dạng, biểu diễn online còn mở ra các ứng dụng phía sau: Ung và cộng sự [20] giải quyết bài toán chấm điểm bán-tự-động bằng cách gom cụm các biểu thức viết tay online với mô hình túi đặc trưng kết hợp sáu cấp đặc trưng từ thị giác đến ký hiệu và cấu trúc, đạt độ tinh khiết tới 0,99 khi số cụm khớp số lớp thực. Ví dụ này minh họa cho việc dữ liệu online giàu cấu trúc mang lại lợi ích cho cả các nhiệm vụ ngoài phiên chép.`
**Việc B3 — Dòng GETD Bảng 1 (user CHỐT 2026-07-12: BỎ DÒNG).** Kho chỉ có [17]_ABSTRACT.txt (Elsevier paywall), số không truy được nguồn → khi áp: xóa dòng `GETD [17] | Offline | GNN graph encoder | Transformer | —` khỏi Bảng 1 (tables[0]) ở CẢ 2 docx; GETD vẫn được bàn trong văn bản 2.2 (đoạn GETD/VLPG giữ nguyên + câu định vị B1c). Đóng luôn mục mở cũ "đối chiếu GETD Bảng 1" theo hướng này (trừ khi sau này có PDF gốc).
**Việc B4 — Thêm refs CROHME từng năm [39][40][41] (user DUYỆT 2026-07-12).** ĐÃ XÁC MINH Crossref: [39] 10.1109/ICFHR.2014.138 (tr. 791–796); [40] 10.1109/ICFHR.2016.0116 (tr. 607–612); [41] 10.1109/ICDAR.2019.00247 (tr. 1533–1538). Giữ [29] làm trích dẫn benchmark chung; thêm 3 entry cuối danh mục (IEEE paywall → tạo [39]/[40]/[41]_ABSTRACT.txt theo quy ước kho, user có quyền IEEE thì bổ sung PDF).
- Câu 4.1 — **EN old:** `evaluated independently on the test sets of CROHME 2014, 2016, and 2019, which contain` → **new:** `evaluated independently on the test sets of CROHME 2014 [39], 2016 [40], and 2019 [41], which contain`; **VI old:** `đánh giá độc lập trên các tập kiểm thử CROHME 2014, 2016 và 2019, lần lượt chứa` → **new:** `đánh giá độc lập trên các tập kiểm thử CROHME 2014 [39], 2016 [40] và 2019 [41], lần lượt chứa`
- Entry mới: [39] `H. Mouchère, C. Viard-Gaudin, R. Zanibbi, and U. Garain, "ICFHR 2014 competition on recognition of on-line handwritten mathematical expressions (CROHME 2014)," in Proc. ICFHR, 2014, pp. 791–796.`; [40] `H. Mouchère, C. Viard-Gaudin, R. Zanibbi, and U. Garain, "ICFHR2016 CROHME: Competition on recognition of online handwritten mathematical expressions," in Proc. ICFHR, 2016, pp. 607–612.`; [41] `M. Mahdavi, R. Zanibbi, H. Mouchère, C. Viard-Gaudin, and U. Garain, "ICDAR 2019 CROHME + TFD: Competition on recognition of handwritten mathematical expressions and typeset formula detection," in Proc. ICDAR, 2019, pp. 1533–1538.`
**Việc C1 — Đánh số công thức (1)–(21) (user DUYỆT 2026-07-12).** Đã rà: cả 2 docx có ĐÚNG 21 paragraph công thức, đều căn giữa (CENTER); KHÔNG có tham chiếu "formula above" nào trong văn bản → phép sửa thuần cộng thêm. Thứ tự: (1) argmax P(y|X_off,X_on;θ), (2) phân rã tự hồi quy Π, (3) x_ℓ=H_ℓ([...]), (4) n_ℓ=n_0+ℓ·k, (5) PE sin, (6) PE cos, (7) cumsum ȳ/x̄, (8) Attn, (9) MultiHead, (10) head_i, (11)–(14) bốn phép decoder layer, (15) softmax đầu ra, (16) ℒ_dir, (17) ℒ, (18) score beam, (19) rescoring (bản mới C2), (20) chuẩn hóa độ dài, (21) ExpRate (Mục 4.3). Cơ chế: đổi alignment→trái, thêm center-tab tại giữa cột + right-tab tại lề phải, text = "\t"+công thức+"\t(n)" — công thức vẫn giữa trang, số hiệu sát lề phải; áp giống hệt 2 bản, soi PDF sau khi áp. **Key nhận diện 21 paragraph (mỗi key khớp đúng 1 paragraph, giống nhau ở cả 2 docx):** (1)`ŷ = argmax_{y}  P(` (2)`= Π_{t=1..L}` (3)`x_ℓ = H_ℓ` (4)`n_ℓ = n_0` (5)`PE(pos, 2i) =` (6)`PE(pos, 2i+1)` (7)`cumsum_h` (8)`Attn(Q, K, V)` (9)`MultiHead(Q,K,V)` (10)`head_i =` (11)`x_q = x + SelfAttn` (12)`CrossAttn( LN₂(x_q) , E_off` (13)`CrossAttn( LN₂(x_q) , E_on` (14)`x = x + FFN` (15)`softmax( W_o` (16)`ℒ_dir` (17)`ℒ = −` (18)`score( y_{1:t}` (19)`score(y)  ←` (SAU khi áp C2; bản cũ là `score  ←`) (20)`argmax_{y}  score` (21)`ExpRate =`.
**Việc C2 — Viết lại công thức rescoring 3.9 (user DUYỆT 2026-07-12; anchor ×1 cả 2 bản).**
- **EN lead-in old:** `Concretely, the additional score of a hypothesis is the negative cross-entropy of applying that hypothesis to the reverse direction:` → **new:** `Concretely, the hypothesis y is reversed into y′ (with the roles of <sos> and <eos> swapped), and the log-likelihood of y′ under the opposite-direction model, denoted p_rev, is added to the score:`
- **Công thức old (2 bản giống nhau):** `score  ←  score  −  Σ_i  CE( ŷ_i , y_i )` → **new:** `score(y)  ←  score(y)  +  Σ_{t=1..|y′|}  log p_rev( y′_t | y′_<t , E_off , E_on )`
- **VI lead-in old:** `Cụ thể, điểm bổ sung của một giả thuyết là âm của entropy chéo khi áp giả thuyết đó vào chiều đối nghịch:` → **new:** `Cụ thể, giả thuyết y được đảo thành y′ (hoán đổi vai trò <sos> và <eos>), rồi log-xác suất của y′ theo mô hình chiều ngược, ký hiệu p_rev, được cộng vào điểm số:` ("log-xác suất" khớp cách dùng sẵn có ở 3.9)
- Tương đương toán học với bản cũ (−ΣCE = Σ log p_rev), chỉ chặt ký hiệu hơn; đoạn "This cross rescoring…" phía sau vẫn khớp, không sửa.
**Việc C3 — Thêm quy mô mô hình + thời gian huấn luyện vào 3.10 (user DUYỆT 2026-07-12, bản RTX 4080).** Nguồn: đọc shape từ checkpoint seed-7 `lightning_logs/abl_dual_shared/.../epoch=39-step=45560-val_loss=0.1566.ckpt` (parse pickle, không cần torch): TỔNG 9.044.261 tham số ≈ 9,04M (decoder 3,35M; encoder ảnh 3,17M; encoder chuỗi 2,53M). Thời gian "chưa đầy hai giờ": user XÁC NHẬN đúng 2026-07-12; GPU = **RTX 4080** (user cung cấp).
- Chèn TRƯỚC câu **EN** `The main hyper-parameters are listed in Table 2.` (×1): `The full model comprises approximately 9.0 million trainable parameters—3.35 M in the multi-modal decoder, 3.17 M in the image encoder, and 2.53 M in the sequence encoder—and a complete 50-epoch training run finishes in under two hours on a single NVIDIA RTX 4080 GPU. `
- Chèn TRƯỚC câu **VI** `Các siêu tham số chính được liệt kê trong Bảng 2.` (×1): `Toàn bộ mô hình có xấp xỉ 9,0 triệu tham số huấn luyện được — 3,35 triệu ở bộ giải mã đa phương thức, 3,17 triệu ở bộ mã hóa ảnh và 2,53 triệu ở bộ mã hóa chuỗi — và một lần huấn luyện đủ 50 epoch hoàn tất trong chưa đầy hai giờ trên một GPU NVIDIA RTX 4080. `
**Việc D1 — Viết lại đoạn trả lời câu hỏi nghiên cứu ở 5.1 (user DUYỆT 2026-07-12, giữ "consume" — đã có 3 tiền lệ trong bài; anchor ×1 cả 2 bản).** Khớp với câu hỏi MỚI đã duyệt ở A2; bỏ tuyên bố "yes" cho vế RNN-multi-modal (mâu thuẫn 4.4). ⚠️ **HẤP THỤ A3#7:** D1a thay cả câu chứa "The question raised in Chapter 1" → khi áp BỎ QUA phép A3#7, D1a đã dùng "Section 1".
- **D1a — thay 2 câu đầu đoạn.** **EN old:** `The question raised in Chapter 1 was whether such a decoder could combine the two modalities well enough to outperform either modality on its own, as well as the earlier multi-modal systems built on recurrent networks. Within the setting we adopted, the answer is yes.` → **EN new:** `The question raised in Section 1 was whether such a decoder could fuse the two modalities well enough to outperform both of its single-modality counterparts under a controlled input setting. Within that setting the answer is affirmative.` / **VI old:** `Câu hỏi nêu ở Chương 1 là liệu một bộ giải mã như vậy có thể kết hợp hai phương thức đủ tốt để vượt từng phương thức đơn lẻ, cũng như các hệ đa phương thức trước đây xây trên mạng hồi quy hay không. Trong thiết lập chúng tôi áp dụng, câu trả lời là có.` → **VI new:** `Câu hỏi nêu ở Phần 1 là liệu một bộ giải mã như vậy có thể hợp nhất hai phương thức đủ tốt để vượt cả hai biến thể đơn-phương-thức của chính nó trong một thiết lập đầu vào có kiểm soát hay không. Trong thiết lập đó, câu trả lời là có.` (VI dùng "Phần 1" theo quy ước sẵn có của bản VI)
- **D1b — thêm câu ranh giới cuối đoạn (bản sửa 2026-07-12 theo yêu cầu user: bỏ em-dash, dùng mệnh đề phẩy).** **EN sau** `stays closer to the true structure of the expression.` **thêm:** ` The results do not, however, support a direct comparison with the earlier RNN-based multi-modal recognizers, which consume raw pen traces rather than a ground-truth structure; what they establish is the effectiveness of fusion inside a Transformer decoder, a combination that prior work had left unexplored.` / **VI sau** `bám sát cấu trúc thật của biểu thức hơn.` **thêm:** ` Tuy vậy, các kết quả này chưa cho phép so sánh trực tiếp với các bộ nhận dạng đa phương thức dựa trên RNN trước đây, vốn tiêu thụ nét bút thô thay vì cấu trúc ground-truth; điều chúng xác lập là tính hiệu quả của việc hợp nhất bên trong bộ giải mã Transformer, một tổ hợp mà các công trình trước còn bỏ ngỏ.` ("raw pen traces"/"nét bút thô", "ground-truth" đều khớp thuật ngữ sẵn có)

## ✅ ĐÃ ÁP 2026-07-13 — 📊 GÓI CHƯƠNG 4 (E1–E5, user DUYỆT TRỌN GÓI 2026-07-12) — số liệu tính từ results/, bằng chứng: results/analysis_significance_editdist.{py,txt}
> → **TOÀN BỘ LƯỢT DUYỆT HOÀN TẤT (A1–A4, B1–B4, C1–C3, D1, E1–E5). Chờ lệnh "sửa chính thức" để áp một lượt vào 2 docx + xuất 2 PDF.** Khi áp nhớ: A3#7 bỏ qua (D1a hấp thụ); backup 2 docx trước; user cần tải 4 PDF refs mới + (tùy chọn) abstract [39]–[41].
> **Số liệu nền (đã sanity-check khớp 100% Bảng 3/4/5; seed7_test ≡ abl_dual_shared từng mẫu):** Wilson 95% CI micro: concat 75,89 [74,40–77,31], shared 74,17 [72,66–75,63], cascaded 73,51 [71,99–74,98] (nửa rộng ≈ ±1,5đ). McNemar exact (gộp 3.330): shared vs concat b=132/c=189 **p=0,0017**; shared vs cascaded b=175/c=153 p=0,246; concat vs cascaded p=6,7e-05; shared vs online-only p=8,3e-43; bi vs uni p=4,7e-55. ExpRate ≤1/≤2 (dual_shared): 2014: 84,26/87,21; 2016: 79,16/83,96; 2019: 80,13/85,31; micro 81,02/85,41. Phân loại 860 lỗi: 63,3% chạm token cấu trúc ({ } ^ _ \frac \sqrt), 6,0% thuần thay ký hiệu, 30,7% thiếu/dư ký hiệu; tỉ lệ lỗi theo độ dài GT: 1–10: 7,8% (98/1261), 11–20: 20,4% (208/1022), 21–30: 40,2% (248/617), >30: 71,2% (306/430).
**Việc E1 — Đoạn kiểm định thống kê, chèn SAU caption Bảng 5 (đoạn mới, cả 2 bản).**
- **EN:** `To check that these comparisons do not rest on a single reading of the numbers, we complement them with paired significance tests. Since every variant is evaluated on the same test expressions, McNemar's exact test applies: pooling the three test sets (n = 3,330), the gain of fusion over the strongest single modality and the gain of bidirectional over unidirectional training are both significant beyond doubt (p < 10⁻⁴⁰). Among the fusion designs the picture is finer: the small edge of early fusion over the shared-query design (1.72 points) is unlikely to be due to chance (p ≈ 0.002), the shared-query versus cascaded difference is not significant (p ≈ 0.25), and the 95% confidence interval of each fusion variant's micro-averaged ExpRate spans about ±1.5 points. These tests compare the trained models at hand on shared test data; variability across training seeds is a separate question, noted in Section 5.3. Overall, the statistics confirm the reading above: what matters is fusing the two sources and training in both directions, far more than the precise wiring of the fusion.`
- **VI:** `Để các so sánh này không chỉ dựa trên một cách đọc số liệu, chúng tôi bổ sung kiểm định ý nghĩa theo cặp. Vì mọi biến thể được đánh giá trên cùng các biểu thức kiểm thử, kiểm định chính xác McNemar áp dụng được: gộp ba tập kiểm thử (n = 3.330), mức hơn của hợp nhất so với đơn-phương-thức mạnh nhất và của huấn luyện hai chiều so với một chiều đều có ý nghĩa không phải bàn cãi (p < 10⁻⁴⁰). Giữa các thiết kế hợp nhất, bức tranh tinh hơn: lợi thế nhỏ của hợp nhất sớm so với thiết kế dùng chung truy vấn (1,72 điểm) khó có thể do ngẫu nhiên (p ≈ 0,002), khác biệt giữa dùng chung truy vấn và nối tiếp không có ý nghĩa thống kê (p ≈ 0,25), còn khoảng tin cậy 95% cho ExpRate trung bình vi mô của mỗi biến thể hợp nhất rộng khoảng ±1,5 điểm. Các kiểm định này so sánh chính những mô hình đã huấn luyện trên cùng dữ liệu kiểm thử; độ biến thiên giữa các seed huấn luyện là câu hỏi riêng, đã nêu ở Mục 5.3. Nhìn chung, thống kê củng cố cách đọc ở trên: điều quan trọng là hợp nhất hai nguồn và huấn luyện theo cả hai chiều, hơn hẳn cách đấu nối cụ thể của phép hợp nhất.`
**Việc E2 — ExpRate ≤1/≤2.** (a) 4.3, chèn sau **EN** `ExpRate is a strict metric, because a single incorrect symbol or relation makes the entire expression wrong.`: ` Following common practice in the CROHME literature, we additionally report ExpRate ≤1 and ≤2: the percentage of expressions whose predicted sequence differs from the ground truth by at most one or two token-level edit operations.` / sau **VI** `ExpRate là độ đo nghiêm ngặt, vì chỉ một ký hiệu hay quan hệ sai cũng làm cả biểu thức bị tính là sai.`: ` Theo thông lệ trong các công bố CROHME, chúng tôi báo cáo thêm ExpRate ≤1 và ≤2: tỉ lệ phần trăm biểu thức có chuỗi dự đoán khác nhãn tối đa một hoặc hai phép sửa ở mức token.` (b) Bảng 3 (tables[2]): thêm 2 cột `≤1 (%)`/`≤2 (%)` với số liệu nền ở trên (VI dùng dấu phẩy thập phân); giữ font 9,5pt như bảng hiện có. (c) 4.5, chèn sau **EN** `confirming its stability.`: ` Under the relaxed metrics, 81.02% of the test expressions lie within one edit operation of the ground truth and 85.41% within two, so a sizeable share of the remaining errors are single-token slips.` / sau **VI** `khẳng định tính ổn định.`: ` Theo các độ đo nới lỏng, 81,02% biểu thức kiểm thử nằm trong phạm vi một phép sửa so với nhãn và 85,41% trong phạm vi hai phép sửa, nghĩa là một phần đáng kể lỗi còn lại chỉ là trượt một token.`
**Việc E3 — Định lượng lỗi 4.7, chèn sau câu "ba loại" (EN `...failures on long expressions, where a single slip invalidates the whole sequence.` / VI `...nơi chỉ một sai sót cũng làm hỏng cả chuỗi.`):**
- **EN:** ` A token-level analysis of all 860 mis-recognized expressions across the three test sets makes this concrete: 63% of the errors touch at least one structural token (braces, superscript and subscript markers, fractions, or radicals), only 6% are pure symbol substitutions, and the remaining 31% involve omitted or spurious symbols with the structure otherwise intact. Length is the other dominant factor: the error rate grows from 7.8% for expressions of at most ten tokens to 20.4%, 40.2%, and 71.2% for lengths 11-20, 21-30, and above 30, respectively.`
- **VI:** ` Phân tích ở mức token trên toàn bộ 860 biểu thức bị nhận dạng sai của ba tập kiểm thử làm rõ nhận định này: 63% số lỗi chạm tới ít nhất một token cấu trúc (ngoặc nhóm, dấu chỉ số trên/dưới, phân số hoặc căn thức), chỉ 6% là thuần thay thế ký hiệu, và 31% còn lại liên quan đến ký hiệu bị bỏ sót hoặc dư thừa trong khi cấu trúc còn nguyên. Độ dài là yếu tố chi phối còn lại: tỉ lệ lỗi tăng từ 7,8% với biểu thức tối đa mười token lên 20,4%, 40,2% và 71,2% cho các độ dài 11–20, 21–30 và trên 30.`
**Việc E4 — Đoạn giải thích khoảng cách BTTR, chèn SAU caption Bảng 4 (đoạn mới, cả 2 bản).** ĐÃ ĐỐI CHIẾU references/23.pdf: BTTR huấn luyện 4×1080Ti, Adadelta wd=1e-4/ρ=0,9/ε=1e-6, không nêu số epoch; số 53,96/52,31/52,96 = "Ours-Bi" khớp Bảng 1.
- **EN:** `A remark on absolute levels is in order. The offline-only baseline reaches 47.61% on CROHME 2014, whereas BTTR, architecturally the same model, reports 53.96% [23]. The gap stems from the training protocol rather than the architecture: every variant in this ablation, including the offline-only one, is trained for the fixed 50-epoch budget of Section 4.2 under one shared configuration, with no per-variant tuning, while the published BTTR figure was obtained under that model's own training setup [23]. The offline-only row is therefore the controlled reference point of this grid, not a re-benchmark of BTTR.`
- **VI:** `Cần một lưu ý về mức tuyệt đối. Biến thể offline-only đạt 47,61% trên CROHME 2014, trong khi BTTR, về kiến trúc là cùng một mô hình, công bố 53,96% [23]. Khoảng cách nằm ở quy trình huấn luyện chứ không phải ở kiến trúc: mọi biến thể trong thí nghiệm loại bỏ này, kể cả biến thể offline-only, được huấn luyện với ngân sách cố định 50 epoch theo một cấu hình chung như Mục 4.2, không tinh chỉnh riêng cho từng biến thể, còn con số công bố của BTTR thu được dưới thiết lập huấn luyện riêng của mô hình đó [23]. Vì vậy dòng offline-only là mốc tham chiếu có kiểm soát của lưới thí nghiệm, không phải một lần đo lại BTTR.`
**Việc E5 — VI-only, đôi với A3#8:** `Các thí nghiệm trong chương này` → `Các thí nghiệm trong phần này` (4.8; "chương" chỉ còn đúng 1 chỗ này trong VI, "Chương 1" ở 5.1 đã do D1a xử lý).

## ⏳ CHỜ ÁP MỘT LƯỢT — ABSTRACT + PHẦN 1 + PHẦN 2 (2.1/2.2) + HÌNH 1 (soạn 2026-07-04, user đang đọc, CHƯA áp; 13 việc)
User thấy câu Abstract "cung cấp cho nhánh online biểu diễn… chuẩn của biểu thức" quá cụt. Đã chốt **cách 2** (thêm lý do, KHÔNG đưa "cận trên" vào Abstract — user không thích). Câu này EN[3]/VI[3], song song; áp cả hai. Anchor verify ×1 khi áp.
- **VI old:** `Tập trung vào bộ giải mã, chúng tôi cung cấp cho nhánh online biểu diễn cấu trúc quan hệ ký hiệu chuẩn của biểu thức.`
- **VI new:** `Vì trọng tâm đặt ở bộ giải mã, chúng tôi cung cấp cho nhánh online biểu diễn quan hệ ký hiệu chuẩn của biểu thức — nhãn cấu trúc ground-truth do CROHME cung cấp — để các thí nghiệm tập trung đánh giá riêng cơ chế hợp nhất ở tầng bộ giải mã.`
- **EN old:** `Focusing on the decoder, we supply the online branch with a standard symbol-relation representation of the expression.`
- **EN new:** `Because our focus is the decoder, we supply the online branch with the standard symbol-relation representation of the expression—CROHME's ground-truth structural annotation—so that the experiments isolate the decoder-level fusion mechanism.`
**Câu 2 — câu ablation (user ĐỒNG Ý 2026-07-04): thêm nhịp nối "cho thấy điều gì tạo nên độ chính xác này".** EN[3]/VI[3], ngay sau câu số kết quả.
- **VI old:** `Một thí nghiệm loại bỏ (ablation) có kiểm soát cho thấy hợp nhất ở tầng bộ giải mã cải thiện khoảng 8,5 điểm phần trăm so với bộ giải mã đơn-phương-thức mạnh nhất, và huấn luyện hai chiều bổ sung thêm khoảng 10,7 điểm, trong khi cách thiết kế fusion cụ thể chỉ ảnh hưởng nhỏ.`
- **VI new:** `Một thí nghiệm loại bỏ (ablation) có kiểm soát cho thấy điều gì tạo nên độ chính xác này: hợp nhất ở tầng bộ giải mã cải thiện khoảng 8,5 điểm phần trăm so với bộ giải mã đơn-phương-thức mạnh nhất và huấn luyện hai chiều bổ sung thêm khoảng 10,7 điểm, trong khi cách thiết kế fusion cụ thể chỉ ảnh hưởng nhỏ.`
- **EN old:** `A controlled ablation shows that decoder-level fusion improves over the strongest single-modality decoder by about 8.5 percentage points, and that bidirectional training adds a further 10.7 points, whereas the particular fusion design has only a minor effect.`
- **EN new:** `A controlled ablation shows what drives this accuracy: decoder-level fusion improves over the strongest single-modality decoder by about 8.5 percentage points and bidirectional training adds a further 10.7 points, whereas the particular fusion design has only a minor effect.`
- (Đã kiểm 4.6: đã nêu rõ "mạnh nhất (online-only)" + so cả offline-only 27–28đ → "mạnh nhất" ở Abstract bất khả bắt bẻ; không cần sửa 4.6.)
**Câu 3 — XÓA dòng phụ đề dưới tiêu đề (user yêu cầu 2026-07-04).** Xóa CẢ đoạn (paragraph [1]) ở cả 2 docx:
- **VI:** xóa đoạn `Phần 1–5 — Giới thiệu, Nghiên cứu liên quan, Phương pháp, Thực nghiệm và Kết luận`
- **EN:** xóa đoạn `Sections 1–5 — Introduction, Related Works, Methodology, Experiments, and Conclusion`
- (Xóa nguyên paragraph, không để dòng trống thừa; verify đoạn khớp ×1 trước khi xóa.)
**Việc 4 — SỬA & NHÚNG LẠI HÌNH 1 (user bắt lỗi 2026-07-04, đã sửa SVG + render, CHỜ NHÚNG).** figure1_example.svg + _vi.svg (Hình 1 = inline_shapes[0]).
- Lỗi 1: tiêu đề cột (b) "(b) Online modality — symbol-relation sequence (SRT)" quá dài, tràn qua CẢ 2 đường gạch dọc. Sửa: gạch dọc 320/640 → **288/672**, cỡ chữ 3 tiêu đề cột 13,5 → **11,5**.
- Lỗi 2: sau khi nới gạch phải, ô LaTeX cột (c) dính sát đường gạch. Sửa: **dời toàn cột (c) sang phải +22px** (mọi phần tử: header x800→822, rect x676→698, mono x800→822, note x800→822, path M762→M784, "4π" x826→848) → ô (c) cách gạch ~26px, bằng ô (a).
- ⚠️ **BÀI HỌC render:** `cairosvg` **LÀM RỚT chữ π** (ô π trống, "4π"→"4"). Với hình có ký tự Hy Lạp (π/α/β…) PHẢI render đường vòng **soffice --convert-to pdf → pdftoppm -r 200 -png** (DejaVu Serif có π), ĐỪNG dùng cairosvg. (Hình 3/4 đã nhúng không có Hy Lạp nên OK.)
- PNG chuẩn (2003×878, có π, hết lỗi) đang ở outputs/f1c/ — khi nhúng: cp về figures/figure1_example.png + _vi.png, rồi thay blob inline_shapes[0] (giữ docPr).
**Việc 5 — NỐI 2 CÂU ở Mục 1 (Giới thiệu), user duyệt 2026-07-04.** Đoạn HME offline (EN[12]/VI[12]); thay dấu chấm giữa 2 câu bằng mệnh đề dẫn ví dụ. Anchor verify ×1.
- **VI old:** `chuyển giao trực tiếp. Các kết quả tiêu biểu theo hướng này gồm mạng nơ-ron sâu LeNet-5`
- **VI new:** `chuyển giao trực tiếp, như minh họa bởi các kết quả tiêu biểu theo hướng này: mạng nơ-ron sâu LeNet-5`
- **EN old:** `can be transferred to it directly. Representative results in this direction include modified LeNet-5`
- **EN new:** `can be transferred to it directly, as illustrated by representative results in this direction: modified LeNet-5`
**Việc 6 — NỐI câu 2+3 ở Mục 1 (đoạn RNN→Transformer, user duyệt hướng 2-câu 2026-07-04).** GIỮ câu 1 ("Tính toán tuần tự… làm chậm huấn luyện [7].") riêng; chỉ nối câu 2 và 3. EN[13]/VI[13]. Anchor verify ×1.
- **VI old:** `một kiến trúc chỉ dựa trên cơ chế chú ý. Cơ chế tự chú ý cho phép mô hình`
- **VI new:** `một kiến trúc chỉ dựa trên cơ chế chú ý; cơ chế tự chú ý của nó cho phép mô hình`
- **EN old:** `an architecture based solely on attention mechanisms. Its self-attention mechanism allows`
- **EN new:** `an architecture based solely on attention mechanisms; its self-attention mechanism allows`
**Việc 7 — bỏ "Formally"/"Một cách hình thức" mở đầu 2.1 (user chọn phương án A, 2026-07-04).** Câu mở phát biểu bài toán, EN[2.1]/VI[2.1]. Anchor verify ×1.
- **EN old:** `Formally, HMER is a sequence-prediction problem: given an input expression`
- **EN new:** `HMER can be cast as a sequence-prediction problem: given an input expression`
- **VI old:** `Một cách hình thức, HMER là bài toán dự đoán chuỗi: cho một biểu thức đầu vào`
- **VI new:** `Có thể phát biểu HMER như một bài toán dự đoán chuỗi: cho một biểu thức đầu vào`
**Việc 8 — thêm trích dẫn [28] câu pipeline truyền thống 2.1 (user chốt "[28] thôi", 2026-07-04).** ĐÃ XÁC MINH kho references/28.pdf bàn kỹ pipeline này (grammar×51, segmentation×40, grammar-based×17, structural analysis×18, symbol classification×12, "predefined grammar"). Anchor verify ×1.
- **EN old:** `structural analysis driven by predefined grammars.` → **new:** `structural analysis driven by predefined grammars [28].`
- **VI old:** `phân tích cấu trúc dựa trên văn phạm định trước.` → **new:** `phân tích cấu trúc dựa trên văn phạm định trước [28].`
- (Câu "dễ vỡ/lỗi lan truyền…" ngay sau KHÔNG cần trích dẫn riêng — [28] câu trước đã phủ, tránh lặp.)
**Việc 9 — thêm 2 trích dẫn nữa trong 2.1 (user chốt "thêm cả hai", 2026-07-04). ĐÃ XÁC MINH kho.**
- 9a — tiền xử lý online → **[21]** (survey online HMER: normalize/resample/pen-up/pen-down đều có).
  - **EN old:** `such as pen-up and pen-down indicators.` → **new:** `such as pen-up and pen-down indicators [21].`
  - **VI old:** `như chỉ báo nhấc/đặt bút.` → **new:** `như chỉ báo nhấc/đặt bút [21].`
- 9b — "tree-aware objectives to a sequence model" → **[30]** (TAMER; LƯU Ý là forward-ref, [30] đã trích đầy đủ ở đoạn sau Phần 2 — user chấp nhận).
  - **EN old:** `adding tree-aware objectives to a sequence model—so as to` → **new:** `adding tree-aware objectives to a sequence model [30]—so as to`
  - **VI old:** `thêm mục tiêu nhận biết cây vào một mô hình chuỗi.` → **new:** `thêm mục tiêu nhận biết cây vào một mô hình chuỗi [30].`
**Việc 10 — vá mắt xích đoạn 2 của 2.1 (user duyệt 2026-07-04): thêm mệnh đề "recurrent/hồi quy" vào nửa đầu để câu "Transformer bỏ hồi quy" ở nửa sau có đối trọng.** Anchor verify ×1.
- **EN old:** `and an attention-equipped decoder generates the output tokens one at a time, learning segmentation and structure implicitly.`
- **EN new:** `and an attention-equipped decoder—recurrent in the earliest such models—generates the output tokens one at a time, learning segmentation and structure implicitly.`
- **VI old:** `và một bộ giải mã trang bị cơ chế chú ý sinh từng token đầu ra, học ngầm việc phân đoạn và cấu trúc.`
- **VI new:** `và một bộ giải mã trang bị cơ chế chú ý — vốn là mạng hồi quy trong các mô hình đầu tiên theo hướng này — sinh từng token đầu ra, học ngầm việc phân đoạn và cấu trúc.`
**Việc 11 — thêm trích dẫn đoạn 2 của 2.1 (user chốt phương án a, 2026-07-04). ĐÃ XÁC MINH kho: 38.pdf = WAP (attention×77, encoder-decoder×11, recurrent×11, translation×8, gru×7); [23]=BTTR, [27]=CoMER.** Anchor verify ×1.
- 11a — câu mở → **[38]** (WAP; chống lưng cả mệnh đề "hồi quy" ở việc 10, khỏi trích riêng).
  - **EN old:** `borrowed from neural machine translation.` → **new:** `borrowed from neural machine translation [38].`
  - **VI old:** `mượn từ dịch máy nơ-ron.` → **new:** `mượn từ dịch máy nơ-ron [38].`
- 11b — câu cuối → **[23], [27]** (BTTR, CoMER; forward-ref, bàn kỹ ở đoạn sau Phần 2).
  - **EN old:** `architecture of choice for recent HMER systems.` → **new:** `architecture of choice for recent HMER systems [23], [27].`
  - **VI old:** `kiến trúc được ưa chuộng cho các hệ HMER gần đây.` → **new:** `kiến trúc được ưa chuộng cho các hệ HMER gần đây [23], [27].`
**Việc 12 — trích dẫn câu phân loại string/tree decoder, đoạn cuối 2.1 (user chọn phương án A, 2026-07-04). ĐÃ XÁC MINH: [23]BTTR/[38]WAP = string; [11]BPD/[16]SAN = tree (tiêu đề ghi rõ tree decoder).** Anchor verify ×1. (Các câu 2/3/5 của đoạn KHÔNG cần trích dẫn — đã phủ; câu 4 có [30] ở việc 9b.)
- **EN old:** `have been explored: string decoders and tree decoders.` → **new:** `have been explored: string decoders [23], [38] and tree decoders [11], [16].`
- **VI old:** `đã được khảo sát: bộ giải mã chuỗi và bộ giải mã cây.` → **new:** `đã được khảo sát: bộ giải mã chuỗi [23], [38] và bộ giải mã cây [11], [16].`
**Việc 13 — trích dẫn DenseNet gốc [24] ở đầu 2.2 (user chốt 2026-07-04).** [24]=Huang et al. "Densely connected convolutional networks" CVPR2017 (đã có trong bib, dùng ở Phần 3; ở 2.2 lần đầu giới thiệu DenseNet lại thiếu). Đặt ngay sau tên. Anchor verify ×1. (Câu DenseNet thứ 2 trong đoạn không cần — mô tả mô hình cụ thể.)
- **EN old:** `Densely connected networks (DenseNet) are an especially popular choice,` → **new:** `Densely connected networks (DenseNet) [24] are an especially popular choice,`
- **VI old:** `Mạng tích chập dày đặc (DenseNet) là lựa chọn đặc biệt phổ biến,` → **new:** `Mạng tích chập dày đặc (DenseNet) [24] là lựa chọn đặc biệt phổ biến,`
**Việc 14 — thêm mốc so sánh câu Truong [13] trong 2.2 (user duyệt 2026-07-04).** ĐÃ XÁC MINH 13.pdf: baseline = "Rec-wsl/Rec-wsl* trained using CROHME" (không có dữ liệu sinh); +1,47/2,88/2,67 là nhờ thêm dữ liệu CYK. Kiểm chéo baseline 63,13/63,20/56,05 khớp. Anchor verify ×1. (Số liệu giữ nguyên, chỉ thêm mệnh đề mốc.)
- **EN old:** `improves expression recognition rates by 1.47, 2.88, and 2.67 points, reaching 64.60%` → **new:** `improves expression recognition rates by 1.47, 2.88, and 2.67 points over the same recognizer trained without the generated data, reaching 64.60%`
- **VI old:** `cải thiện tỉ lệ nhận dạng biểu thức thêm 1,47, 2,88 và 2,67 điểm, đạt 64,60%` → **new:** `cải thiện tỉ lệ nhận dạng biểu thức thêm 1,47, 2,88 và 2,67 điểm so với chính bộ nhận dạng đó khi huấn luyện không có dữ liệu sinh thêm, đạt 64,60%`
**Việc 15 — đoạn đầu 2.3: đảo câu survey [21] xuống cuối + thêm câu nối "đầu-cuối" sau câu điểm→nét (user duyệt 2026-07-04).** Thay CẢ đoạn, thứ tự: câu1 (giữ) → câu điểm→nét (giữ) → CÂU NỐI MỚI → câu survey [21] (chuyển xuống cuối). Trích dẫn cho các câu khẳng định: ĐANG TRA (việc 16). Khi áp: thay nguyên đoạn, anchor = đoạn cũ verify ×1.
- **Câu nối mới EN:** `Building on these representations, modern online recognizers are typically neural and end-to-end, encoding the pen sequence and decoding it directly into LaTeX.`
- **Câu nối mới VI:** `Trên nền các biểu diễn đó, các bộ nhận dạng online hiện đại phần lớn là mạng nơ-ron đầu-cuối: mã hóa chuỗi bút rồi giải mã thẳng thành LaTeX.`
**Việc 16 — trích dẫn [22] SCAN cho câu điểm→nét ở 2.3 (user chốt "chỉ 22", 2026-07-04). ĐÃ XÁC MINH 22.pdf: stroke-level×76, point-level×27, segmentation×16, stroke-constrained×13.** Anchor verify ×1. (Câu 1 định nghĩa + câu nối chủ đề KHÔNG cần trích dẫn — đã lý giải.)
- **EN old:** `reduce segmentation ambiguity.` → **new:** `reduce segmentation ambiguity [22].`
- **VI old:** `giảm nhập nhằng phân đoạn.` → **new:** `giảm nhập nhằng phân đoạn [22].`
- ⚠️ Lưu ý áp: việc 16 sửa câu NẰM TRONG đoạn mà việc 15 thay-cả-đoạn → khi áp phải để bản đoạn mới (việc 15) ĐÃ chứa sẵn "[22]" ở câu điểm→nét, làm gộp, tránh đụng anchor.
**Việc 17 — thêm câu dẫn mở đầu 2.4 (user duyệt 2026-07-04).** Chèn TRƯỚC câu MAN ("Bridging the two modalities…"/"Bắc cầu giữa hai phương thức…"), cùng đoạn. Không cần trích dẫn (câu khung). Anchor verify ×1.
- **EN — chèn trước `Bridging the two modalities, Wang et al. [35]`:** `Because offline images and online trajectories describe the same expression in complementary ways—appearance and stroke shape on one hand, temporal order and structure on the other—recognizers that exploit both can, in principle, surpass either input alone. `
- **VI — chèn trước `Bắc cầu giữa hai phương thức, Wang và cộng sự [35]`:** `Vì ảnh offline và quỹ đạo online mô tả cùng một biểu thức theo những cách bổ trợ nhau — một bên là hình thái và dáng nét, một bên là thứ tự thời gian và cấu trúc — các bộ nhận dạng khai thác cả hai về nguyên tắc có thể vượt trội so với dùng riêng một đầu vào. `
**Việc 18 — thêm từ nối câu cuối đoạn đầu 2.5 (user chọn A, 2026-07-04).** Câu chỉ-đường móc vào mạch. Anchor verify ×1.
- **EN old:** `Several Transformer-based decoders are particularly relevant to our work.` → **new:** `Among these, several Transformer-based decoders are particularly relevant to our work.`
- **VI old:** `Vài bộ giải mã Transformer đặc biệt liên quan đến công trình của chúng tôi.` → **new:** `Trong số đó, một vài bộ giải mã Transformer đặc biệt liên quan đến công trình của chúng tôi.`
**Việc 19 — nối câu 1 với câu 2 đoạn đầu 2.6 bằng dấu hai chấm (user duyệt 2026-07-04).** Anchor verify ×1.
- **EN old:** `reveals a consistent pattern. Offline methods—particularly` → **new:** `reveals a consistent pattern: offline methods—particularly`
- **VI old:** `cho thấy một quy luật nhất quán. Các phương pháp offline` → **new:** `cho thấy một quy luật nhất quán: các phương pháp offline`
**Việc 20 — bỏ gạch-treo (suspended hyphen) trong EN, viết đủ cho nhất quán (user chốt "đổi", 2026-07-04).** replace-ALL, **3 vị trí** trong EN (đoạn SCAN "…use…information", 2.6 "…fusing…features", 3.13 "…using…features"). VI cả 3 chỗ đã viết đủ, không đụng.
- **EN replace-all:** `stroke-, point-, and pixel-level` → `stroke-level, point-level, and pixel-level` (đếm=3, đổi hết).
**Việc 21 — mở rộng câu đầu đoạn cuối 2.6 (fusion location) bằng mệnh đề lý do (user chọn A, 2026-07-04).** Anchor verify ×1.
- **EN old:** `It is also instructive to distinguish where the two modalities are combined.` → **new:** `It is also instructive to distinguish where the two modalities are combined, because the point at which fusion happens largely determines how much the two sources can interact.`
- **VI old:** `Cũng cần phân biệt nơi hai phương thức được kết hợp.` → **new:** `Cũng cần phân biệt nơi hai phương thức được kết hợp, vì vị trí xảy ra hợp nhất phần lớn quyết định mức độ hai nguồn có thể tương tác với nhau.`
**Việc 22 — GỘP 2 câu cuối 2.6 (khoảng trống + mục tiêu) thành 1, đổi mở đầu "To date, however,"→"So far,", bỏ em-dash aside (user chọn B, 2026-07-04).** Thay 2 câu bằng 1. Anchor verify ×1.
- **EN old (2 câu):** `To date, however, the multi-modal and the Transformer-decoder lines of research have largely remained separate. Closing this gap—an effective multi-modal HMER model built on a Transformer decoder—is precisely the objective of the present work.`
- **EN new (1 câu):** `So far, the multi-modal and the Transformer-decoder lines of research have remained largely separate, and the objective of this thesis is to close this gap by building an effective multi-modal HMER model on a Transformer decoder.`
- **VI old (2 câu):** `Đến nay, hai dòng nghiên cứu đa phương thức và bộ-giải-mã-Transformer phần lớn vẫn tách biệt. Lấp khoảng trống này — một mô hình HMER đa phương thức hiệu quả xây trên bộ giải mã Transformer — chính là mục tiêu của luận văn.`
- **VI new (1 câu):** `Đến nay, hai dòng nghiên cứu đa phương thức và bộ-giải-mã-Transformer phần lớn vẫn tách biệt, và mục tiêu của luận văn là lấp khoảng trống này bằng cách xây dựng một mô hình HMER đa phương thức hiệu quả trên bộ giải mã Transformer.`
**Việc 23 — bỏ em-dash aside "the strategy adopted in this work" ở câu cuối 2.6 (fusion taxonomy) — thừa vì đã có "which we argue is the most suitable choice" (user chọn A, 2026-07-04).** Anchor verify ×1.
- **EN old:** `Decoder-level fusion—the strategy adopted in this work—keeps each encoder specialized yet allows` → **new:** `Decoder-level fusion keeps each encoder specialized yet allows`
- **VI old:** `Hợp nhất ở tầng bộ giải mã — chiến lược chúng tôi chọn — giữ mỗi bộ mã hóa chuyên biệt nhưng cho phép` → **new:** `Hợp nhất ở tầng bộ giải mã giữ mỗi bộ mã hóa chuyên biệt nhưng cho phép`
**Việc 24 — bỏ em-dash → dấu phẩy ở câu offline đoạn đầu 2.7 (giữ ý "mô hình cây/đồ thị"; user chọn B, 2026-07-04).** Anchor verify ×1.
- **EN old:** `Offline methods—especially tree-structured and graph-based models—excel at parsing` → **new:** `Offline methods, especially tree-structured and graph-based models, excel at parsing`
- **VI old:** `Các phương pháp offline — nhất là mô hình cây và đồ thị — xuất sắc ở phân tích` → **new:** `Các phương pháp offline, nhất là mô hình cây và đồ thị, xuất sắc ở phân tích`
**Việc 25 — làm rõ "RNN-based approaches" → "RNN-based multi-modal approaches" ở câu Giới thiệu (Phần 1) (user chốt 2026-07-04).** Tránh chồng lấn với "uni-modal" (vì có cả RNN đơn-phương-thức như TAP); các chỗ "RNN-based" khác đã có "multi-modal" cạnh → GIỮ NGUYÊN, chỉ sửa câu này. Anchor verify ×1.
- **EN old:** `the limitations of uni-modal and RNN-based approaches` → **new:** `the limitations of uni-modal and RNN-based multi-modal approaches`
- **VI old:** `hạn chế của các hướng đơn-phương-thức và dựa trên RNN` → **new:** `hạn chế của các hướng đơn-phương-thức và các hướng đa phương thức dựa trên RNN`
**Việc 26 — thay câu chủ đề đoạn cuối 2.7 (caveat Bảng 1) bằng câu có luận điểm (user chọn C, 2026-07-04).** Thay "should be read with care" bằng "số phản ánh khác biệt thiết lập không kém khác biệt thiết kế". Anchor verify ×1.
- **EN old:** `Finally, the comparison in Table 1 should be read with care.` → **new:** `Finally, the results in Table 1 reflect differences in experimental setup as much as in model design.`
- **VI old:** `Cuối cùng, cần đọc Bảng 1 một cách thận trọng.` → **new:** `Cuối cùng, các kết quả trong Bảng 1 phản ánh khác biệt về thiết lập thí nghiệm không kém gì khác biệt về thiết kế mô hình.`
**Việc 27 — câu roadmap đầu Phần 4: (a) thêm cụm dẫn "In what follows"; (b) thêm "implementation/cài đặt" (4.2) vào danh sách cho khớp đủ 4 mục, ĐÚNG THỨ TỰ dữ liệu(4.1)→cài đặt(4.2)→độ đo(4.3)→thiết lập(4.4) (user chốt 2026-07-04).** Anchor verify ×1.
- **EN old:** `online recognition pipeline. We describe the datasets, metric, and setting (Sections 4.1-4.4)` → **new:** `online recognition pipeline. In what follows, we describe the datasets, implementation, metric, and setting (Sections 4.1-4.4)`
- **VI old:** `pipeline nhận dạng online đầy đủ. Chúng tôi mô tả dữ liệu, độ đo và thiết lập (Mục 4.1–4.4)` → **new:** `pipeline nhận dạng online đầy đủ. Ở các mục tiếp theo, chúng tôi mô tả dữ liệu, cài đặt, độ đo và thiết lập (Mục 4.1–4.4)`
**Việc 28 — viết lại câu đầu 4.4 (chủ ngữ "This section", bỏ "we/our", bỏ "read carefully", thêm vế "kết luận có thể rút ra"; user chọn A, 2026-07-04).** Anchor verify ×1.
- **EN old:** `The scope of our evaluation must be read carefully, as it directly shapes how the reported numbers should be interpreted.` → **new:** `This section sets out the scope under which the proposed model is evaluated, which directly shapes how the reported numbers should be interpreted and what conclusions they support.`
- **VI old:** `Phạm vi của đánh giá này cần được hiểu chính xác, vì nó ảnh hưởng trực tiếp đến cách diễn giải các con số báo cáo.` → **new:** `Phần này nêu rõ phạm vi đánh giá mô hình đề xuất, vốn ảnh hưởng trực tiếp đến cách diễn giải các con số báo cáo và những kết luận có thể rút ra từ chúng.`
**Việc 29 — bỏ 3 em-dash còn lại trong đoạn 4.4 (user đồng ý hết A+B+C, 2026-07-04). Kết hợp việc 28 → đoạn 4.4 mới KHÔNG còn em-dash (chỉ giữ ngoặc đơn "(External methods…)").** 3 phép, mỗi phép anchor verify ×1.
- **A (câu 3, sửa cả lỗi EN " - " spaced-hyphen):** EN old `standard symbol-relation representation of the expression - the structural annotation provided by CROHME - which is treated as a given, clean input.` → new `standard symbol-relation representation of the expression provided by CROHME, which is treated as a given, clean input.` | VI old `biểu diễn quan hệ ký hiệu chuẩn của biểu thức — nhãn cấu trúc do CROHME cung cấp — và được xem là đầu vào cho sẵn, sạch.` → new `biểu diễn quan hệ ký hiệu chuẩn của biểu thức do CROHME cung cấp, và được xem là đầu vào cho sẵn, sạch.`
- **B+D (GỘP: VI đổi em-dash "— vốn"→", vốn" KÈM gộp câu ngoặc đơn vào câu trước bằng `;`, bỏ ngoặc đơn đứng riêng; user duyệt 2026-07-04):**
  - **EN old:** `offline-only systems, which never receive such structural information. (External methods are listed only for context in Section 2, Table 1.)` → **new:** `offline-only systems, which never receive such structural information; external methods are listed only for context in Section 2, Table 1.`
  - **VI old:** `các hệ chỉ-offline — vốn không hề nhận thông tin cấu trúc đó. (Các phương pháp ngoài chỉ được liệt kê để đặt bối cảnh ở Phần 2, Bảng 1.)` → **new:** `các hệ chỉ-offline, vốn không hề nhận thông tin cấu trúc đó; các phương pháp ngoài chỉ được liệt kê để đặt bối cảnh ở Phần 2, Bảng 1.`
- **C (câu cuột, cleft giữ nhấn, đưa mệnh đề xuống cuối):** EN old `It is the ablation in Section 4.6—where every variant receives exactly the same inputs and is trained under the same protocol—that provides the fair, controlled comparison underpinning our claims.` → new `It is the ablation in Section 4.6 that provides the fair, controlled comparison underpinning our claims, since there every variant receives exactly the same inputs and is trained under the same protocol.` | VI old `Chính thí nghiệm loại bỏ ở Mục 4.6 — nơi mọi biến thể nhận cùng đầu vào và được huấn luyện theo cùng giao thức — mới là phép so sánh công bằng, có kiểm soát; và đây mới là cơ sở cho các khẳng định của chúng tôi.` → new `Chính thí nghiệm loại bỏ ở Mục 4.6 mới là phép so sánh công bằng, có kiểm soát làm cơ sở cho các khẳng định của chúng tôi, vì ở đó mọi biến thể nhận cùng đầu vào và được huấn luyện theo cùng giao thức.`
**Việc 30 — câu 4.5 "editions/competitions" → "test sets", né lặp bằng "their/của chúng" (user chọn A, 2026-07-04).** Anchor verify ×1.
- **EN old:** `The consistency across the three editions indicates that the model generalizes well across the differing distributions of the three competitions.` → **new:** `The consistency across the three test sets indicates that the model generalizes well across their differing distributions.`
- **VI old:** `Sự nhất quán giữa ba kỳ thi cho thấy mô hình tổng quát tốt qua các phân phối khác nhau của ba cuộc thi.` → **new:** `Sự nhất quán trên ba tập kiểm thử cho thấy mô hình tổng quát tốt qua các phân phối khác nhau của chúng.`
**Việc 31 — viết lại câu đầu 4.7 (dài hơn, 1 phẩy, bỏ "more telling"; user duyệt 2026-07-04).** Chỉ đổi câu 1, câu ví dụ sau giữ nguyên. Anchor verify ×1.
- **EN old:** `Beyond aggregate scores, the per-expression predictions reveal where the model succeeds and where it fails.` → **new:** `Beyond the aggregate scores, individual predictions reveal the model's characteristic successes as well as the failures that expose its remaining limitations.`
- **VI old:** `Ngoài các điểm số tổng hợp, dự đoán theo từng biểu thức cho thấy mô hình thành công và thất bại ở đâu.` → **new:** `Ngoài các điểm số tổng hợp, dự đoán theo từng biểu thức hé lộ những thành công đặc trưng của mô hình cùng những thất bại vốn phơi bày các hạn chế còn lại của nó.`
**Việc 32 — viết lại câu cuối 4.7 (bỏ "by…by…by"/"bởi…bởi…bởi", liệt kê "ba loại" bằng dấu hai chấm; user chọn B, 2026-07-04).** Anchor verify ×1.
- **EN old:** `The remaining errors are dominated by this kind of structural mis-attachment, by confusions between visually similar symbols, and by very long expressions in which a single slip invalidates the whole sequence.` → **new:** `Most of the remaining errors fall into three kinds: structural mis-attachments of the sort shown above, confusions between visually similar symbols, and failures on long expressions, where a single slip invalidates the whole sequence.`
- **VI old:** `Các lỗi còn lại bị chi phối bởi loại gắn-sai cấu trúc này, bởi nhầm lẫn giữa các ký hiệu giống nhau về thị giác, và bởi các biểu thức rất dài mà chỉ một sai sót cũng làm hỏng cả chuỗi.` → **new:** `Phần lớn lỗi còn lại rơi vào ba loại: gắn-sai cấu trúc kiểu như trên, nhầm lẫn giữa các ký hiệu giống nhau về thị giác, và thất bại ở các biểu thức dài, nơi chỉ một sai sót cũng làm hỏng cả chuỗi.`
**Việc 33 — câu đầu 4.8: bỏ 2 em-dash bằng cách TÁCH 2 câu (danh sách 3 câu hỏi vào dấu `:`; câu 2 = phương án A khung phạm vi). User chọn A, 2026-07-04.** Thay 1 câu → 2 câu. Anchor verify ×1.
- **EN old:** `The experiments in this chapter speak directly to the questions that motivated our design—whether to fuse the two modalities at all, how that fusion should be wired inside the decoder, and how much the reading direction matters—and they support three main conclusions.` → **new:** `The experiments in this chapter directly address the questions that motivated our design: whether to fuse the two modalities at all, how that fusion should be wired inside the decoder, and how much the reading direction matters. Taken together, these results support three main conclusions about how best to build a multi-modal HMER decoder.`
- **VI old:** `Các thí nghiệm trong chương này trả lời trực tiếp những câu hỏi đã thúc đẩy thiết kế của chúng tôi — có nên hợp nhất hai phương thức hay không, việc hợp nhất nên được đấu nối thế nào bên trong bộ giải mã, và chiều đọc quan trọng đến mức nào — và củng cố ba kết luận chính sau đây.` → **new:** `Các thí nghiệm trong chương này trả lời trực tiếp những câu hỏi đã thúc đẩy thiết kế của chúng tôi: có nên hợp nhất hai phương thức hay không, việc hợp nhất nên được đấu nối thế nào bên trong bộ giải mã, và chiều đọc quan trọng đến mức nào. Gộp lại, các kết quả này củng cố ba kết luận chính về cách xây dựng tốt nhất một bộ giải mã HMER đa phương thức.`
**Việc 34 — 5.1 đóng góp #1: làm rõ "recurrent (RNN-based)" + trích dẫn MAN [35]/SCAN [22] (đã xác minh cả hai là RNN đa phương thức, 2.4/3.13). User chốt bản "(RNN-based)", 2026-07-04.** Anchor verify ×1.
- **EN old:** `inside a Transformer decoder rather than a recurrent one, so that the model retains` → **new:** `inside a Transformer decoder rather than a recurrent (RNN-based) one, as used by earlier multi-modal recognizers such as MAN [35] and SCAN [22], so that the model retains`
- **VI old:** `bên trong một bộ giải mã Transformer thay vì hồi quy, nhờ đó mô hình giữ được` → **new:** `bên trong một bộ giải mã Transformer thay vì một bộ giải mã hồi quy (dựa trên RNN) như ở các hệ đa phương thức trước đây MAN [35] và SCAN [22], nhờ đó mô hình giữ được`
**Việc 35 — 5.1 đóng góp #2: đổi em-dash giới thiệu danh sách → dấu hai chấm (các hyphen từ ghép giữ nguyên). User chốt 2026-07-04.** Anchor verify ×1.
- **EN old:** `three decoder-level fusion designs—early (concat), cascaded, and shared-query dual cross-attention.` → **new:** `three decoder-level fusion designs: early (concat), cascaded, and shared-query dual cross-attention.`
- **VI old:** `ba thiết kế hợp nhất ở tầng bộ giải mã — hợp nhất sớm (concat), nối tiếp (cascaded), và dùng chung truy vấn (shared-query dual cross-attention).` → **new:** `ba thiết kế hợp nhất ở tầng bộ giải mã: hợp nhất sớm (concat), nối tiếp (cascaded), và dùng chung truy vấn (shared-query dual cross-attention).`
**Việc 36 — 5.1 câu cuối: "A recurring observation/Một quan sát lặp lại" → "One consistent observation across our experiments/Một quan sát nhất quán qua các thí nghiệm" (học thuật hơn, không overclaim "finding"; user chọn #2, 2026-07-04).** Anchor verify ×1.
- **EN old:** `A recurring observation is that, once the structure is known` → **new:** `One consistent observation across our experiments is that, once the structure is known`
- **VI old:** `Một quan sát lặp lại là: một khi đã biết cấu trúc` → **new:** `Một quan sát nhất quán qua các thí nghiệm của chúng tôi là: một khi đã biết cấu trúc`
**Việc 37 — 5.2 câu 2: bỏ em-dash→phẩy, nâng giọng học thuật + bớt lặp đại từ (user chốt 2026-07-04).** Anchor verify ×1.
- **EN old:** `We state them explicitly—both for transparency and because they mark where our claims hold and where further validation is needed.` → **new:** `We make these limitations explicit, both for transparency and because they delineate where our claims hold and where further validation is needed.`
- **VI old:** `Chúng tôi nêu rõ chúng — vừa để minh bạch, vừa vì chúng cho thấy các khẳng định của chúng tôi đúng trong phạm vi nào và cần kiểm chứng thêm ở đâu.` → **new:** `Chúng tôi nêu rõ những hạn chế này, vừa để minh bạch, vừa vì chúng khoanh vùng phạm vi các khẳng định còn đúng và những điểm cần kiểm chứng thêm.`
**Việc 38 — 5.2: gộp 3 câu oracle-assumption thành 2 câu (gộp câu "giả định có sẵn lúc suy luận" vào câu 1; nối câu 2 bằng "and thus/do đó" bỏ dấu `;`). User duyệt 2026-07-04.** Thay khối 3 câu → 2 câu. Anchor verify ×1.
- **EN old:** `The online branch is given the ground-truth symbol-relation structure of each expression. We made this choice so that the work could concentrate on the decoder, but it means that the figures reported here describe how well the fusion performs when the structure is already correct; they are not a fair comparison with offline-only systems, which receive no such information. The model likewise assumes that this structure is available at inference time.` → **new:** `The online branch is given the ground-truth symbol-relation structure of each expression, which the model also assumes to be available at inference. This choice let the work concentrate on the decoder, but it means that the figures reported here describe how well the fusion performs when the structure is already correct, and thus are not a fair comparison with offline-only systems, which receive no such information.`
- **VI old:** `Nhánh online được cấp cấu trúc quan hệ ký hiệu ground-truth của mỗi biểu thức. Chúng tôi chọn vậy để công trình tập trung vào bộ giải mã, nhưng điều đó nghĩa là các con số ở đây mô tả mức hợp nhất hoạt động tốt thế nào khi cấu trúc đã đúng; chúng không phải so sánh công bằng với các hệ chỉ-offline vốn không nhận thông tin đó. Mô hình cũng giả định cấu trúc này có sẵn lúc suy luận.` → **new:** `Nhánh online được cấp cấu trúc quan hệ ký hiệu ground-truth của mỗi biểu thức, và mô hình cũng giả định cấu trúc này có sẵn lúc suy luận. Lựa chọn này giúp công trình tập trung vào bộ giải mã, nhưng cũng khiến các con số ở đây chỉ mô tả mức hợp nhất hoạt động tốt thế nào khi cấu trúc đã đúng, do đó không phải là so sánh công bằng với các hệ chỉ-offline vốn không nhận thông tin đó.`
**Việc 39 — 5.2 câu "ablation đánh giá thấp offline": viết rõ cơ chế (đo dưới cấu trúc hoàn hảo = lúc offline ít việc nhất), bỏ mơ hồ "image"→"offline image/ảnh offline", bớt lặp "branch". User duyệt 2026-07-04.** Anchor verify ×1.
- **EN old:** `A further consequence is that the ablation probably understates what the offline branch can do, since its value grows precisely when the structure is imperfect.` → **new:** `A further consequence is that the ablation probably understates the offline branch's contribution: since the structure is always perfect, the offline image is assessed in exactly the condition where it adds least, whereas it would help most when the structure is imperfect.`
- **VI old:** `Một hệ quả nữa là thí nghiệm loại bỏ có lẽ đánh giá thấp đóng góp của nhánh offline, vì giá trị của nó tăng lên đúng lúc cấu trúc không hoàn hảo.` → **new:** `Một hệ quả nữa là thí nghiệm loại bỏ có lẽ đánh giá thấp đóng góp của nhánh offline: vì cấu trúc luôn hoàn hảo, ảnh offline bị đánh giá đúng vào điều kiện mà nó bổ khuyết ít nhất, trong khi nó lại hữu ích nhất khi cấu trúc không hoàn hảo.`
**Việc 40 — 5.3 câu cuối: bỏ hai "và/and" sát nhau (đổi liên từ mục cuối danh sách "và/and"→"cũng như/as well as", giữ "lặp và bỏ sót" đúng nghĩa cả hai). User chọn A, duyệt 2026-07-04.** Anchor (chuỗi con) verify ×1.
- **EN old:** `repeated and omitted symbols, and verifying the small differences` → **new:** `repeated and omitted symbols, as well as verifying the small differences`
- **VI old:** `ký hiệu lặp và bỏ sót, và kiểm chứng các khác biệt nhỏ` → **new:** `ký hiệu lặp và bỏ sót, cũng như kiểm chứng các khác biệt nhỏ`
**Việc 41 — 5.4 câu 2: thay kết luận chung chung "cải thiện lớn và nhất quán" bằng con số cụ thể của HỢP NHẤT (+8,5 điểm phần trăm so với bộ giải mã đơn-phương-thức mạnh nhất, nhất quán trên cả ba tập CROHME) rồi mới kết luận. KHÔNG dùng +10,66đ vì đó là công của huấn luyện hai chiều, không phải hợp nhất. User chốt 2026-07-04.** Anchor verify ×1.
- **EN old:** `By placing fusion inside the decoder and demonstrating that this approach yields large, consistent gains, the thesis shows that this intersection is worth pursuing.` → **new:** `By placing fusion inside the decoder and demonstrating that it raises expression recognition by about 8.5 percentage points over the strongest single-modality decoder, consistently across all three CROHME test sets, the thesis shows that this intersection is worth pursuing.`
- **VI old:** `Bằng cách đưa cơ chế hợp nhất vào bên trong bộ giải mã và chứng minh rằng cách làm này mang lại cải thiện lớn và nhất quán, luận văn cho thấy điểm giao này đáng được theo đuổi.` → **new:** `Bằng cách đưa cơ chế hợp nhất vào bên trong bộ giải mã và chứng minh rằng nó nâng tỷ lệ nhận dạng biểu thức thêm khoảng 8,5 điểm phần trăm so với bộ giải mã đơn-phương-thức mạnh nhất, một cách nhất quán trên cả ba tập kiểm tra CROHME, luận văn cho thấy điểm giao này đáng được theo đuổi.`
- Khi áp (CẢ 41 MỤC một lượt): Abstract (3) + Mục 1 (5,6,25) + 2.1 (7,10,8,9,11,12) + 2.2 (13,14) + 2.3 (15,16) + 2.4 (17) + 2.5 (18) + 2.6 (19–23) + 2.7 (24,26) + Phần 4 (27–33) + 5.1 (34,35,36) + 5.2 (37,38,39) + 5.3 (40) + 5.4 (41) + nhúng Hình 1 (4) vào 2 docx → xuất lại 2 PDF (mẹo LibreOffice: outdir thư mục mới rỗng → cp -f đè). **➡ ĐÃ SOÁT XONG TOÀN BỘ Abstract→Phần 5; sẵn sàng áp một lượt khi user ra lệnh "sửa chính thức".**

---

## ⏳ CHỜ ÁP — SỬA TÀI LIỆU THAM KHẢO (rà soát 2026-07-04)

> **User chốt: GỘP CHUNG 11 sửa TLTK này vào cùng lượt áp với 41 sửa prose Abstract→Phần 5.** Tổng một lượt = 41 prose + Hình 1 + 11 TLTK → rồi xuất lại 2 PDF. Chờ lệnh "sửa chính thức".

Rà soát toàn bộ [1]–[38] đối chiếu kho `references/` (32 PDF gốc + 6 ABSTRACT Crossref) + tra Crossref/Semantic Scholar cho vol/trang chưa soi được. **Áp vào CẢ `references_ieee.txt` LẪN References trong 2 docx.** Lưu ý trình bày docx khác txt (nháy cong, bỏ DOI/tháng, viết tắt venue) → khi áp, chỉ đổi đúng trường; DOI chỉ đổi trong txt (docx phần lớn không in DOI, trừ [17]/[18] hiện có — sẽ bỏ DOI, thêm vol/trang cho khớp style).

Nhóm sai rõ (đối chiếu nguồn gốc):
- **[8]** trang: `1007–1012` → **`1243–1248`** (PDF header IFAC-PapersOnLine 53-2 (2020) 1243–1248).
- **[10]** trang: `4553–4562` → **`4543–4552`** (Crossref DOI 10.1109/CVPR52688.2022.00451).
- **[12]** volume: `128` → **`132`** (PDF: Pattern Recognition 132 (2022) 108910).
- **[14]** tác giả: `Y. Zhu, Y. Wang, and D. Liu` → **`J. Zhu, L. Gao, and W. Zhao`** (arXiv:2405.09032: Jianhua Zhu, Liangcai Gao, Wenqi Zhao).
- **[16]** tác giả: `M. Lin, S. Zhang, and Y. Tang` → **`Z. Lin, J. Li, F. Yang, S. Huang, J. Lin, M. Yang, and X. Yang`** (arXiv:2303.07077).
- **[21]** trang + DOI: `pp. 115206–115234 … doi:…3096291` → **`pp. 38352–38373 … doi:10.1109/ACCESS.2021.3063413`** (PDF gốc). *Docx: chỉ đổi trang.*
- **[24]** trang: `4700–4708` → **`2261–2269`** (user chọn theo IEEE Xplore, DOI 10.1109/CVPR.2017.243; bản CVF là 4700–4708).
- **[25]** tác giả: `W. Bo` → **`B. Wen`** (RoFormer, tác giả thứ 5 là Bo Wen). Các tác giả/thứ tự còn lại khớp bản Neurocomputing.

Nhóm bổ sung thông tin (Crossref):
- **[17]** `Pattern Recognition, 2023, doi:…110155` → **`Pattern Recognition, vol. 148, p. 110155, 2024`** (user chốt năm 2024 — số vol 148 thuộc số tháng 4/2024).
- **[18]** `Pattern Recognition, 2025, doi:…111346` → **`Pattern Recognition, vol. 162, p. 111346, 2025`**.
- **[29]** thêm **`pp. 553–565`** (đã có LNCS 14188).

Đã xác nhận ĐÚNG, không đổi: [1],[2],[3],[4],[5],[6],[7],[9],[11],[13],[15],[19],[20],[22],[23],[26],[27],[28],[30],[31],[32],[33],[34],[35],[36],[37],[38]. ([22]: kho chỉ có bản arXiv 3 tác giả, luận văn trích bản Pattern Recognition 5 tác giả — hợp lệ.)

## (lịch sử) REVIEW PHẦN 3 vs CODE (2026-07-04) — sửa ĐÃ DUYỆT + ĐÃ SOẠN, CHƯA ÁP DỤNG (user chốt: gom lại, review xong toàn bộ mới sửa MỘT LƯỢT)
User yêu cầu rà Phần 3 đối chiếu code (decoder/encoder_img/encoder_seq/bttr/lit_bttr/pos_enc/datamodule/vocab/config/custom_train). Kết luận: cốt lõi khớp code (công thức lớp giải mã tgt+CA1+CA2+FFN, pre-norm, mask, DenseNet k=24/16 lớp/nén 0,5/H16, 7 nhãn quan hệ đúng vocab, ví dụ `\sqrt Inside 4 Right \pi` đúng format crohme_all.txt). 17 phát hiện + 1 ngoài phạm vi. Khi áp dụng: từng phép verify khớp ĐÚNG 1 vị trí → sửa SVG render user duyệt → nhúng (chỉ thay blob, giữ docPr) → xuất 2 PDF vào thesis\.
> **TRẠNG THÁI 2026-07-04 (phiên tiếp):** phần "review còn lại" (3.4/3.6/3.7/3.9/3.10/3.11/3.13/4.2) nay ĐÃ soạn nguyên văn old→new — xem mục "⏳ ĐÃ SOẠN old→new — CHỜ DUYỆT" bên dưới. Đối chiếu code đã xác minh (post_norm+ReLU+684; RoPE dormant; dual_shared chung self._mha_block+norm2; custom_train hard-code không đọc config.yaml; datamodule val=test_year 2014). ⚠️ **R3.4 gộp & thay E4/V4.** (ĐÃ ÁP DỤNG 2026-07-04 — xem khối ✅ ở đầu mục.)

### ✅ ĐÃ DUYỆT — CHỜ ÁP DỤNG (mọi chuỗi old verify khớp đúng 1 vị trí/bản ngày 2026-07-04)
**(1) 3.1-a (chỉ VI):** "Hai mũi tên gộp đi vào cạnh trên" ⇒ "Hai mũi tên đi vào cạnh trên" (bỏ "gộp" nói ngược ý; EN 3.1 giữ nguyên).
**(2) GÓI KÝ HIỆU** (user chốt 2026-07-04: heads H→n_h; t không gian→m; chỉ số k ở 3.9→j; "N decoder layers"→"three"; EN Notation thêm m, l₁; VI chèn đoạn "Ký hiệu."):
- EN (docx Sections1-5):
  - E1: `d the model dimension, H the number of attention heads` ⇒ `d the model dimension, n_h the number of attention heads`
  - E2: `E_off ∈ ℝ^{t × d} where t = h·w is the number of spatial locations` ⇒ `E_off ∈ ℝ^{m × d} where m = h·w is the number of spatial locations`
  - E3: `d (model dimension); H (number of heads); k (DenseNet growth rate)` ⇒ `d (model dimension); m and l₁ (lengths of the image and sequence memories); n_h (number of heads); k (DenseNet growth rate)`
  - E4: `into the sequence E_off ∈ ℝ^{t × d} with t = h·w` ⇒ `into the sequence E_off ∈ ℝ^{m × d} with m = h·w` — ⚠️ **BỎ khi áp**: gộp vào phép R3.4 (soạn 2026-07-04, bản NEW đã chứa `m`); xem mục "ĐÃ SOẠN old→new — CHỜ DUYỆT".
  - E5: `Multi-head attention runs H such operations` ⇒ `Multi-head attention runs n_h such operations`
  - E6: `Concat( head_1 , … , head_H )` ⇒ `Concat( head_1 , … , head_{n_h} )`
  - E7: `Splitting the representation into H heads` ⇒ `Splitting the representation into n_h heads`
  - E8: `come from E_off ([2b, t, d]); the resulting attention map has shape [2b, H, L, t]` ⇒ `come from E_off ([2b, m, d]); the resulting attention map has shape [2b, n_h, L, m]`
  - E9: `giving an attention map of shape [2b, H, L, l₁]` ⇒ `giving an attention map of shape [2b, n_h, L, l₁]`
  - E10: `Σ_{k=1..t}` ⇒ `Σ_{j=1..t}` và `log p( y_k | y_<k )` ⇒ `log p( y_j | y_<j )`
  - E11: `image features t (a downsampled` ⇒ `image features m (a downsampled`
  - E12: `to yield E_off ∈ ℝ^{t × d}` ⇒ `to yield E_off ∈ ℝ^{m × d}`
  - E13: `passed through N decoder layers` ⇒ `passed through three decoder layers`
  - E14 (Bảng 2 = tables[1]): ô `Attention heads H` ⇒ `Attention heads n_h`
- VI (docx TiengViet), các phép tương ứng:
  - V1: `d là số chiều mô hình, H là số đầu chú ý` ⇒ `d là số chiều mô hình, n_h là số đầu chú ý`
  - V2: `E_off ∈ ℝ^{t × d} với t = h·w là số vị trí không gian` ⇒ `E_off ∈ ℝ^{m × d} với m = h·w là số vị trí không gian`
  - V4: `thành chuỗi E_off ∈ ℝ^{t × d} với t = h·w` ⇒ `thành chuỗi E_off ∈ ℝ^{m × d} với m = h·w` — ⚠️ **BỎ khi áp**: gộp vào phép R3.4 (bản NEW đã chứa `m`).
  - V5: `Chú ý nhiều đầu chạy H phép` ⇒ `Chú ý nhiều đầu chạy n_h phép`
  - V6: `Concat( head_1 , … , head_H )` ⇒ `Concat( head_1 , … , head_{n_h} )`
  - V7: `Chia biểu diễn thành H đầu` ⇒ `Chia biểu diễn thành n_h đầu`
  - V8: `đến từ E_off ([2b, t, d]); bản đồ chú ý có dạng [2b, H, L, t]` ⇒ `đến từ E_off ([2b, m, d]); bản đồ chú ý có dạng [2b, n_h, L, m]`
  - V9: `cho bản đồ chú ý [2b, H, L, l₁]` ⇒ `cho bản đồ chú ý [2b, n_h, L, l₁]`
  - V10: `Σ_{k=1..t}` ⇒ `Σ_{j=1..t}` và `log p( y_k | y_<k )` ⇒ `log p( y_j | y_<j )`
  - V11: `số đặc trưng ảnh t (` ⇒ `số đặc trưng ảnh m (`
  - V12: `để cho ra E_off ∈ ℝ^{t × d}` ⇒ `để cho ra E_off ∈ ℝ^{m × d}`
  - V13: `đi qua N lớp giải mã` ⇒ `đi qua ba lớp giải mã`
  - V14 (Bảng 2): ô `Số đầu chú ý H` ⇒ `Số đầu chú ý n_h`
  - V3 (CHÈN đoạn mới cuối mục 3.2 VI, sau đoạn kết thúc "…ba token đặc biệt <pad>, <sos> và <eos>.", trước heading 3.3; chữ "Ký hiệu." in đậm như "Notation." bản EN): "Ký hiệu. Để tiện tra cứu, các ký hiệu chính gồm: X_off (ảnh offline) và X_on (chuỗi online); E_off và E_on (bộ nhớ mã hóa tương ứng); y và ŷ (chuỗi token LaTeX nhãn đúng và chuỗi dự đoán); L (độ dài đích); d (số chiều mô hình); m và l₁ (độ dài bộ nhớ ảnh và bộ nhớ chuỗi); n_h (số đầu chú ý); k (tốc độ tăng trưởng DenseNet); N (số token đích hợp lệ trong một lô); α (số mũ chuẩn hóa độ dài); và β (độ rộng chùm). Chữ W hoa kèm chỉ số trên là các ma trận chiếu học được, LN là chuẩn hóa lớp. Chúng tôi dùng [·] cho phép nối theo kênh và ¬mask cho phần bù của mặt nạ đệm."
**(3) SVG kèm gói ký hiệu (làm SAU khi văn bản xong):** `[b, t, d]` ⇒ `[b, m, d]` trong 4 file: arch_diagram_en/vi.svg (Hình 3) + densenet_diagram_en/vi.svg (Hình 4). Đề xuất kèm (xác nhận lại với user lúc sửa hình): arch `[2b, l2]`⇒`[2b, L]`, `[2b, l2, |V_dec|]`⇒`[2b, L, |V_dec|]` (l1 giữ). decoder_layer svg không có nhãn dims.
**(4) MỤC 3.3 (user duyệt 2026-07-04, "đồng ý"; anchors verify ×1):**
- E15: `does not exceed a fixed threshold.` ⇒ `does not exceed a fixed threshold (3.2×10⁵ pixels in our configuration); the number of samples in a batch is additionally capped at a fixed maximum.`
- V15: `không vượt một ngưỡng cho trước.` ⇒ `không vượt một ngưỡng cho trước (3,2×10⁵ điểm ảnh trong cấu hình của chúng tôi); số mẫu mỗi lô còn bị chặn trên bởi một hằng số.`
- E16: sau `keeping peak memory roughly constant.` thêm câu: `A small number of outliers are discarded at loading time: samples whose target sequence exceeds the maximum target length, and images whose own area already exceeds the batching threshold.`
- V16: sau `giữ mức tiêu thụ bộ nhớ ổn định.` thêm câu: `Một số ít mẫu ngoại lệ bị loại ngay khi nạp dữ liệu: mẫu có chuỗi đích dài quá độ dài đích tối đa, và ảnh có diện tích tự thân vượt ngưỡng gom lô.`
- E17: sau `is mapped back to a LaTeX string.` thêm câu: `In our data, V_enc contains 111 entries and V_dec 113, including the three special tokens.`
- V17: sau `được ánh xạ ngược thành chuỗi LaTeX.` thêm câu: `Với dữ liệu của chúng tôi, V_enc gồm 111 mục và V_dec gồm 113 mục, đã tính ba token đặc biệt.`
(Con số cap = 32 mẫu/lô: sẽ nêu ở Bảng 2 khi xử lý 3.10.)

### ⏳ ĐÃ SOẠN old→new — CHỜ DUYỆT (2026-07-04, phiên tiếp — mọi anchor verify khớp ĐÚNG 1 vị trí trên docx bản 2026-07-03 18:47; nguồn code đã đối chiếu)
> Đây là các mục "review còn lại" nay ĐÃ soạn nguyên văn (trước đây chỉ liệt kê). Khi user "sửa chính thức": áp GÓI KÝ HIỆU + 3.3 + các phép R dưới đây MỘT LƯỢT rồi xuất 2 PDF. ⚠️ **R3.4 gộp và thay cho E4/V4** (bỏ E4/V4).

**R3.4 — thêm BatchNorm cuối + ReLU sau Conv1×1 + nêu 684 kênh (GỘP E4/V4, bản NEW đã có `m`).** Nguồn: encoder_img.py post_norm=BatchNorm2d(684) dòng112/139; feature_proj Conv1×1→ReLU dòng149-152; 684 = 300(sau trans2)+16·24.
- E-3.4 (EN [103], câu cuối) OLD: `Finally, the feature map is projected to the model dimension d by a 1×1 convolution, normalized with layer normalization, given a two-dimensional positional encoding, and flattened from [b, h, w, d] into the sequence E_off ∈ ℝ^{t × d} with t = h·w, ready for the decoder.` → NEW: `Finally, the 684-channel output of the last dense block is batch-normalized and then projected to the model dimension d by a 1×1 convolution followed by a ReLU; the result is normalized with layer normalization, given a two-dimensional positional encoding, and flattened from [b, h, w, d] into the sequence E_off ∈ ℝ^{m × d} with m = h·w, ready for the decoder.`
- V-3.4 (VI [102], câu cuối) OLD: `Cuối cùng, bản đồ đặc trưng được chiếu về số chiều d bằng tích chập 1×1, chuẩn hóa lớp (LayerNorm), cộng mã hóa vị trí hai chiều, rồi làm phẳng từ [b, h, w, d] thành chuỗi E_off ∈ ℝ^{t × d} với t = h·w, sẵn sàng cho bộ giải mã.` → NEW: `Cuối cùng, đầu ra 684 kênh của khối dày đặc cuối được chuẩn hóa theo lô (BatchNorm), rồi chiếu về số chiều d bằng tích chập 1×1 kèm một ReLU; kết quả được chuẩn hóa lớp (LayerNorm), cộng mã hóa vị trí hai chiều, rồi làm phẳng từ [b, h, w, d] thành chuỗi E_off ∈ ℝ^{m × d} với m = h·w, sẵn sàng cho bộ giải mã.`

**R3.6 — RoPE có cài đặt nhưng KHÔNG kích hoạt + đồng bộ VI (thiếu 2 ý so EN).** Nguồn: pos_enc.py có WordRotaryEmbed/ImageRotaryEmbed nhưng không nơi nào instantiate; decoder dùng WordPosEnc (153), encoder ảnh dùng ImgPosEnc (155).
- E-3.6 (EN [112], câu cuối) OLD: `Whether additive or rotary encoding is preferable can depend on the dataset; the architecture is agnostic to the choice, and both variants are available.` → NEW: `Whether additive or rotary encoding is preferable can depend on the dataset; the architecture is agnostic to the choice. A rotary variant is included in our implementation but was left inactive, and all experiments in this thesis use the additive sinusoidal encoding described above.`
- V-3.6 (VI [113], câu cuối) OLD: `Mô hình còn hỗ trợ biến thể mã hóa vị trí quay (RoPE) [25], chèn thông tin vị trí tương đối trực tiếp vào tích vô hướng chú ý bằng cách quay các véc-tơ truy vấn và khóa theo vị trí của chúng.` → NEW: `Mô hình còn hỗ trợ biến thể mã hóa vị trí quay (RoPE) [25]: thay vì cộng một véc-tơ vị trí vào biểu diễn token, RoPE quay các véc-tơ truy vấn và khóa theo góc tỉ lệ với vị trí, nhờ đó tích vô hướng chú ý giữa hai vị trí chỉ phụ thuộc độ lệch tương đối; cách này tích hợp thông tin vị trí tương đối trực tiếp vào phép chú ý và được ghi nhận là ngoại suy tốt sang các chuỗi dài hơn. Việc chọn mã hóa cộng hay quay có thể tùy vào bộ dữ liệu; kiến trúc không phụ thuộc lựa chọn này. Biến thể quay có trong cài đặt của chúng tôi nhưng không được kích hoạt, và mọi thí nghiệm trong luận văn dùng mã hóa sin–cosin cộng tính mô tả ở trên.`

**R3.7 — hai cross-attention DÙNG CHUNG trọng số (phát hiện #6).** Nguồn: decoder.py dual_shared dòng 55-57 cùng self._mha_block (chung W^Q/W^K/W^V/W^O) + norm2.
- (A) E-3.7A (EN, THÊM câu cuối đoạn [131], sau `The layer output is therefore x = tgt + cross-attn₁ + cross-attn₂ + FFN.`): `These two cross-attention blocks are in fact one shared sub-layer applied twice: beyond the common query LN₂(x_q), they also share their projection weights, so both modalities are read by the same attention parameters and adding the second one costs no extra parameters (Section 3.11).`
- (A) V-3.7A (VI, THÊM cuối đoạn [129], sau `…x = tgt + chú ý chéo₁ + chú ý chéo₂ + FFN.`): `Trên thực tế, hai khối chú ý chéo này là cùng một khối con dùng chung, được áp dụng hai lần: ngoài truy vấn chung LN₂(x_q), chúng còn dùng chung các trọng số chiếu, nên cả hai phương thức được đọc bởi cùng một bộ tham số chú ý và việc thêm phương thức thứ hai không làm tăng tham số của bộ giải mã (Mục 3.11).`
- (B) Sửa thuật ngữ VI: [128] caption `Hình 5. Hợp nhất hai memory trong một tầng giải mã. Output của tự chú ý…` → `Hình 5. Hợp nhất hai bộ nhớ trong một lớp giải mã. Đầu ra của tự chú ý…` (memory→bộ nhớ, tầng→lớp, Output→Đầu ra); [129] `cộng output của hai khối chú ý chéo`→`cộng đầu ra của…` và `Do đó output của lớp là`→`Do đó đầu ra của lớp là`; [130] `output của khối được cộng trở lại`→`đầu ra của khối…` và `qua các tầng.`→`qua các lớp.`; [131] `Có thể truy theo chiều của một lớp.`→`Có thể lần theo kích thước tensor của một lớp.`
- (C) ✅ CHỐT LẤY (user duyệt 2026-07-04): E-3.7C (EN [135]) OLD `two cross-attention blocks that share a query` → NEW `two cross-attention blocks that share a query and their weights`; V-3.7C (VI [133]) OLD `Việc dùng hai khối chú ý chéo chung truy vấn là điều phân biệt` → NEW `Việc dùng hai khối chú ý chéo chung truy vấn và chung trọng số là điều phân biệt`. (anchor verify ×1 mỗi bản.)

**R3.9 — Thuật toán 1 (VI) thêm dấu tiếng Việt** (từ khóa code input/init/for/return giữ nguyên). [148] `Thuat toan 1: Tim kiem chum hai chieu (minh hoa mot chieu)`→`Thuật toán 1: Tìm kiếm chùm hai chiều (minh họa một chiều)`; [149] `bo nho…do rong chum B ; do dai toi da M`→`bộ nhớ…độ rộng chùm B ; độ dài tối đa M`; [152] `for moi beam`→`for mỗi beam`; [153] `tinh p(...)`→`tính p(...)`; [154] `tao ung vien…voi moi w`→`tạo ứng viên…với mỗi w`; [155] `top-B ung vien theo score tren tat ca beam`→`top-B ứng viên theo score trên tất cả beam`; [156] `chuyen cac beam ket thuc…vao tap hoan thanh`→`chuyển các beam kết thúc…vào tập hoàn thành`; [157] `return cac gia thuyet hoan thanh`→`return các giả thuyết hoàn thành`. (E10/V10 đổi Σ_k→Σ_j ở [146] — khác đoạn, không đụng.)

**R3.10 — bỏ mệnh đề "tệp cấu hình" sai + gỡ lặp câu mở với 4.2.** Nguồn: custom_train.py hard-code hyperparam dòng40-58, KHÔNG đọc config.yaml.
- E-3.10 (EN [167]) OLD: `The model is implemented in PyTorch with the PyTorch Lightning framework, and all hyper-parameters are managed through a single configuration file.` → NEW: `The model is implemented in PyTorch with the PyTorch Lightning framework.`
- V-3.10 (VI [165]) OLD: `Mô hình được cài đặt bằng PyTorch với khung PyTorch Lightning, và mọi siêu tham số được quản lý qua một tệp cấu hình.` → NEW: `Mô hình được cài đặt bằng PyTorch với khung PyTorch Lightning.`

**R3.11 — nêu cross-attention thứ hai không thêm tham số** (THÊM câu sau `…a term proportional to L·l₁·d.`/`…tỉ lệ với L·l₁·d.`).
- E-3.11 (EN [171], thêm): `Because both cross-attention blocks are the same shared sub-layer, this second block adds computation but no parameters.`
- V-3.11 (VI [169], thêm): `Vì hai khối chú ý chéo là cùng một khối con dùng chung, khối thứ hai chỉ thêm chi phí tính toán chứ không thêm tham số.` (không đụng anchor E11 "image features t" cùng đoạn.)

**R3.13 — ❌ BỎ (user chốt 2026-07-04):** giữ nguyên tên "shared-query" ở câu định vị vs BTTR ([175]EN/[173]VI) cho gọn, không chèn "weight-sharing" (ý chung-trọng-số đã nêu ở R3.7-A + R3.7-C).

**R4.2 — gỡ lặp câu mở + minh bạch validation = CROHME 2014.** Nguồn: datamodule.py:196 val=test_year, default "2014" (174); custom_train.py EarlyStopping/ModelCheckpoint monitor="val_loss", check_val_every_n_epoch=2.
- (A) E-4.2 (EN [182]) OLD: `The model is implemented in PyTorch with the PyTorch Lightning framework; the architecture hyper-parameters are those listed in Table 2 (model dimension 256, eight attention heads, three decoder layers, DenseNet growth rate 24, dropout 0.3).` → NEW: `The architecture hyper-parameters are those listed in Table 2 (model dimension 256, eight attention heads, three decoder layers, DenseNet growth rate 24, dropout 0.3).`
- (A) V-4.2 (VI [180]) OLD: `Mô hình được cài đặt bằng PyTorch với khung PyTorch Lightning; các siêu tham số kiến trúc như liệt kê trong Bảng 2 (số chiều mô hình 256, tám đầu chú ý, ba lớp giải mã, tốc độ tăng trưởng DenseNet 24, dropout 0,3).` → NEW: `Các siêu tham số kiến trúc như liệt kê trong Bảng 2 (số chiều mô hình 256, tám đầu chú ý, ba lớp giải mã, tốc độ tăng trưởng DenseNet 24, dropout 0,3).`
- (B) E-4.2B (EN, THÊM cuối [182], sau `…length-normalization exponent alpha = 1.0 is used.`): `CROHME defines no separate development split, so the CROHME 2014 test set doubles as the validation set during training: it is evaluated every two epochs, and both early stopping and checkpoint selection are driven by the loss on it. The 2016 and 2019 test sets are never used during training or model selection and are thus fully held out.`
- (B) V-4.2B (VI, THÊM cuối [180], sau `…hệ số chuẩn hóa độ dài α = 1,0.`): `CROHME không định nghĩa một tập phát triển riêng, nên tập kiểm thử CROHME 2014 đồng thời đóng vai trò tập kiểm định trong huấn luyện: nó được đánh giá mỗi hai epoch, và cả dừng sớm lẫn việc chọn checkpoint đều dựa trên mất mát trên tập này. Hai tập kiểm thử 2016 và 2019 không hề được dùng trong huấn luyện hay khi chọn mô hình, nên hoàn toàn tách biệt (held-out).`

**R-Bảng2 — cập nhật Bảng 2 (tables[1], cả 2 docx).** Nguồn: DenseNet dropout=0,2 hard-code (encoder_img.py _Bottleneck/_SingleLayer/_Transition p=0.2); Transformer dropout=0,3 (custom_train→bttr.py encoder_seq+decoder dòng45/54); MAX_SIZE=32e4=3,2×10⁵ (datamodule.py:19, dùng cho cả ngưỡng tích diện-tích×số-mẫu lẫn lọc ảnh đơn); cap=32 (custom_train batch_size=32; datamodule `i==batch_size`); |V_enc|=108+3=111, |V_dec|=110+3=113 (vocab crohme_seq_vocab.txt 108 dòng, dictionary.txt 110 dòng, +<pad>/<sos>/<eos>).
- SỬA ô: `Dropout | 0.3` → `Dropout (Transformer / DenseNet) | 0.3 / 0.2` (VI: `Dropout | 0,3` → `Dropout (Transformer / DenseNet) | 0,3 / 0,2`).
- THÊM 3 dòng (chèn ngay SAU dòng `Maximum target length | 200` / `Độ dài đích tối đa | 200`):
  - EN: `Batching threshold (area × count) | 3.2×10⁵` · `Max samples per batch | 32` · `Vocabulary sizes |V_enc| / |V_dec| | 111 / 113`
  - VI: `Ngưỡng gom lô (diện tích × số mẫu) | 3,2×10⁵` · `Số mẫu tối đa mỗi lô | 32` · `Kích thước từ vựng |V_enc| / |V_dec| | 111 / 113`
- Lưu ý: ô `Attention heads H`→`n_h` do E14/V14 (gói ký hiệu) lo; R-Bảng2 chỉ đụng ô Dropout + 3 dòng mới → không xung đột. Bảng 2 sau: 14→17 dòng.

### ✅ SOÁT NGHĨA TOÀN PHẦN 3 (3.1–3.13, EN+VI) — 2026-07-04
Đã đọc kỹ từng câu 3.1→3.13 cả 2 bản. Kết luận: **văn xuôi rõ nghĩa, không phát sinh phép sửa mới** ngoài các phép đã soạn. Cụ thể đã xác minh:
- Câu 3.5 [105] (embed→LayerNorm→sin-cosin PE→stack self-attn + key-padding mask) KHỚP encoder_seq.py.
- EN/VI 3.8 khớp nhau (câu "…mutual learning in ABM [26]"/"…học tương hỗ… ABM [26]" có ở CẢ hai).
- VI [79] `V_dec cho đầu ra LaTeX` KHÔNG lỗi mã hóa (kiểm byte docx: không có U+FFFD; chỉ là artifact terminal).
- Ẩn số H bị nạp chồng (heads vs chiều cao ảnh H·W/H,W): gói ký hiệu H→n_h giải phóng H về chỉ còn nghĩa "chiều cao ảnh" → hết mập mờ.
- Các chỗ tối nghĩa/lẫn tiếng Anh của VI đều nằm trong R3.7(B) (memory→bộ nhớ, output→đầu ra, tầng→lớp, "truy theo chiều"→"lần theo kích thước tensor") — đã soạn.

### ✅ MỤC TÙY CHỌN — ĐÃ CHỐT BỎ CẢ 3 (user chốt 2026-07-04)
- 3.9 chú thích live-beam giảm dần: BỎ (hành vi beam search tiêu chuẩn, không phục vụ luận điểm).
- 3.11 đếm tham số thật từ ckpt: BỎ (không câu nào dùng con số này).
- 3.7.1 câu "cascaded cộng cả self-attn vào residual": BỎ (mô tả hiện tại đã đúng; khác biệt phần-dư chỉ là hệ quả kế thừa nn.TransformerDecoderLayer, không phải dụng ý — nêu ra dễ gây hiểu nhầm).

### ✅ ĐÃ ĐỒNG BỘ 2026-07-13 (trước là CÒN MỞ, ngoài phạm vi chốt Phần 3)
- KIEN_TRUC_HE_THONG.md + .docx: đã đồng bộ ký hiệu sơ đồ theo gói ký hiệu luận văn — `t→m` (độ dài bộ nhớ ảnh: `[b,t,d]→[b,m,d]`, `t=h·w→m=h·w`) và `l2→L` (độ dài đích: `[2b,l2]→[2b,L]`, `[2b,l2,|V_dec|]→[2b,L,|V_dec|]`); giữ nguyên `l1` (độ dài chuỗi online) và `t` chỉ-số-bước ở beam search + `l2r` (chiều đọc L2R). .md 4 phép, .docx 1 phép (`t=h·w→m=h·w`), mỗi phép khớp đúng 1 vị trí. Backup: `_backup_f1_20260713_233431/KIEN_TRUC_HE_THONG.{md,docx}`.

## File thành phẩm — nơi lưu DUY NHẤT: `C:\Users\Admin\lv\lvtn\thesis\`
**⚠️ QUY TẮC MỚI (user chốt 2026-07-02): KHÔNG đồng bộ sang C:\Users\Admin\Downloads nữa** — mọi cập nhật chỉ lưu vào `thesis\` để file + code tập trung một chỗ. Các bản copy hiện có trong Downloads là snapshot cũ (lần sync cuối 2026-07-02), coi như lỗi thời; user chưa yêu cầu xóa.
**⚠️ TÁI TỔ CHỨC THƯ MỤC 2026-07-02 (theo yêu cầu user):** mọi file luận văn giờ nằm trong **`thesis\`** (2 docx + 2 pdf + references_ieee.txt + references.bib + KIEN_TRUC_HE_THONG.* + ABLATION_GUIDE.md), nguồn hình trong **`thesis\figures\`** (24 file svg+png), kết quả thí nghiệm trong **`results\`** (24 file: abl_*×18, seed7_*×3, test_*×3). Code/data/config GIỮ NGUYÊN ở gốc (data.zip, crohme_all.txt, best.ckpt do code tham chiếu; command.txt = ghi chú lệnh của user). Đã sửa 3 đường dẫn trong `demo/build/{build_data,extract_images,gen_seed7}.py` sang `results/...` (chạy từ gốc dự án như cũ). → Xuất PDF/sync từ nay dùng đường dẫn `C:\Users\Admin\lv\lvtn\thesis\Multi-modal_HMER_*.docx`.
- `Multi-modal_HMER_Sections1-5_full.docx` — bản TIẾNG ANH hoàn chỉnh (Abstract→Phần 5+References), ~38–40 trang, 6 hình, validate PASSED.
- `Multi-modal_HMER_TiengViet_full.docx` — bản TIẾNG VIỆT hoàn chỉnh (dịch toàn bộ; hình giữ nhãn tiếng Anh + chú thích Việt; references giữ tiếng Anh).
- PDF (xuất 2026-06-22 bằng Word COM): `Multi-modal_HMER_Sections1-5_full.pdf` (~43 trang) và `Multi-modal_HMER_TiengViet_full.pdf` (~40 trang). Có bookmark theo heading.
- Hình: `figure1_example.png` (√(4π) 3 biểu diễn), `figure2_taxonomy.png`, `arch_diagram_en.png` (Fig3), `densenet_diagram_en.png` (Fig4), `decoder_layer_en.png` (Fig5), `figure6_fusion_variants.png` (5 kiểu fusion).
- `thesis\references_ieee.txt` (danh mục [1]-[38], đã xác minh online — file tham khảo DUY NHẤT), `thesis\ABLATION_GUIDE.md`, `thesis\KIEN_TRUC_HE_THONG.docx/.md`, `command.txt` ở gốc (ghi chú lệnh chạy của user — giữ). (ĐÃ XÓA: `ABSTRACT_draft.txt` 2026-06-24; `SECTION5_draft.txt`, `_arch_diagram.png`, `_densenet_diagram.png`, `_chrome.log` 2026-07-02; `references.bib` 2026-07-02 — lỗi thời, user xác nhận xóa.)
- Kết quả test thật: `abl_offline_*`, `abl_online_*`, `abl_concat_*`, `abl_cascaded_*`, `abl_uni_*`, `abl_dual_shared_*` (`*`=2014/2016/2019)`_results.txt`.

## Kết quả ablation cuối (ExpRate %, seed 7, đánh giá 985/1147/1198 mẫu)
| Cấu hình | 2014 | 2016 | 2019 | micro |
|---|---|---|---|---|
| offline-only | 47.61 | 47.78 | 45.74 | 47.00 |
| online-only | 67.72 | 64.69 | 65.03 | 65.71 |
| concat (early fusion) | 77.66 | 73.32 | 76.88 | 75.89 |
| cascaded | 74.62 | 71.14 | 74.87 | 73.51 |
| shared-query (Ours, seed7) | 77.06 | 72.62 | 73.29 | 74.17 |
| unidirectional (dual_shared,L2R) | 64.67 | 62.69 | 63.36 | 63.51 |
(best.ckpt cũ = 74.05 micro, tái lập Ours.) Fusion vượt online-only ~+8.5đ; hai chiều vượt một chiều +10.66đ. Quan trọng: **concat ≥ shared-query** (~+1.7đ) → KHÔNG tuyên bố shared-query tốt nhất.

## Quyết định khung trình bày (đã chốt với user)
- **Phương án A**: KHÔNG đối đầu trực tiếp với SOTA ngoài ở phần Kết quả; so sánh chính = ablation nội bộ; SOTA chỉ đặt bối cảnh ở Bảng 1 (Related Works).
- **Cách 1**: đóng góp = "fusion ở tầng decoder + huấn luyện hai chiều" (2 phát hiện mạnh); fusion-design trình bày trung thực là ablation (concat nhỉnh nhất, các kiểu tương đương); giữ shared-query làm cấu hình chính (giữ 2 nguồn tách biệt). Đã sửa đóng góp #2 và Mục 3.7.1 cho khớp.
- **⚠️ Caveat cốt lõi**: nhánh online dùng SRT GROUND-TRUTH (oracle) → số là cận trên, không so offline-only. Đã nêu rõ ở Mục 3.3.2 và 4.4.
- Số "Ours" dùng seed-7 (77.06/72.62/73.29, micro 74.17) xuyên suốt Abstract/Bảng 3/4/5/Phần 5.

## Bảng 1 (Related Works) — ExpRate đã điền (có nguồn): SAN 56.20/53.60/53.50, PosFormer 62.68/61.03/64.97, BTTR 53.96/52.31/52.96 (bảng PosFormer); GETD 63.45/60.42/61.05 (snippet, CẦN đối chiếu bản gốc — chưa xong); Yang/Truong/SCAN từ related works. **VLPG 60.41/60.51/62.34 ĐÃ XÁC MINH 2026-07-02** khớp repo chính thức tác giả github.com/guohy17/VLPG (repo ghi ±0.36/±0.79/±0.30). Còn phải đối chiếu bản gốc: [11] BPD (số từ bảng thứ cấp MMHMER) + [17] GETD. **GETD 63.45/60.42/61.05: nguồn = snippet ban đầu, CHƯA truy được. Đã kiểm tra survey [28] (2026-07-02): hàng GETD trong bảng để TRỐNG, 3 số này KHÔNG xuất hiện trong [28] → [28] không dùng làm nguồn được. Cần PDF gốc [17] (DOI 10.1016/j.patcog.2023.110155) để xác nhận; nếu không có, cân nhắc để "—".** references/28.pdf đã có (user bổ sung).

### ✅ ĐÃ LÀM 2026-07-02 (user lệnh "chỉnh sửa và xuất file") — gói 3 việc dưới đây ĐÃ áp dụng vào 2 docx + xuất 2 PDF vào thesis\ (EN 1.073.424 B, VI 1.256.528 B), KHÔNG sync Downloads. Verify PASS trên docx: câu 3.10 mới có/câu cũ hết, GETD = "—", Bảng 2 = 14 dòng có dòng encoder. Lưu ý: pdftotext không trích được chuỗi tiếng Việt có dấu từ PDF (lỗi extraction đã biết) — verify chuẩn phải đọc docx. Chi tiết gói (hồ sơ):
- **Đổi ô ExpRate của GETD [17] trong Bảng 1 thành "—"** (cả EN + VI). Hiện: EN dòng "GETD [17]" ô ExpRate = "63.45 / 60.42 / 61.05"; VI = "63,45 / 60,42 / 61,05" → đổi CẢ Ô thành "—" (em dash U+2014, trùng kiểu dấu ô SCAN 2019). Lý do: số chưa truy được nguồn, [28] không có. Dùng dash khớp caption "'—' = not reported".
  - Đã xác nhận: THÂN BÀI không có số GETD theo năm (chỉ Bảng 1) → chỉ sửa 1 ô/bản.
  - Bảng 1 = tables[0], tìm row cell[0].startswith("GETD"), set cell[4] = "—" (giữ format: set run[0].text, xóa run thừa).
  - SAU KHI SỬA: xuất lại 2 PDF (Start-Job pattern, kill WINWORD trước, docPr đã unique nên không treo) + đồng bộ 4 file sang C:\Users\Admin\Downloads.
  - Khi nào user tìm được 17.pdf thì đọc bảng gốc, điền lại số thật thay cho "—".
- **Sửa câu scheduler Mục 3.10 theo Phương án C** (user chốt 2026-07-02, sau khi phân tích mode="max" quirk — KHÔNG sửa code):
  - VI old (nguyên văn, P160 dump): "Tốc độ học còn được điều chỉnh bởi bộ lập lịch ReduceLROnPlateau, nhân với 0,1 mỗi khi mất mát kiểm định ngừng cải thiện sau một số epoch, cho hội tụ ổn định và tránh dao động cuối quá trình." → VI new: "Tốc độ học còn được điều chỉnh bởi bộ lập lịch ReduceLROnPlateau với hệ số giảm 0,1."
  - EN: chưa có nguyên văn — lúc áp dụng grep "ReduceLROnPlateau" trong thân bài EN (sẽ có ≥2 chỗ: Mục 3.10 + Mục 4.2; chỗ 4.2 "Adadelta ... and a ReduceLROnPlateau schedule" đã trung tính, GIỮ NGUYÊN); chỉ thay câu 3.10 (câu chứa mệnh đề "whenever/stops improving/0.1") → EN new: "The learning rate is further adjusted by a ReduceLROnPlateau scheduler with a decay factor of 0.1." Bảng 2 ("ReduceLROnPlateau (factor = 0.1)") đã trung tính, giữ nguyên.
- **Thêm dòng số lớp encoder chuỗi vào Bảng 2** (user chốt 2026-07-02): tables[1] cả 2 docx, chèn TRƯỚC dòng "Decoder layers | 3" (VI: dòng nhãn chứa "giải mã"): EN "Sequence-encoder layers (online branch) | 3"; VI "Số lớp bộ mã hóa chuỗi (nhánh online) | 3".
- CẢ BA việc (GETD "—" + câu 3.10 + dòng Bảng 2) làm chung MỘT lần khi user ra lệnh → xuất 2 PDF (Start-Job) vào `thesis\`. KHÔNG sync Downloads (quy tắc mới 2026-07-02). Caption đã ghi "không so sánh trực tiếp".
- **BPD [11] ĐÃ ĐIỀN (2026-06-22): 60.65 / 58.50 / 61.47** (EN; VI dùng phẩy). Nguồn thứ cấp = bảng MMHMER (arXiv 2502.05557v3), dòng "BPD-Coverage" (mô hình đầy đủ, có scale-aug); không truy cập được bản gốc ScienceDirect (403/paywall). Nhất quán với cách bảng đã dùng số thứ cấp cho dòng khác; caption P60 đã disclaim chuyện augmentation/ensemble.
- **Dòng "Ours" ĐÃ BỎ khỏi Bảng 1 (2026-06-24, CHỈ bản EN)** theo yêu cầu user: số Ours là cận trên oracle (dùng SRT ground-truth) nên KHÔNG so trực tiếp với method ngoài → Bảng 1 giờ chỉ liệt kê 9 method ngoài; số Ours báo ở Bảng 3 + ablation. Caption `"—" = not reported here` giờ đúng (chỉ còn SCAN-2019 "—"). **Bản VI vẫn còn dòng Ours** trong Bảng 1 — chưa đồng bộ.
- **Citation [11] đã sửa**: Crossref xác nhận Pattern Recognition **vol. 149, art. 110220, 2024, DOI 10.1016/j.patcog.2023.110220** (trước ghi sai vol.146/109917). Đã sửa ở `references_ieee.txt` + 2 docx + 2 PDF. (references.bib KHÔNG có entry [11] — chỉ chứa [23]-[30],[4].) Lưu ý: ref [20] Ung et al. đúng là PRL vol.146 — đừng nhầm.

## Thuật ngữ (rà soát 2026-06-22) — ĐÃ ĐỒNG NHẤT
Bản VI rất nhất quán: online/offline giữ tiếng Anh đồng đều (không lẫn "trực tuyến/ngoại tuyến"); bộ giải mã/bộ mã hóa/phương thức/hai chiều/tự chú ý/tìm kiếm chùm/suy luận/huấn luyện/đa phương thức/DenseNet/mô-đun/"tỉ lệ" đều đồng nhất. Các chỗ lẫn tiếng Anh đều hợp lệ (tên paper trong References, tên kiến trúc riêng "(shared-query) dual cross-attention", tính từ "ground-truth"). Số kết quả VI dùng dấu phẩy; "3.x" là số mục (đúng). **Đã sửa duy nhất 1 biến thể**: "biểu thức toán" → "biểu thức toán học" (10 chỗ thân bài) cho khớp tiêu đề/abstract. Bản EN nhất quán ("off-line"/"math expression" chỉ trong tên paper).

## Code đã sửa (cho ablation) — đã test (env conda `bttr`)
- `bttr/model/decoder.py`, `bttr/model/bttr.py`, `bttr/lit_bttr.py`: thêm 2 công tắc `fusion` (dual_shared|offline|online|concat|cascaded) + `bidirectional` (True/False). Mặc định dual_shared+True = Ours, không đổi hành vi gốc.
- `custom_train.py`: `seed_everything(7)`; FUSION/BIDIRECTIONAL ở đầu __main__; mỗi biến thể lưu `lightning_logs/abl_<run_name>/` (run_name=fusion, +"_uni" nếu bidirectional=False).
- `config.yaml`: thêm fusion, bidirectional, num_encoder_layers:3.
- Eval: sửa cuối `test_all.py` (loop 3 năm, trỏ CKPT) — cần `if __name__=='__main__'` (Windows spawn). Chạy: `conda run -n bttr python <script>`. Best ckpt = val_loss thấp nhất trong thư mục biến thể.

## ✅ ĐÃ ÁP DỤNG — GÓI MỞ RỘNG PHẦN 2 (user lệnh "sửa chính thức" 2026-07-02, áp dụng + verify xong cùng ngày)
Toàn bộ mục A–G bên dưới ĐÃ ghi vào 2 docx + references_ieee.txt (header → [1]-[38]); Hình 2 mới (user đã duyệt render trước) nhúng shapes[1]; 2 PDF xuất lại (EN 1.073.375 B, VI 1.256.459 B); 4 file đồng bộ Downloads 2026-07-02. Verify PASS: cross-ref "Mục/Section 2.4–2.6" = 0 trước khi đánh số lại; mọi phép thay khớp đúng 1 vị trí; heading 2.1–2.7 mỗi cái ×1; thứ tự 2.3 (mở đầu→TAP→SRD→Ung) + 2.4 (MAN→SCAN→PAL/Le) đúng; Bảng 1 = 12 dòng (TAP, MAN/E-MAN trước SCAN); pdftotext xác nhận "2.4. Multi-modal methods", "2.7. Summary", 48.47/44.81, 54.05/50.56 (VI dấu phẩy), [33]/[38] trong cả 2 PDF. Cấu trúc Phần 2 hiện hành: 2.1 nền tảng · 2.2 offline · 2.3 online · 2.4 đa phương thức · 2.5 Transformer+hai chiều · 2.6 thảo luận/khoảng trống · 2.7 tổng kết (Bảng 1).
Chi tiết gói đã áp dụng (giữ làm hồ sơ):

### A. Tài liệu tham khảo mới [33]–[38] (append sau [32]; docx không DOI, references_ieee.txt CÓ DOI; header txt → [1]-[38])
- [33] J. Zhang, J. Du, and L. Dai, "Track, Attend, and Parse (TAP): An end-to-end framework for online handwritten mathematical expression recognition," IEEE Transactions on Multimedia, vol. 21, no. 1, pp. 221–233, 2019. doi: 10.1109/TMM.2018.2844689
- [34] J. Zhang, J. Du, Y. Yang, Y.-Z. Song, and L. Dai, "SRD: A tree structure based decoder for online handwritten mathematical expression recognition," IEEE Transactions on Multimedia, vol. 23, pp. 2471–2480, 2021. doi: 10.1109/TMM.2020.3011316
- [35] J. Wang, J. Du, J. Zhang, and Z.-R. Wang, "Multi-modal attention network for handwritten mathematical expression recognition," in Proc. Int. Conf. Document Analysis and Recognition (ICDAR), 2019, pp. 1181–1186. doi: 10.1109/ICDAR.2019.00191
- [36] J.-W. Wu, F. Yin, Y.-M. Zhang, X.-Y. Zhang, and C.-L. Liu, "Handwritten mathematical expression recognition via paired adversarial learning," International Journal of Computer Vision, vol. 128, pp. 2386–2401, 2020. doi: 10.1007/s11263-020-01291-5
- [37] A. D. Le, "Recognizing handwritten mathematical expressions via paired dual loss attention network and printed mathematical expressions," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2020, pp. 2413–2418. doi: 10.1109/CVPRW50498.2020.00291
- [38] J. Zhang, J. Du, S. Zhang, D. Liu, Y. Hu, J. Hu, S. Wei, and L. Dai, "Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition," Pattern Recognition, vol. 71, pp. 196–206, 2017. doi: 10.1016/j.patcog.2017.06.017
(Số TAP/E-MAN lấy từ bảng bài SCAN arXiv 2002.08670, khớp survey [28]: TAP 48.47/44.81; E-MAN 54.05/50.56. Survey [28] ghi SAI tác giả MAN — dùng Crossref. MMHMER 2502.05557 ĐÃ LOẠI: là multi-viewer offline, không phải online+offline.)

### B. Cấu trúc Phần 2 (phương án TÁCH)
- 2.3 GIỮ tên "2.3. Online methods" / "2.3. Các phương pháp online". Thứ tự đoạn: mở đầu (giữ) → TAP (MỚI) → SRD (MỚI) → Ung (giữ).
- Chèn heading MỚI sau đoạn Ung: EN "2.4. Multi-modal methods" / VI "2.4. Các phương pháp đa phương thức" (clone format heading 2.3). Nội dung 2.4: MAN (MỚI) → SCAN (đoạn cũ DI CHUYỂN NGUYÊN VĂN từ 2.3, chỉ vá 2 ngoặc mục B4) → PAL-v2/Le (MỚI).
- Đánh số lại heading (match prefix): "2.4. "→"2.5. " (Transformer; VI "2.4. Bộ giải mã dựa trên Transformer và huấn luyện hai chiều"), "2.5. "→"2.6. " (VI "2.5. Thảo luận và khoảng trống nghiên cứu"), "2.6. "→"2.7. " (VI "2.6. Tổng kết"). TRƯỚC KHI GHI: grep "Mục 2.4|Mục 2.5|Mục 2.6|Section 2.4|Section 2.5|Section 2.6" cả 2 docx — dump 2026-07-02 không thấy cross-ref nào, xác nhận lại lần cuối.

### B1 TAP (đoạn mới, sau mở đầu 2.3)
EN: "Among end-to-end approaches, Zhang et al. [33] introduced TAP (Track, Attend, and Parse), a framework that translates the pen-trajectory point sequence directly into LaTeX. The encoder processes the point sequence with stacked bidirectional GRUs, and the decoder generates tokens with a hybrid attention mechanism equipped with coverage, using symbol-level information to guide the attention during training. TAP established the encoder–decoder paradigm for online HMER and remains the standard point-level baseline against which later stroke-level models are measured."
VI: "Trong các hướng đầu-cuối, Zhang và cộng sự [33] giới thiệu TAP (Track, Attend, and Parse), một khung dịch thẳng chuỗi điểm quỹ đạo bút thành LaTeX. Bộ mã hóa xử lý chuỗi điểm bằng các tầng GRU hai chiều xếp chồng, còn bộ giải mã sinh token bằng cơ chế chú ý lai có độ phủ, đồng thời dùng thông tin mức ký hiệu để dẫn hướng chú ý trong huấn luyện. TAP xác lập khuôn mẫu mã hóa–giải mã cho HMER online và vẫn là baseline cấp điểm chuẩn để các mô hình cấp nét sau này so sánh."

### B2 SRD (đoạn mới, sau TAP)
EN: "Later work replaced the flat LaTeX target with an explicitly structured one: Zhang et al. [34] proposed SRD, a tree-structure based decoder that generates the symbol-relation tree (SRT) of an online expression as a sequence of parent–child subtrees. Making the two-dimensional structure explicit in the output brings the robustness of tree decoding to the online setting [28]. This line is directly relevant to our work: the symbol-relation representation that SRD produces as output is precisely the kind of structural sequence our online branch consumes as input (Section 3.3.2)."
VI: "Các nghiên cứu sau đó thay đích LaTeX phẳng bằng một đích có cấu trúc tường minh: Zhang và cộng sự [34] đề xuất SRD, một bộ giải mã dựa trên cấu trúc cây sinh ra cây quan hệ ký hiệu (SRT) của biểu thức online dưới dạng một chuỗi các cây con cha–con. Việc làm tường minh cấu trúc hai chiều ở đầu ra mang lại cho bài toán online tính bền vững của giải mã cây [28]. Hướng này liên quan trực tiếp đến luận văn: biểu diễn quan hệ ký hiệu mà SRD sinh ra ở đầu ra chính là loại chuỗi cấu trúc mà nhánh online của chúng tôi dùng làm đầu vào (Mục 3.3.2)."

### B3 MAN (đoạn mới, mở đầu mục 2.4)
EN: "Bridging the two modalities, Wang et al. [35] proposed MAN (Multi-modal Attention Network), which feeds the dynamic trajectory and the static image into online and offline channels of a multi-modal encoder and lets a multi-modal decoder attend to both feature streams while generating LaTeX; an enhanced variant, E-MAN, adds a re-attention mechanism. E-MAN reaches 54.05% on CROHME 2014 and 50.56% on CROHME 2016, outperforming its single-modality counterparts and providing early end-to-end evidence that the two input forms are complementary in practice."
VI: "Bắc cầu giữa hai phương thức, Wang và cộng sự [35] đề xuất MAN (Multi-modal Attention Network): quỹ đạo động và ảnh tĩnh được đưa vào hai kênh online và offline của một bộ mã hóa đa phương thức, rồi một bộ giải mã đa phương thức chú ý tới cả hai luồng đặc trưng trong khi sinh LaTeX; biến thể tăng cường E-MAN bổ sung cơ chế tái chú ý (re-attention). E-MAN đạt 54,05% trên CROHME 2014 và 50,56% trên CROHME 2016, vượt các phiên bản đơn-phương-thức tương ứng — bằng chứng đầu-cuối sớm cho thấy hai dạng đầu vào bổ trợ nhau trong thực tế."

### B4 Vá 2 ngoặc trong đoạn SCAN (đoạn di chuyển sang 2.4, câu giữ nguyên)
EN: "the point-level baseline (TAP)" → "the point-level baseline (TAP [33])"; "the pixel-level WAP" → "the pixel-level WAP [38]".
VI: "baseline cấp điểm (TAP)" → "baseline cấp điểm (TAP [33])"; "thấp hơn WAP cấp điểm ảnh" → "thấp hơn WAP [38] cấp điểm ảnh".

### B5 PAL-v2 + Le (đoạn mới, sau SCAN trong 2.4)
EN: "Multi-modality in HMER is not limited to pairing online and offline signals. A separate line pairs handwritten input with printed (rendered) mathematical expressions: Wu et al. [36] train the recognizer with paired adversarial learning so that it extracts semantic-invariant features shared by handwritten and printed forms of the same expression, and Le [37] couples a paired dual-loss attention network with printed expressions and an existing LaTeX corpus to regularize both the encoder and the decoder. These works treat the printed form as an auxiliary source of clean, style-free supervision — a perspective complementary to the online/offline fusion pursued in this thesis."
VI: "Tính đa phương thức trong HMER không chỉ giới hạn ở việc ghép tín hiệu online với offline. Một hướng riêng ghép đầu vào viết tay với biểu thức toán học in (kết xuất): Wu và cộng sự [36] huấn luyện bộ nhận dạng bằng học đối kháng theo cặp để trích các đặc trưng bất biến ngữ nghĩa chung giữa dạng viết tay và dạng in của cùng một biểu thức, còn Le [37] kết hợp mạng chú ý dual-loss theo cặp với biểu thức in và một kho ngữ liệu LaTeX sẵn có để điều chuẩn cả bộ mã hóa lẫn bộ giải mã. Các công trình này xem dạng in như một nguồn giám sát phụ trợ sạch, không phụ thuộc nét chữ — một góc nhìn bổ trợ cho hướng hợp nhất online/offline mà luận văn theo đuổi."

### C. Ba câu nối (old→new, EN+VI)
- C1 P23: EN "then reviews representative offline, online, and Transformer-based methods" → "then reviews representative offline, online, multi-modal, and Transformer-based methods". VI "điểm lại các phương pháp offline, online và dựa trên Transformer tiêu biểu" → "điểm lại các phương pháp offline, online, đa phương thức và dựa trên Transformer tiêu biểu".
- C2 (2.5→sẽ là 2.6 Thảo luận): EN "Multi-modal recognition has emerged to combine these strengths, and SCAN [22] demonstrates clear gains from fusing stroke-, point-, and pixel-level features." → "Multi-modal recognition has emerged to combine these strengths: MAN [35] fuses the two modalities within an RNN-based encoder–decoder, and SCAN [22] demonstrates clear gains from fusing stroke-, point-, and pixel-level features." VI "Nhận dạng đa phương thức đã xuất hiện để kết hợp các điểm mạnh này, và SCAN [22] cho thấy lợi ích rõ rệt từ việc hợp nhất đặc trưng cấp nét, cấp điểm và cấp điểm ảnh." → "Nhận dạng đa phương thức đã xuất hiện để kết hợp các điểm mạnh này: MAN [35] hợp nhất hai phương thức trong một khung mã hóa–giải mã dựa trên RNN, và SCAN [22] cho thấy lợi ích rõ rệt từ việc hợp nhất đặc trưng cấp nét, cấp điểm và cấp điểm ảnh."
- C3 (2.6→sẽ là 2.7 Tổng kết): EN "with SCAN [22] as a representative example" → "with MAN [35] and SCAN [22] as representative examples". VI "với SCAN [22] là ví dụ tiêu biểu" → "với MAN [35] và SCAN [22] là những ví dụ tiêu biểu".

### D. Bảng 1 (tables[0], chèn 2 dòng TRƯỚC dòng "SCAN / MMSCAN-E [22]" = row index 8)
EN row1: TAP [33] | Online | GRU (point-level) | RNN (hybrid attention) | 48.47 / 44.81 / —
EN row2: MAN / E-MAN [35] | Multi-modal | GRU / CNN | RNN (multi-modal attention) | 54.05 / 50.56 / —
VI row1: TAP [33] | Online | GRU (cấp điểm) | RNN (chú ý lai) | 48,47 / 44,81 / —
VI row2: MAN / E-MAN [35] | Đa phương thức | GRU / CNN | RNN (chú ý đa phương thức) | 54,05 / 50,56 / —
Caption: EN câu 2 "The list is offline-dominated because the online works reviewed are surveys or auxiliary tasks without a comparable ExpRate; SCAN [22] is the online/multi-modal representative." → "TAP [33], MAN [35], and SCAN [22] represent the online and multi-modal lines; the remaining entries are offline methods." VI câu 2 "Danh sách nghiêng về các phương pháp offline vì các công trình online được điểm lại là các bài tổng quan hoặc nhiệm vụ phụ trợ không có ExpRate so sánh được; SCAN [22] là đại diện cho hướng online/đa phương thức." → "TAP [33], MAN [35] và SCAN [22] đại diện cho các hướng online và đa phương thức; các mục còn lại là phương pháp offline."

### E. Hình 2 (figure2_taxonomy.svg + _vi.svg → render Chrome ×3 → GỬI USER XEM → nhúng inline_shapes[1] 2 docx)
1. Ô Stroke/trajectory (VI Nét / quỹ đạo): refs "[21] [22]" → "[21] [22] [33] [34]".
2. Ô RNN-based fusion (VI Hợp nhất dựa trên RNN): sub "SCAN [22]" → "MAN [35] · SCAN [22]".
3. Cột Multi-modal thêm ô thứ 3 "Handwritten + printed" / VI "Viết tay + bản in", sub "[36] [37]"; thứ tự cột trên→dưới: Viết tay + bản in (y≈212) → Hợp nhất RNN (y≈290) → Transformer/Ours (y≈368, giữ ô đậm); nối line dọc như cột Offline 3 ô.

### F. F1 + F2 (đều làm)
- F1 (Mục 3.3.2): chuỗi "[11], [12], [13], [16]." (duy nhất, chung EN+VI) → "[11], [12], [13], [16], [34]."
- F2 (Mục 5.3): EN "such as HME100K," → "such as HME100K [10],"; VI "như HME100K," → "như HME100K [10],".

## ✅ ĐÃ ÁP DỤNG "sửa chính thức" (2026-07-01) — 27 mục, EN+VI, đã verify
Tất cả 27 mục dưới đây ĐÃ áp dụng vào CẢ 2 docx (EN 33 phép, VI 32 phép kể cả caption Hình 6), + thêm [31][32] vào `references_ieee.txt` và References của 2 docx, + đổi nhãn Hình 2 & Hình 6 (render lại SVG→PNG, re-embed inline_shapes[1]&[5]). Đã xuất lại 2 PDF (Word COM) + đồng bộ 4 file sang `C:\Users\Admin\Downloads` (23:39 2026-07-01). Verify PASS: docPr id duy nhất 1–6; [31][32] có; số 75.89/2.4đ có; câu 4.8 "A limitation…"/"Một hạn chế…" đã xóa; caption Hình 6 EN/VI mới.
- **Mục #7 caption Bảng 1 VI (user chốt 2026-07-01)**: "Bảng 1. Các phương pháp HMER tiêu biểu có báo cáo tỉ lệ nhận dạng biểu thức trên CROHME (theo công bố của từng bài; "—" = không báo cáo). Danh sách nghiêng về các phương pháp offline vì các công trình online được điểm lại là các bài tổng quan hoặc nhiệm vụ phụ trợ không có ExpRate so sánh được; SCAN [22] là đại diện cho hướng online/đa phương thức."
- **⚠️ BÀI HỌC xuất PDF (2026-07-01)**: (1) Word COM `ExportAsFixedFormat` với CreateBookmarks=1 sẽ **TREO VÔ HẠN** nếu docx có **`wp:docPr id` trùng** (5 hình đều id=1 sau các lần re-embed python-docx) → phải renumber id duy nhất (script `_fix_docpr.py` kiểu cũ, đã xóa). (2) Trong môi trường này, gọi Word COM **trực tiếp foreground bị treo**; phải chạy qua **`Start-Job` + `Wait-Job -Timeout`** (mỗi lần 1 file, kill WINWORD trước). Mở/ComputeStatistics/Export-không-bookmark thì KHÔNG treo.

(Lịch sử) Áp dụng CẢ EN (`Multi-modal_HMER_Sections1-5_full.docx`) + VI (`Multi-modal_HMER_TiengViet_full.docx`); P-number EN/VI khác nhau → định vị bằng nội dung (grep). Danh mục references KHÔNG đánh số theo thứ tự xuất hiện → chỉ append [31][32], KHÔNG renumber.
1. **P11 thách thức HMER** (refs có sẵn): câu mở "harder than ordinary text" +[19],[28]; "two-dimensional…linear order" +[28],[10]; "segmentation difficult" +[28]; "scale-aware" +[28]; "long-range context" +[23],[27]; "end-to-end…jointly" +[28],[23].
2. **P13**: câu "inherently sequential computation also limits parallelization and slows training" / "Tính toán tuần tự…hạn chế song song hóa…" → +[7] cuối câu.
3. **P14**: câu "most contemporary systems cast HMER as seq2seq…LaTeX" / "đa số…dịch chuỗi-sang-chuỗi…LaTeX" → +[28].
4. **Hình 2** (figure2_taxonomy): EN "Transformer decoder"→"Transformer-based"; VI "Bộ giải mã Transformer"→"Dựa trên Transformer". Sửa `figure2_taxonomy.svg` + `figure2_taxonomy_vi.svg` → render PNG (Chrome, 960×512, scale 2) → thay inline_shapes[1] cả 2 docx.
5. **P31** câu mở: EN→"Within this encoder–decoder paradigm, two broad families of decoders have been explored: string decoders and tree decoders." VI→"Trong khung mã hóa–giải mã này, hai họ bộ giải mã đã được khảo sát: bộ giải mã chuỗi và bộ giải mã cây."
6. **Đoạn fusion taxonomy (early/late/decoder-level)** + 2 REF MỚI: [31] T. Baltrušaitis, C. Ahuja, L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," IEEE TPAMI 41(2):423-443, 2019, doi:10.1109/TPAMI.2018.2798607. [32] P. Xu, X. Zhu, D. A. Clifton, "Multimodal learning with transformers: A survey," IEEE TPAMI, 2023, doi:10.1109/TPAMI.2023.3275156. Đặt: câu giới thiệu sớm/muộn/decoder-level →[31],[32]; câu "early làm nhòe / late bỏ tương tác chéo" →[31]. PHẢI thêm 2 entry vào `references_ieee.txt` + danh mục References trong CẢ 2 docx.
7. **Caption Bảng 1** (EN/VI): nêu rõ phạm vi. EN "Table 1. Representative HMER methods with a reported expression-recognition rate on CROHME (as reported by the respective papers; "—" = not reported). The list is offline-dominated because the online works reviewed are surveys or auxiliary tasks without a comparable ExpRate; SCAN [22] is the online/multi-modal representative." VI tương ứng (xem nội dung đã soạn).
8. **Câu "late fusion"** (EN P132/VI P130) viết lại chủ động: EN→"Late fusion runs two independent recognizers and then combines their output probabilities; it preserves each modality but cannot model any interaction between them during decoding. For example, it cannot use a symbol relation to resolve an ambiguous image region at the moment a token is generated." VI→"Hợp nhất muộn chạy hai bộ nhận dạng độc lập rồi gộp xác suất đầu ra của chúng; cách này giữ nguyên từng phương thức nhưng không mô hình hóa được tương tác giữa hai nguồn trong lúc giải mã. Chẳng hạn, nó không thể dùng một quan hệ ký hiệu để làm rõ một vùng ảnh nhập nhằng ngay tại bước sinh token."
9. **Câu mở Mục 3.13** (EN P170/VI P168): EN→"We situate the proposed model with respect to its two closest predecessors, BTTR [23] and SCAN [22]." VI→"Chúng tôi định vị mô hình đề xuất trong tương quan với hai công trình tiền nhiệm gần nhất, BTTR [23] và SCAN [22]."
10. **Câu mở Mục 4.4** (EN P183/VI P181) [PA-B đã chọn]: EN→"The scope of our evaluation must be read carefully, as it directly shapes how the reported numbers should be interpreted." VI→"Phạm vi của đánh giá này cần được hiểu chính xác, vì nó ảnh hưởng trực tiếp đến cách diễn giải các con số báo cáo." (thay "It is important to state the scope…" / "Cần nêu chính xác phạm vi đánh giá.")
11. **Câu "cận trên / không so trực tiếp" (CÙNG đoạn 4.4, EN P183/VI P181)** — tách câu dài thành 3 câu ngắn. EN→"The reported numbers therefore measure how the fusion decoder performs when it is given an ideal, ground-truth structure. They should be read as an upper bound, not as a direct comparison with offline-only systems, which never receive such structural information. (External methods are listed only for context in Section 2, Table 1.)" VI→"Do đó, các con số báo cáo cho biết bộ giải mã hợp nhất hoạt động ra sao khi được cấp một cấu trúc lý tưởng (ground-truth). Chúng nên được hiểu là một cận trên, không phải so sánh trực tiếp với các hệ chỉ-offline — vốn không hề nhận thông tin cấu trúc đó. (Các phương pháp ngoài chỉ được liệt kê để đặt bối cảnh ở Phần 2, Bảng 1.)" [cùng paragraph với #10 → làm cả 2 thay thế]
12. **Câu ablation 4.6 (EN + VI, cùng đoạn 4.4, EN P183/VI P181)** — viết lại CẢ HAI. EN→"It is the ablation in Section 4.6—where every variant receives exactly the same inputs and is trained under the same protocol—that provides the fair, controlled comparison underpinning our claims." VI→"Chính thí nghiệm loại bỏ ở Mục 4.6 — nơi mọi biến thể nhận cùng đầu vào và được huấn luyện theo cùng giao thức — mới là phép so sánh công bằng, có kiểm soát; và đây mới là cơ sở cho các khẳng định của chúng tôi."
LƯU Ý: mục #10, #11, #12 đều trong CÙNG paragraph (VI P181 / EN P183) → khi áp dụng làm hết các thay thế trong đoạn đó.
13. **Hình 6 — đổi tiêu đề + caption** ("fusion strategies" → "decoder configurations", vì offline/online là baseline không phải fusion). SVG title: EN→"Five decoder configurations compared in the ablation (within one decoder layer)"; VI→"Năm cấu hình bộ giải mã được so sánh trong ablation (trong một lớp giải mã)" (sửa figure6_fusion_variants.svg + _vi.svg, render lại, thay inline_shapes[5] cả 2 docx). Caption: EN→"Figure 6. The five decoder configurations compared in the ablation (within one decoder layer)."; VI→"Hình 6. Năm cấu hình bộ giải mã được so sánh trong thí nghiệm loại bỏ (trong một lớp giải mã)." Nội dung 5 panel giữ nguyên.
14. **Caption Bảng 4** (EN P197/VI P195) — đổi câu đầu (Phương án 1, nêu rõ 2 baseline + 3 thiết kế hợp nhất). EN→"Table 4. Ablation of decoder-level fusion: single-modality baselines versus three fusion designs." VI→"Bảng 4. Thí nghiệm loại bỏ hợp nhất ở tầng bộ giải mã: các baseline đơn-phương-thức so với ba thiết kế hợp nhất." (giữ nguyên phần "All variants…/Mọi biến thể…"). BẢNG 5 GIỮ NGUYÊN (đã xác nhận OK).
15. **Câu mở Mục 4.8 Thảo luận** (EN P208/VI P206) [PA-C, dài+giàu ý, 3 câu hỏi↔3 kết luận]: EN→"The experiments in this chapter speak directly to the questions that motivated our design—whether to fuse the two modalities at all, how that fusion should be wired inside the decoder, and how much the reading direction matters—and they support three main conclusions." VI→"Các thí nghiệm trong chương này trả lời trực tiếp những câu hỏi đã thúc đẩy thiết kế của chúng tôi — có nên hợp nhất hai phương thức hay không, việc hợp nhất nên được đấu nối thế nào bên trong bộ giải mã, và chiều đọc quan trọng đến mức nào — và củng cố ba kết luận chính sau đây." (thay "Three observations summarize the study." / "Ba quan sát tóm tắt nghiên cứu.")
16. **Câu "Thứ nhất…" Mục 4.8 (EN P208/VI P206, cùng đoạn #15)** — tách câu, diễn đạt rõ (vế sau nêu rõ nhánh online=cấu trúc, ảnh offline=tín hiệu hình thái/appearance). EN→"First, decoder-level fusion is highly effective: every fusion variant outperforms the best single modality by roughly 8 to 10 points. Moreover, even when the online branch already provides a clean (ground-truth) structure, the offline image still adds value: it supplies appearance cues that help resolve cases that online structural cues alone leave ambiguous." VI→"Thứ nhất, hợp nhất ở tầng bộ giải mã rất hiệu quả: mọi biến thể hợp nhất đều vượt phương thức đơn-phương-thức tốt nhất khoảng 8–10 điểm. Hơn nữa, ngay cả khi nhánh online đã cung cấp cấu trúc sạch (ground-truth), ảnh offline vẫn có ích: nó mang lại tín hiệu hình thái giúp làm rõ những trường hợp mà riêng tín hiệu cấu trúc (online) còn để ngỏ."
17. **XÓA câu "hạn chế/future work" cuối Mục 4.8 (EN P208/VI P206)** — TRÙNG với 5.2 Limitations + 5.3 Future Work → xóa hẳn khỏi đoạn 4.8 (cả EN+VI); đoạn kết ở kết luận "Thứ ba…". EN xóa: "A limitation, noted earlier, is that the online branch uses a clean structural representation; replacing it with the output of an independent online recognizer is left to future work, and is the regime in which the offline branch is expected to contribute most." VI xóa: "Một hạn chế, đã nêu, là nhánh online dùng biểu diễn cấu trúc sạch; thay nó bằng đầu ra của một bộ nhận dạng online độc lập là hướng để dành cho tương lai, và là chế độ mà nhánh offline được kỳ vọng đóng góp nhiều nhất."
18. **Câu mở "Ba kết quả…" Mục 5.1 (EN P213/VI)** [PA-A]: EN→"The work delivers three results, each corresponding to one of the contributions stated in the Introduction." VI→"Luận văn mang lại ba kết quả, mỗi kết quả tương ứng với một đóng góp đã nêu ở phần Giới thiệu." (thay "Three results correspond to the contributions stated at the outset." / "Ba kết quả tương ứng với các đóng góp nêu ban đầu.")
19. **Câu "Thứ hai…" (khảo sát fusion) Mục 5.1 (EN P213/VI, cùng đoạn #18)** — tách câu, nêu "ba thiết kế" + tên đủ. EN→"The second is a systematic study of three decoder-level fusion designs—early (concat), cascaded, and shared-query dual cross-attention. All three perform comparably, within about 2.4 points on the micro-averaged ExpRate (early 75.89%, shared-query 74.17%, cascaded 73.51%), with early fusion marginally best; this shows that the benefit comes from decoder-level fusion itself, not from the specific way the cross-attentions are wired." VI→"Thứ hai là một khảo sát có hệ thống ba thiết kế hợp nhất ở tầng bộ giải mã — hợp nhất sớm (concat), nối tiếp (cascaded), và dùng chung truy vấn (shared-query dual cross-attention). Cả ba cho kết quả tương đương, chênh nhau khoảng 2,4 điểm theo trung bình vi mô (hợp nhất sớm 75,89%, dùng chung truy vấn 74,17%, nối tiếp 73,51%), với hợp nhất sớm nhỉnh nhất; điều này cho thấy lợi ích đến từ chính việc hợp nhất ở tầng bộ giải mã, chứ không phải từ cách đấu nối cụ thể giữa các khối chú ý chéo."
20. **Câu mở Mục 5.2 Limitations** (EN P215/VI) [PA-C, dài+học thuật]: EN→"Like any empirical study, this work has limitations that bound the scope of its conclusions. We state them explicitly—both for transparency and because they mark where our claims hold and where further validation is needed." VI→"Như mọi nghiên cứu thực nghiệm, công trình này có những hạn chế giới hạn phạm vi của các kết luận. Chúng tôi nêu rõ chúng — vừa để minh bạch, vừa vì chúng cho thấy các khẳng định của chúng tôi đúng trong phạm vi nào và cần kiểm chứng thêm ở đâu." (thay "The study has clear boundaries, and it is better to name them plainly." / "Nghiên cứu có những ranh giới rõ ràng, và nên nêu