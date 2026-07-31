# CLAUDE.md — Luận văn thạc sĩ "Multi-modal HMER" (Dual-Modality BTTR)

Dự án gồm 2 phần trong CÙNG thư mục: **code mô hình** (gốc repo) và **luận văn** (`thesis/`).
Trạng thái chi tiết + lịch sử mọi quyết định nằm trong **`thesis/TRANG_THAI_LUAN_VAN.md`**
(file trong repo — Cowork lẫn Claude Code đều đọc được) — **đọc nó trước khi sửa bất cứ thứ gì
của luận văn, và CẬP NHẬT nó sau mỗi quyết định mới.**

## Cấu trúc thư mục

| Đường dẫn       | Nội dung                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `thesis/`         | **Nơi lưu DUY NHẤT** của luận văn: `Multi-modal_HMER_Sections1-5_full.docx/.pdf` (EN) + `Multi-modal_HMER_TiengViet_full.docx/.pdf` (VI), `references_ieee.txt` ([1]–[38], đã xác minh), `KIEN_TRUC_HE_THONG.md/.docx` (tài liệu kiến trúc — ĐÃ đối chiếu khớp code 2026-07-02, dùng làm chuẩn đối chiếu luận văn), `ABLATION_GUIDE.md` |
| `thesis/figures/` | 24 file nguồn hình (6 hình × EN/VI × svg+png) — SỬA HÌNH TẠI ĐÂY rồi render + nhúng lại, đừng vẽ mới                                                                                                                                                                                                                                                               |
| `references/`     | PDF tài liệu tham khảo, tên`N.pdf` khớp số [N]; bài paywall có `[N]_ABSTRACT.txt`; xem `INDEX.txt`                                                                                                                                                                                                                                                                     |
| `results/`        | 24 file kết quả test — BẰNG CHỨNG cho mọi con số ở Bảng 3/4/5, không xóa                                                                                                                                                                                                                                                                                                  |
| `demo/`           | Gallery web tĩnh kết quả seed-7 (mở`demo/index.html` trực tiếp, không cần server)                                                                                                                                                                                                                                                                                          |
| gốc                | Code:`bttr/`, `custom_train.py` (đường huấn luyện chính thức, seed 7), `test_all.py`, `predict_test.py`; dữ liệu `data.zip`, `crohme_all.txt`, `best.ckpt` — **không di chuyển** (code tham chiếu đường dẫn gốc)                                                                                                                               |

## Quy tắc làm việc với luận văn (user đặt ra — BẮT BUỘC)

1. **Không bịa thông tin.** Mọi trích dẫn/con số phải có nguồn xác minh (Crossref, IEEE, arXiv, trang tác giả). Không chắc → hỏi lại user.
2. **Quy trình sửa:** soạn nguyên văn old→new → user duyệt → user ra lệnh "sửa chính thức" mới được ghi vào file.
3. **Mỗi phép sửa docx phải khớp ĐÚNG 1 vị trí** (script python-docx; 0 hoặc >1 match → không lưu, báo lại).
4. Sau khi sửa docx: **xuất lại 2 PDF vào `thesis/`**. KHÔNG đồng bộ sang Downloads (quy tắc từ 2026-07-02).
5. Sửa hình: sửa SVG trong `thesis/figures/` → render → **gửi user xem trước** → mới nhúng vào docx.
6. **Ưu tiên tra cứu kho `references/` TRƯỚC** (38 tài liệu đánh số khớp [1]–[38], xem `INDEX.txt`); chỉ khi kho không có mới tìm trên internet — và vẫn phải là nguồn chính thống.
7. **Giọng văn khi viết/sửa bài:** học thuật, chuyên nghiệp, tự nhiên như người viết — hòa vào giọng hiện có của luận văn (câu dài ngắn đan xen, lập luận liền mạch, chủ ngữ rõ). TRÁNH khuôn mẫu AI: không thay văn xuôi bằng gạch đầu dòng, không sáo ngữ dồn dập ("Ngoài ra/Hơn nữa/Tóm lại", "đóng vai trò quan trọng", "không thể phủ nhận"), không lặp cấu trúc "không chỉ… mà còn…", không kết đoạn tổng kết máy móc. Bản EN: tránh boilerplate kiểu "delve", "crucial", "It is worth noting", chuỗi "Moreover/Furthermore".

## Gotchas kỹ thuật (đã trả giá — đừng lặp lại)

- **Python cần torch/PIL:** phải `conda run -n bttr --no-capture-output python ...` (gọi thẳng python.exe → WinError 126 thiếu DLL CUDA). Script chỉ dùng python-docx thì gọi thẳng `C:/Users/Admin/miniconda3/envs/bttr/python.exe` được.
- **In tiếng Việt ra console crash cp1252** → luôn ghi kết quả ra file UTF-8 rồi đọc file.
- **Xuất PDF (Word COM):** chạy trong PowerShell `Start-Job` + `Wait-Job -Timeout` (gọi foreground bị treo); kill WINWORD trước; mỗi lần 1 file. Mẫu lệnh trong memory.
- **docx phải có `wp:docPr id` DUY NHẤT** — id trùng làm `ExportAsFixedFormat` treo vô hạn khi CreateBookmarks=1 (đã fix, giữ nguyên khi nhúng hình: chỉ thay blob).
- **pdftotext không trích được tiếng Việt có dấu** từ PDF xuất bằng Word → verify nội dung bằng cách đọc docx, không tin grep trên PDF tiếng Việt.
- **Render hình:** Chrome headless `--headless=new --force-device-scale-factor=3 --window-size=WxH --screenshot=...` trên file HTML bọc SVG. Máy KHÔNG có cairo/rsvg/inkscape/magick/pdftoppm.
- Mapping hình trong cả 2 docx: `inline_shapes[0..5]` = Hình 1..6. Bảng: `tables[0]`=Bảng 1, `tables[1]`=Bảng 2…

## Số liệu chốt (không được sai lệch)

- "Ours" = shared-query dual cross-attention, **seed 7**: ExpRate **77.06 / 72.62 / 73.29** (CROHME 2014/16/19), micro **74.17%**. Checkpoint: `lightning_logs/abl_dual_shared/lightning_logs/version_1/checkpoints/epoch=39-step=45560-val_loss=0.1566.ckpt`.
- Ablation: concat 75.89 > shared-query 74.17 > cascaded 73.51 (micro) — KHÔNG tuyên bố shared-query tốt nhất; hai chiều hơn một chiều +10.66đ.
- ⚠️ Caveat cốt lõi: nhánh online dùng SRT **ground-truth** (oracle) → mọi số là CẬN TRÊN, không so trực tiếp với hệ offline-only (đã ghi rõ ở Mục 4.4).

## Việc còn mở (tùy chọn)

- Có PDF gốc [11] BPD / [17] GETD → đối chiếu số Bảng 1 (ô GETD hiện để "—" vì số cũ chưa truy được nguồn).
- (Tùy chọn) chạy thêm seed xác nhận chênh lệch fusion-design; review style References.
