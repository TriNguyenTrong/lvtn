import torch
import os
import zipfile
from PIL import Image
from torchvision.transforms import transforms
from bttr.lit_bttr import LitBTTR
from bttr.datamodule.datamodule import load_online

image_name = "2016/UN_101_em_0.bmp"

def predict(image_name="2014/18_em_1.bmp"):
    # 1. Load model từ file best.ckpt
    print("Loading model...")
    model = LitBTTR.load_from_checkpoint("best.ckpt", strict=False)
    model.eval()
    
    # Đưa model lên GPU nếu có
    if torch.cuda.is_available():
        model = model.cuda()

    # 2. Đọc ảnh từ zip hoặc folder
    img_name_only = os.path.splitext(os.path.basename(image_name))[0]
    file_name_with_ext = image_name if image_name.endswith(".bmp") else f"{image_name}.bmp"
    
    print(f"Loading image {image_name} from data.zip...")
    with zipfile.ZipFile("data.zip") as archive:
        with archive.open(file_name_with_ext, "r") as f:
            img = Image.open(f).copy()
    
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(model.device) # [1, 1, H, W]
    img_mask = torch.zeros((1, img_tensor.shape[2], img_tensor.shape[3]), dtype=torch.bool, device=model.device)

    # 3. Lấy quỹ đạo online tương ứng (do tools/prep_online.py sinh ra)
    split = image_name.split("/")[0]
    traj_dict = load_online(os.path.join("online", f"{split}.npz"))

    if img_name_only not in traj_dict:
        print(f"Lỗi: Không tìm thấy quỹ đạo cho ảnh {img_name_only} trong online/{split}.npz")
        return

    traj = torch.from_numpy(traj_dict[img_name_only]).unsqueeze(0).to(model.device)  # [1, l1, 8]
    traj_mask = torch.zeros(traj.shape[:2], dtype=torch.bool, device=model.device)

    # 4. Tiến hành giải mã (Beam Search)
    print("Running Inference (Beam Search)...")
    with torch.no_grad():
        hyps = model.bttr.beam_search(
            img_tensor,
            img_mask,
            traj,
            traj_mask,
            model.hparams.beam_size,
            model.hparams.max_len
        )
    
    # Lấy chuỗi có xác suất cao nhất (best hypothesis)
    best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
    
    # 5. Chuyển đổi từ index sang LaTeX sequence
    pred_latex = model.vocab_dec.indices2label(best_hyp.seq)
    
    print("\n" + "="*40)
    print("                KẾT QUẢ                 ")
    print("="*40)
    print(f"Mã ảnh: {img_name_only}")
    print(f"LaTeX dự đoán: {pred_latex}")
    print("="*40)

if __name__ == "__main__":
    # Đọc từ file zip (Mặc định)
    # predict(image_name, from_zip=True)
    
    # Đọc từ file/folder bất kỳ:
    path = "example/18_em_1.bmp"
    predict()
