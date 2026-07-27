import torch
import os
import zipfile
from PIL import Image
from torchvision.transforms import transforms
from bttr.lit_bttr import LitBTTR
from bttr.datamodule.vocab import CROHMEVocab

image_name = "2016/UN_101_em_0.bmp"

def to_src(seq_indices, device):
    # Tạo tensor seq và seq_mask cho logic beam search  
    seq = torch.tensor([seq_indices], dtype=torch.long, device=device)
    seq_mask = torch.zeros((1, len(seq_indices)), dtype=torch.bool, device=device)
    return seq, seq_mask

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

    # 3. Lấy offline sequence features tương ứng từ annotation file
    seq_dict = {
        os.path.splitext(os.path.basename(line.strip().split('\t')[0]))[0]: line.strip().split('\t')[1]
        for line in open("crohme_all.txt").readlines() if len(line.strip().split('\t')) == 2
    }
    
    if img_name_only not in seq_dict:
        print(f"Lỗi: Không tìm thấy sequence feature cho ảnh {img_name_only} trong crohme_all.txt")
        return

    vocab_enc = CROHMEVocab("vocab/crohme_seq_vocab.txt")
    seq_indices = vocab_enc.words2indices(seq_dict[img_name_only].split())
    seq, seq_mask = to_src(seq_indices, model.device)
    
    # 4. Tiến hành giải mã (Beam Search)
    print("Running Inference (Beam Search)...")
    with torch.no_grad():
        hyps = model.bttr.beam_search(
            img_tensor, 
            img_mask, 
            seq, 
            seq_mask, 
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
