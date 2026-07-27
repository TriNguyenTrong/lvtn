import torch
import os
from tqdm import tqdm
from bttr.lit_bttr import LitBTTR
from bttr.datamodule import CROHMEDatamodule

def test_on_dataset(test_year="2014", ckpt_path="best.ckpt", output_file="test_results.txt"):
    # 1. Load model
    print(f"Loading model from {ckpt_path}...")
    model = LitBTTR.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. Setup DataModule
    print(f"Setting up dataset for year {test_year}...")
    dm = CROHMEDatamodule(test_year=test_year)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # 3. Predict and save
    print(f"Running inference and saving to {output_file}...")
    results = []
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            
            # 1. Prepare offline sequence features
            seq_indices = batch.seq_indices[0] # batch_size is 1
            seq = torch.tensor([seq_indices], dtype=torch.long, device=device)
            seq_mask = torch.zeros((1, len(seq_indices)), dtype=torch.bool, device=device)

            # 2. Beam Search
            hyps = model.bttr.beam_search(
                batch.imgs,
                batch.mask,
                seq,
                seq_mask,
                model.hparams.beam_size,
                model.hparams.max_len
            )

            # 3. Get best hypothesis
            best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))
            pred_latex = model.vocab_dec.indices2label(best_hyp.seq)
            
            # 4. Ground truth
            gt_latex = model.vocab_dec.indices2label(batch.indices[0])
            img_name = batch.img_bases[0]
            
            # 5. Check correctness
            is_correct = (pred_latex == gt_latex)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            status = "CORRECT" if is_correct else "WRONG"
            results.append(f"{img_name}\t{status}\t{pred_latex}\t{gt_latex}")

    # 4. Write to file
    exprate = (correct_count / total_count) * 100 if total_count > 0 else 0
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Test Year: {test_year}\n")
        f.write(f"Total: {total_count}, Correct: {correct_count}, ExpRate: {exprate:.2f}%\n")
        f.write("-" * 40 + "\n")
        f.write("Image\tStatus\tPrediction\tGroundTruth\n")
        f.write("\n".join(results))
    
    print(f"\nSummary for {test_year}:")
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"ExpRate: {exprate:.2f}%")
    print(f"Results saved to {output_file}")
    
    print(f"Done! Results saved to {output_file}")

test_year = "2019"
if __name__ == "__main__":
    # You can change the year or output file here
    test_on_dataset(test_year=test_year, output_file=f"test_{test_year}_results.txt")
