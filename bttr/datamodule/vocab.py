import os
from functools import lru_cache
from typing import Dict, List, Tuple
import torch
from torch import LongTensor


vocab_file_name = "dictionary.txt"
# vocab_file_name = "new_dictionary.txt"

@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), vocab_file_name)


class CROHMEVocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = dict()
        self.word2idx["<pad>"] = self.PAD_IDX
        self.word2idx["<sos>"] = self.SOS_IDX
        self.word2idx["<eos>"] = self.EOS_IDX

        with open(dict_path, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        # print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def to_tgt_output(self,
        tokens: List[List[int]], direction: str, device: torch.device
    ) -> Tuple[LongTensor, LongTensor]:
        """Generate tgt and out for indices

        Parameters
        ----------
        tokens : List[List[int]]
            indices: [b, l]
        direction : str
            one of "l2f" and "r2l"
        device : torch.device

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            tgt, out: [b, l], [b, l]
        """
        assert direction in {"l2r", "r2l"}

        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
        if direction == "l2r":
            tokens = tokens
            start_w = self.SOS_IDX
            stop_w = self.EOS_IDX
        else:
            tokens = [torch.flip(t, dims=[0]) for t in tokens]
            start_w = self.EOS_IDX
            stop_w = self.SOS_IDX

        batch_size = len(tokens)
        lens = [len(t) for t in tokens]
        tgt = torch.full(
            (batch_size, max(lens) + 1),
            fill_value=self.PAD_IDX,
            dtype=torch.long,
            device=device,
        )
        out = torch.full(
            (batch_size, max(lens) + 1),
            fill_value=self.PAD_IDX,
            dtype=torch.long,
            device=device,
        )

        for i, token in enumerate(tokens):
            tgt[i, 0] = start_w
            tgt[i, 1 : (1 + lens[i])] = token

            out[i, : lens[i]] = token
            out[i, lens[i]] = stop_w

        return tgt, out


    def to_bi_tgt_out(self,
        tokens: List[List[int]], device: torch.device
    ) -> Tuple[LongTensor, LongTensor]:
        """Generate bidirection tgt and out

        Parameters
        ----------
        tokens : List[List[int]]
            indices: [b, l]
        device : torch.device

        Returns
        -------
        Tuple[LongTensor, LongTensor]
            tgt, out: [2b, l], [2b, l]
        """
        l2r_tgt, l2r_out = self.to_tgt_output(tokens, "l2r", device)
        r2l_tgt, r2l_out = self.to_tgt_output(tokens, "r2l", device)

        tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
        out = torch.cat((l2r_out, r2l_out), dim=0)

        return tgt, out
    
    def to_src(self,
        tokens: List[List[int]], device: torch.device
    ) -> Tuple[LongTensor, LongTensor]:
        """Generate bidirection tgt and out

        Parameters
        ----------
        tokens : List[List[int]]
            indices: [b, l]
        device : torch.device

        Returns
        -------
        Tuple[LongTensor, LongTensor]
            src: [b, l]
        """
        l2r_src, _ = self.to_tgt_output(tokens, "l2r", device)
        src_padding_mask = (l2r_src == self.PAD_IDX)

        return l2r_src, src_padding_mask
