import random 
import torchaudio 
import torch
import datasets 
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F 
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from pathlib import Path
import sounddevice as sd 


def collatee_fn(batch):
    max_audio_len = max([item["audio"].shape[0] for item in batch])
    max_ids_len = 0 
    has_input_ids = "input_ids" in batch[0] 
    if has_input_ids: 
        max_ids_len = max([len(item["input_ids"]) for item in batch]) 
    #pad audio sequences 
    audio_tensor = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[0])) 
            for item in batch
        ]
    )
    output_dict = {
        "audio": audio_tensor,
        "text": [item["text"] for item in batch] 
    }
    if has_input_ids: 
        input_ids = torch.stack(
            [
                F.pad(
                    torch.tensor(item["input_ids"]),
                    (0, max_ids_len - len(item["input_ids"]))
                )
                for item in batch 
            ]
        )
        output_dict["input_ids"] = input_ids
    return output_dict

def get_tokenzier(save_path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE()) 
    tokenizer.add_special_tokens(["□"])
    tokenizer.add_tokens(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ '"))

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel() 
    tokenizer.decoder = decoders.ByteLevel() 
    tokenizer.blank_token = tokenizer.token_to_id("□")
    tokenizer.save(save_path)
    return tokenizer 

class CommonVoiceDataset(Dataset): 
    def __init__(
            self, 
            common_voice_dataset,
            num_examples=None,
            tokenizer=None 
    ): 
        self.dataset = common_voice_dataset 
        self.num_examples = (
            min(num_examples, len(common_voice_dataset)) 
            if num_examples is not None 
            else len(common_voice_dataset) 
        )
        self.tokenizer = tokenizer 
    def __len__(self): 
        return self.num_examples 
    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        waveform = torch.from_numpy(item["audio"]["array"]).float() 
        text = item[
            "transcription"
        ].upper() 
        if self.tokenizer: 
            encoded = self.tokenizer.encode(text) 
            return {"audio": waveform, "text": text, "input_ids": encoded.ids}
def get_dataset(batch_size=32, num_examples=None, num_workers=4): 
    dataset = datasets.load_dataset(
    "m-aliabbas/idrak_timit_subsample1",
    split="train"
)
    tokenizer = get_tokenzier() 
    #create a new dataset with the trained tokenizer 
    dataset = CommonVoiceDataset(
        dataset,
        tokenizer=tokenizer,
        num_examples=num_examples
    )
    datasetloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collatee_fn, # <--- Corrected to collate_fn (with one 'l' and no 'e' at the end)
        num_workers=num_workers
    )
    return datasetloader
if __name__ == "__main__": 
    dataloader = get_dataset(batch_size=32)
    for batch in dataloader: 
        audio = batch["audio"] 
        input_ids = batch["input_ids"]
        print(audio.shape)
        print(input_ids.shape)
        breakpoint()
        break