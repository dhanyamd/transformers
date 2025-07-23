import os 

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTHON_ENABLE_MPS_FALLBACK"] = "1" 
from torch.utils.tensorboard import SummaryWriter 
import torch 

torch.autograd.set_detect_anomaly(True)
from dataset import get_dataset, get_tokenzier 
from transcribe_model import TranscribeModel
from torch import nn 

vq_initial_loss_weight = 10 
vq_warmup_steps = 1000 
vq_final_loss_weight = 0.5 
num_epochs = 1000 
starting_steps = 0
num_examples = 100 
model_id = "test37"
num_batch_repeats = 1 

starting_steps = 0
BATCH_SIZE = 64 
LEARNING_RATE = 0.005

def run_loss_function(log_probs, target, blank_token): 
    # Add log_softmax to ensure proper probability ditribution 

    loss_function = nn.CTCLoss(blank=blank_token) 
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0])) 
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first= log_probs.permute(1,0,2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    return loss 

def decode_predictions(output, tokenizer, blank_token):
    """Decode model predictions to text"""
    predictions = []
    batch_size = output.shape[0]
    
    for i in range(batch_size):
        # Get the most likely tokens
        pred_tokens = torch.argmax(output[i], dim=-1)
        
        # Remove blanks and consecutive duplicates (CTC decoding)
        decoded_tokens = []
        prev_token = None
        for token in pred_tokens:
            token_id = token.item()
            if token_id != blank_token and token_id != prev_token:
                decoded_tokens.append(token_id)
            prev_token = token_id
        
        # Convert to text
        try:
            text = tokenizer.decode(decoded_tokens)
            predictions.append(text)
        except:
            predictions.append("")
    
    return predictions

def main():
    log_dir = f"runs/speech2text_training/{model_id}"
    if os.path.exists(log_dir): 
        import shutil 
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir) 

    tokenizer = get_tokenzier()
    blank_token = tokenizer.token_to_id("□")

    device = torch.device(
        "cuda"
         if torch.cuda.is_available() 
         else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device : {device}")

    if os.path.exists(f"models/{model_id}/model_latest.pth"): 
        print(f"Loading model from models/{model_id}/model_latest.pth") 
        model = TranscribeModel.load(f"models/{model_id}/model_latest.pth").to(device)
    else:
        model = TranscribeModel(
            num_codebooks=2,
            codebook_size=32,
            embedding_dim=16,
            num_transformer_layers=2,
            vocab_size=len(tokenizer.get_vocab())
        ).to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameter: {num_trainable_params}") 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataloader = get_dataset(
        batch_size=BATCH_SIZE,
        num_examples=num_examples,
        num_workers=1 
    )

    ctc_losses = []
    vq_losses = []
    num_batches = len(dataloader)
    steps = starting_steps 

    for i in range(num_epochs): 
        for idx, batch in enumerate(dataloader): 
            for repeatBatch in range(num_batch_repeats): 
                audio = batch["audio"]
                target = batch["input_ids"]
                text = batch["text"] 

                if target.shape[1] > audio.shape[1]: 
                    print(
                        "Padding audio, target is longer than audio. Audio Shape: ",
                        audio.shape,
                        "Target Shape: ",
                        target.shape
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0,0,0,target.shape[1] - audio.shape[1])
                    )
                    print("After padding: ", audio.shape) 
                audio = audio.to(device)
                target = target.to(device) 

                optimizer.zero_grad()
                output, vq_loss = model(audio) 
                ctc_loss = run_loss_function(output, target, blank_token) 
                
                #Calculate vq_loss_weight using linear warmup schedule 
                vq_loss_weight =  max(
                    vq_final_loss_weight,
                    vq_initial_loss_weight 
                    - (vq_initial_loss_weight - vq_final_loss_weight) 
                    * (steps / vq_warmup_steps)
                )
                if vq_loss is None: 
                    loss = ctc_loss
                else:
                    loss = ctc_loss + vq_loss_weight * vq_loss
                if torch.isinf(loss): 
                    print("Loss is inf, skipping step", audio.shape, target.shape) 
                    continue 
                loss.backward() 
                #Increase gradient clipping threshold 
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0
                ) #changed from 1.0
                optimizer.step() 

                ctc_losses.append(ctc_loss.item()) 
                vq_losses.append(vq_loss.item() if vq_loss is not None else 0)  # Fixed bug here
                steps += 1

                #Log to tensorboard every step
                if steps % 20 == 0: 
                    avg_ctc_loss = sum(ctc_losses) / len(ctc_losses) 
                    avg_vq_loss = sum(vq_losses) / len(vq_losses) 
                    avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss
                    print(
                        f"Num Steps: {steps}, Batch: {idx + 1}/{num_batches}, ctc_loss: {avg_ctc_loss:.3f}, vq_loss: {avg_vq_loss:.3f}, total_loss: {avg_loss:.3f}"
                    )

                    # Log to tensorboard
                    writer.add_scalar('Loss/CTC', avg_ctc_loss, steps)
                    writer.add_scalar('Loss/VQ', avg_vq_loss, steps)
                    writer.add_scalar('Loss/Total', avg_loss, steps)
                    writer.add_scalar('Loss/VQ_Weight', vq_loss_weight, steps)

                    # Generate transcription examples
                    if steps % 20 == 0:
                        model.eval()
                        with torch.no_grad():
                            # Use current batch for examples
                            sample_output, _ = model(audio[:4])  # Use first 4 samples
                            predictions = decode_predictions(sample_output, tokenizer, blank_token)
                            ground_truths = text[:4]  # First 4 ground truth texts
                            
                            print("Transcription Examples")
                            print("=" * 80)
                            print(f"{'Example #':<10} {'Model Output':<35} {'Ground Truth'}")
                            print("=" * 80)
                            
                            for j, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                                print(f"{j:<10} {pred:<35} {truth}")
                            
                            print("=" * 80)
                        model.train()

                    ctc_losses = []
                    vq_losses = []

                # Save model periodically
                if steps % 100 == 0:
                    os.makedirs(f"models/{model_id}", exist_ok=True)
                    model.save(f"models/{model_id}/model_latest.pth")
                    print(f"Model saved at step {steps}")

    # Final model save
    os.makedirs(f"models/{model_id}", exist_ok=True)
    model.save(f"models/{model_id}/model_final.pth")
    print("Training completed. Final model saved.")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    main()