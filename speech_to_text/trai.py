import os
import torch.nn.functional as F
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

BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Keep this lower for stability

def run_loss_function(log_probs, target, blank_token):
    """
    Calculates the CTC loss.
    Ensures log_softmax is applied to model output and reduction is 'mean'.
    """
    # Apply log_softmax to model output to get log-probabilities.
    log_probs = F.log_softmax(log_probs, dim=2)

    loss_function = nn.CTCLoss(blank=blank_token, reduction='mean')

    # CRITICAL FIX: Ensure input_lengths is a tuple of integers, matching target_lengths_tuple
    # input_lengths: A tuple of lengths of the inputs (each a length of one batch item)
    # This should be the length of the sequence dimension of log_probs for each item in the batch
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0])) # <--- REVERTED to original correct logic

    # Calculate target lengths (actual length of transcription, excluding padding)
    target_lengths = (target != blank_token).sum(dim=1)
    target_lengths_tuple = tuple(t.item() for t in target_lengths)

    # Permute log_probs for CTCLoss: (sequence_length, batch_size, vocab_size)
    input_seq_first = log_probs.permute(1, 0, 2)

    # Calculate the CTC loss
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths_tuple) # input_lengths is now a tuple

    return loss

# Rest of your trainer.py code remains the same as the last corrected version
# (including the changes for total_loss, isinf/isnan checks, and strides in main)

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
            # Fallback for decoding errors, append empty string
            predictions.append("")
    
    return predictions

def main():
    # Setup TensorBoard writer
    log_dir = f"runs/speech2text_training/{model_id}"
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir) # Clear previous logs for fresh run
    writer = SummaryWriter(log_dir)

    # Initialize tokenizer and blank token
    tokenizer = get_tokenzier()
    blank_token = tokenizer.token_to_id("□")

    # Device configuration
    device = torch.device(
        "cuda"
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device : {device}")

    # Model initialization or loading
    if os.path.exists(f"models/{model_id}/model_latest.pth"):
        print(f"Loading model from models/{model_id}/model_latest.pth")
        model = TranscribeModel.load(f"models/{model_id}/model_latest.pth").to(device)
    else:
        # Initializing TranscribeModel with strides as a list
        model = TranscribeModel(
            num_codebooks=2,
            codebook_size=64,
            embedding_dim=64,
            num_transformer_layers=2,
            strides=[6, 8, 4, 2], # FIXED: Passed strides as a list
            intial_mean_pooling_kernel_size=4,
            vocab_size=len(tokenizer.get_vocab())
        ).to(device)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameter: {num_trainable_params}")

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data loader setup
    dataloader = get_dataset(
        batch_size=BATCH_SIZE,
        num_examples=num_examples,
        num_workers=1
    )

    # Loss tracking
    ctc_losses = []
    vq_losses = []
    num_batches = len(dataloader)
    steps = starting_steps

    # Training loop
    for i in range(num_epochs):
        for idx, batch in enumerate(dataloader):
            # The 'repeatBatch' loop is kept as per your original code,
            # though it only runs once if num_batch_repeats is 1.
            for repeatBatch in range(num_batch_repeats):
                audio = batch["audio"]
                target = batch["input_ids"]
                text = batch["text"]

                # Handle padding if target sequence is longer than audio output sequence
                # This might need refinement based on actual model output lengths vs. target lengths
                # after downsampling.
                if target.shape[1] > audio.shape[1]:
                    print(
                        "Padding audio, target is longer than audio. Audio Shape: ",
                        audio.shape,
                        "Target Shape: ",
                        target.shape
                    )
                    # Pad audio to match target length (might need to be more sophisticated
                    # to match model's output sequence length after convolutions/downsampling)
                    audio = torch.nn.functional.pad(
                        audio, (0, 0, 0, target.shape[1] - audio.shape[1])
                    )
                    print("After padding: ", audio.shape)
                
                audio = audio.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output, vq_loss = model(audio) # Model forward pass

                # --- Debugging prints for model output (can remove later) ---
                # print(f"Model output shape: {output.shape}")
                # print(f"Model output min: {output.min().item():.4f}, max: {output.max().item():.4f}")
                # if torch.isnan(output).any() or torch.isinf(output).any():
                #     print("!!! WARNING: Model output contains NaN or Inf before CTC loss !!!")
                #     # Consider adding breakpoint() here for interactive debugging
                # -----------------------------------------------------------

                ctc_loss = run_loss_function(output, target, blank_token)

                # Calculate vq_loss_weight using linear warmup schedule
                vq_loss_weight = max(
                    vq_final_loss_weight,
                    vq_initial_loss_weight 
                    - (vq_initial_loss_weight - vq_final_loss_weight) 
                    * (steps / vq_warmup_steps)
                )

                # Ensure vq_loss is a scalar if it's not None and has multiple elements
                if vq_loss is not None and vq_loss.numel() > 1:
                    vq_loss = vq_loss.mean() # Aggregate to a scalar if needed

                # Combine losses into a single scalar total_loss
                total_loss = ctc_loss # Start with ctc_loss
                if vq_loss is not None:
                    total_loss = ctc_loss + vq_loss_weight * vq_loss
                
                # Ensure total_loss is scalar before isinf check and backward pass
                if total_loss.numel() > 1:
                    total_loss = total_loss.mean() # Aggregate to a scalar if needed

                # --- Debugging prints for losses (can remove later) ---
                # print(f"CTC Loss: {ctc_loss.item():.4f}")
                # if vq_loss is not None:
                #     print(f"VQ Loss: {vq_loss.item():.4f}")
                # print(f"Total Loss: {total_loss.item():.4f}")
                # -----------------------------------------------------

                # Check for NaN/Inf in total_loss before backward pass
                if torch.isinf(total_loss) or torch.isnan(total_loss): # CRITICAL: Check for NaN as well
                    print("Loss is Inf or NaN, skipping step. Audio Shape:", audio.shape, "Target Shape:", target.shape)
                    continue # Skip this step if loss is bad

                total_loss.backward() # CRITICAL: Call backward on the scalar total_loss

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0
                )
                optimizer.step()

                # Append losses for averaging
                ctc_losses.append(ctc_loss.item())
                vq_losses.append(vq_loss.item() if vq_loss is not None else 0)
                steps += 1

                # Log to tensorboard and print progress every 20 steps
                if steps % 20 == 0:
                    avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
                    avg_vq_loss = sum(vq_losses) / len(vq_losses)
                    avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss # Recalculate avg_loss for printing

                    print(
                        f"Num Steps: {steps}, Batch: {idx + 1}/{num_batches}, ctc_loss: {avg_ctc_loss:.3f}, vq_loss: {avg_vq_loss:.3f}, total_loss: {avg_loss:.3f}"
                    )

                    # Log to tensorboard
                    writer.add_scalar('Loss/CTC', avg_ctc_loss, steps)
                    writer.add_scalar('Loss/VQ', avg_vq_loss, steps)
                    writer.add_scalar('Loss/Total', avg_loss, steps)
                    writer.add_scalar('Loss/VQ_Weight', vq_loss_weight, steps)

                    # Generate transcription examples periodically
                    if steps % 20 == 0: # Redundant but harmless if outer is also 20
                        model.eval() # Set model to evaluation mode for inference
                        with torch.no_grad(): # Disable gradient calculation for inference
                            # Use first 4 samples from the current batch for examples
                            sample_output, _ = model(audio[:4])
                            predictions = decode_predictions(sample_output, tokenizer, blank_token)
                            ground_truths = text[:4] # Corresponding ground truth texts

                            print("Transcription Examples")
                            print("=" * 80) # Separator line
                            print(f"{'Example #':<10} {'Model Output':<35} {'Ground Truth'}") # Table header
                            print("=" * 80) # Separator line

                            # Print individual examples
                            for j, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                                print(f"{j:<10} {pred:<35} {truth}")

                            print("=" * 80) # Separator line
                        model.train() # Set model back to training mode

                    # Reset loss lists for the next logging interval
                    ctc_losses = []
                    vq_losses = []

                # Save model periodically
                if steps % 100 == 0:
                    os.makedirs(f"models/{model_id}", exist_ok=True)
                    model.save(f"models/{model_id}/model_latest.pth")
                    print(f"Model saved at step {steps}")

    # Final model save after all epochs
    os.makedirs(f"models/{model_id}", exist_ok=True)
    model.save(f"models/{model_id}/model_final.pth")
    print("Training completed. Final model saved.")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    main()
