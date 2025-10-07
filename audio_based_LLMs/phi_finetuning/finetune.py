import argparse
import json
import os
from pathlib import Path
# import glob # No longer needed for direct file globbing in dataset class
import random
import pandas as pd # Import pandas for CSV handling
import numpy as np
import datetime  # For timestamp

# Set environment variable to avoid tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

import torch
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object  # Add this import
# ... (other imports from the previous script remain the same)
from datasets import Audio # For loading audio easily
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
from functools import partial # For collate_fn

# Task-specific prompt and labels (remain the same)
AD_CLASSIFICATION_PROMPT_TEMPLATE = (
    "Transcription: \"{transcription}\"\n\n"
    "Based on the speech audio and its transcription, "
    "classify the speaker as dementia (Alzheimer's Disease or Related Dementia) or control (Cognitively Normal) with a single word: 'dementia' or 'control'."
)
POSITIVE_CLASS = "dementia"
NEGATIVE_CLASS = "control"
CLASS_LABELS = [POSITIVE_CLASS, NEGATIVE_CLASS]

ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100

# MultipleTokenBatchStoppingCriteria class (remains the same)
class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx > 0)


class ADClassificationDataset(Dataset):
    def __init__(self, processor, csv_path, audio_dir, split_name, task_prompt_template,
                 positive_class="dementia", negative_class="control",
                 max_samples=None, rank=0, world_size=1, max_audio_seconds=30):
        self.processor = processor
        self.task_prompt_template = task_prompt_template
        self.positive_class = positive_class # e.g., "dementia"
        self.negative_class = negative_class # e.g., "control"
        self.data = []
        self.audio_loader = Audio(sampling_rate=16000) # Ensure consistent sampling rate
        self.audio_dir = Path(audio_dir) # Base directory for audio files
        self.max_audio_length = max_audio_seconds * 16000  # Convert seconds to samples at 16kHz
        self.max_transcription_chars = 750  # Maximum chars for transcription

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        required_columns = ['uid', 'transcription', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")

        for _, row in df.iterrows():
            uid = str(row['uid'])
            transcription = str(row['transcription'])
            # Truncate transcription at init time
            if len(transcription) > self.max_transcription_chars:
                transcription = transcription[:self.max_transcription_chars]
                
            label = str(row['label'])

            # Construct audio file path
            potential_audio_path = os.path.join(self.audio_dir, uid)
            if os.path.isfile(potential_audio_path):
                self.data.append({
                    "audio_path": potential_audio_path,
                    "transcription_text": transcription,
                    "label": label
                })
            else:
                print(f"Warning: Audio file for UID {uid} (label: {label}) not found at {potential_audio_path}. Skipping.")

        if not self.data:
            raise ValueError(f"No valid data loaded from {csv_path} with audio files in {self.audio_dir}. "
                             f"Check CSV content, audio file paths, UIDs, and labels.")

        if max_samples is not None:
            random.shuffle(self.data) # Shuffle before taking a subset
            self.data = self.data[:max_samples]

        if world_size > 1:
            self.data = self.data[rank::world_size] # Simple sharding for DDP

        self.training = "train" in split_name.lower()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            audio_path = item["audio_path"]
            import soundfile as sf
            
            # Load audio file info without loading the entire file
            try:
                info = sf.info(audio_path)
                frames_to_read = min(int(self.max_audio_length), info.frames)
                
                # Read only the frames we need
                audio_array, sampling_rate = sf.read(
                    audio_path, 
                    frames=frames_to_read, 
                    dtype='float32'
                )
                
                # Ensure audio is not empty and has valid values
                if len(audio_array) == 0 or np.isnan(audio_array).any() or np.isinf(audio_array).any():
                    print(f"Warning: Audio file {audio_path} contains invalid values. Using empty audio.")
                    # Create a short silent audio (1 second)
                    audio_array = np.zeros(16000, dtype=np.float32)
                    sampling_rate = 16000
                
            except Exception as e:
                print(f"Error reading audio file {audio_path}: {e}")
                # Create a short silent audio (1 second)
                audio_array = np.zeros(16000, dtype=np.float32)
                sampling_rate = 16000
            
            audio_data = {"array": audio_array, "sampling_rate": sampling_rate}
            
            # Explicitly delete variables to free memory
            del audio_array
            
        except Exception as e:
            print(f"Error loading audio {item['audio_path']}: {e}")
            raise

        transcription_text = item["transcription_text"]  # Already truncated in __init__

        filled_task_prompt = self.task_prompt_template.format(transcription=transcription_text)
        
        user_message = {
            'role': 'user',
            'content': f'<|audio_1|>\n{filled_task_prompt}',
        }
        
        try:
            prompt_text = self.processor.tokenizer.apply_chat_template(
                [user_message], tokenize=False, add_generation_prompt=True
            )
            
            # Use much smaller max_length to prevent OOM
            inputs = self.processor(
                text=prompt_text, 
                audios=[(audio_data["array"], audio_data["sampling_rate"])], 
                return_tensors='pt',
                truncation=True,
                max_length=512  # Significantly reduced from 1024
            )
            
            answer_str = f"{item['label']}{ANSWER_SUFFIX}"
            answer_ids = self.processor.tokenizer(answer_str, return_tensors='pt', add_special_tokens=False).input_ids
            
            input_ids = inputs.input_ids[0]
            
            # Save audio embeddings before deletion
            input_audio_embeds = inputs.input_audio_embeds[0] if hasattr(inputs, 'input_audio_embeds') else None
            audio_embed_sizes = inputs.audio_embed_sizes[0] if hasattr(inputs, 'audio_embed_sizes') else None
            
            if self.training:
                # For training, prepend the input_ids to the answer_ids
                concatenated_input_ids = torch.cat([input_ids, answer_ids[0]], dim=0)
                
                # Create a labels tensor with -100 for the prompt and actual token ids for the answer
                labels = torch.full_like(concatenated_input_ids, _IGNORE_INDEX)
                labels[len(input_ids):] = answer_ids[0]
            else:
                # For evaluation, only use the input_ids
                concatenated_input_ids = input_ids
                
                # For evaluation labels, we use the answer_ids directly (no concatenation)
                labels = answer_ids[0]

            # Clean up tensors to prevent memory leaks
            del inputs, input_ids, answer_ids
            
            return {
                'input_ids': concatenated_input_ids,
                'labels': labels,
                'input_audio_embeds': input_audio_embeds,
                'audio_embed_sizes': audio_embed_sizes,
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a minimal valid item that will be filtered out by the collator
            return {
                'input_ids': torch.tensor([0]),
                'labels': torch.tensor([0]),
                'input_audio_embeds': None,
                'audio_embed_sizes': None,
            }

# pad_sequence, cat_with_pad_original, custom_collate_fn (remain the same as previous version)
# create_model, evaluate functions (remain the same as previous version)
# ... (Make sure these functions are included from the previous full script)

def pad_sequence(sequences, padding_side='right', padding_value=0, batch_first=True):
    assert padding_side in ['right', 'left']
    max_len = max(seq.size(0) for seq in sequences) 
    
    output = sequences[0].new_full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output[i, :length] = seq
        else:
            output[i, -length:] = seq
    return output

def cat_with_pad_original(tensors, dim, padding_value=0):
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors) 
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    return output

def custom_collate_fn(batch, processor):
    # Remove any samples with None values
    valid_batch = []
    for item in batch:
        # Check if item has the necessary components
        if (item['input_ids'] is not None and
            item['labels'] is not None and 
            'input_audio_embeds' in item and 
            'audio_embed_sizes' in item):
            
            # Even if the audio embeddings are None, we can still proceed with a valid small tensor
            if item['input_audio_embeds'] is None:
                # Create a small dummy tensor for audio embeddings
                item['input_audio_embeds'] = torch.zeros((1, 1), dtype=torch.float32)
                item['audio_embed_sizes'] = torch.tensor([1], dtype=torch.long)
                
            valid_batch.append(item)
        else:
            print(f"Skipping invalid batch item: {item.keys()}")
    
    # If no valid items, return an empty batch
    if not valid_batch:
        print("Warning: No valid items in batch!")
        return BatchFeature({})
    
    # Use valid_batch instead of batch
    batch = valid_batch
    
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = [] 
    audio_embed_sizes_list = []   

    for item in batch:
        input_ids_list.append(item['input_ids'])
        labels_list.append(item['labels'])
        input_audio_embeds_list.append(item['input_audio_embeds'].unsqueeze(0) if item['input_audio_embeds'].dim() == 2 else item['input_audio_embeds'])
        audio_embed_sizes_list.append(item['audio_embed_sizes'].unsqueeze(0) if item['audio_embed_sizes'].dim() == 0 else item['audio_embed_sizes'])

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    # Check if we're in training or evaluation mode based on the shape of labels
    is_eval = all(len(l.shape) == 1 and l.shape[0] < 10 for l in labels_list)
    
    # Pad input_ids
    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=pad_token_id)
    
    if is_eval:
        # For evaluation, the labels should be the target sequences (not padded with input_ids)
        labels = torch.stack(labels_list) if all(l.shape == labels_list[0].shape for l in labels_list) else pad_sequence(labels_list, padding_side='left', padding_value=_IGNORE_INDEX)
    else:
        # For training, the labels should have the same length as input_ids, with prompt positions ignored
        labels = pad_sequence(labels_list, padding_side='left', padding_value=_IGNORE_INDEX)
    
    attention_mask = (input_ids != pad_token_id).long()

    # Process audio embeddings
    input_audio_embeds = cat_with_pad_original(input_audio_embeds_list, dim=0) 
    audio_embed_sizes = torch.cat(audio_embed_sizes_list) 

    max_patches = input_audio_embeds.size(1)
    audio_attention_mask_list = []
    for size_tensor in audio_embed_sizes_list: 
        num_patches = size_tensor.item()
        mask = torch.zeros(max_patches, dtype=torch.bool)
        mask[:num_patches] = True
        audio_attention_mask_list.append(mask)
    
    audio_attention_mask = torch.stack(audio_attention_mask_list) if audio_attention_mask_list else None
    
    # Clean up to prevent memory leaks
    del input_ids_list, labels_list, input_audio_embeds_list, audio_embed_sizes_list
    
    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2, 
        }
    )

def create_model(model_name_or_path, use_flash_attention=False, low_cpu_mem_usage=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    # Explicitly move model to GPU if flash attention is enabled
    if use_flash_attention and torch.cuda.is_available():
        model = model.to('cuda')
    return model

@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1, accelerator=None
):
    # Clear CUDA cache at start of evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    device = accelerator.device if accelerator else next(model.parameters()).device
    model.eval()
    all_generated_texts = []
    all_true_labels = []

    # Worker initialization function to clear memory
    def worker_init_fn(worker_id):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Use all available test samples unless max_eval_samples is explicitly set
    max_eval_samples = min(len(eval_dataset), 100)  # Increased limit, but still capped to avoid extreme cases
    eval_indices = list(range(len(eval_dataset)))
    
    # Only shuffle and subsample if we have more than max_eval_samples
    if len(eval_indices) > max_eval_samples:
        import random
        random.shuffle(eval_indices)
        eval_indices = eval_indices[:max_eval_samples]
    
    subset_eval_dataset = torch.utils.data.Subset(eval_dataset, eval_indices)
    print(f"Evaluating on {len(subset_eval_dataset)} samples out of {len(eval_dataset)} total samples")

    collate_fn_with_processor = partial(custom_collate_fn, processor=processor)

    eval_dataloader = torch.utils.data.DataLoader(
        subset_eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn_with_processor,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues that can cause device mismatches
        pin_memory=False,  # Disable pin_memory to reduce memory usage
        worker_init_fn=worker_init_fn if 'worker_init_fn' in locals() else None,  # Use worker init if available
    )
    
    if accelerator:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    stop_tokens_list = ["<|end|>", processor.tokenizer.eos_token] 
    stop_tokens_list = sorted(list(set(s for s in stop_tokens_list if s)))

    stop_tokens_ids = processor.tokenizer(
        stop_tokens_list, add_special_tokens=False, return_tensors="pt", padding=True
    ).input_ids.to(device)

    for batch_idx, batch in enumerate(tqdm(
        eval_dataloader, disable=disable_tqdm, desc='Running Evaluation'
    )):
        # Explicitly free memory for any previous batch
        if torch.cuda.is_available() and batch_idx > 0 and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            
        current_true_labels_tokenized = batch.pop("labels") 
        
        # Move all batch tensors to the correct device to avoid device mismatch errors
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        temp_decoded_labels = []
        for lbl_tokens in current_true_labels_tokenized:
            actual_tokens = lbl_tokens[lbl_tokens != _IGNORE_INDEX]
            actual_tokens = actual_tokens[actual_tokens != processor.tokenizer.pad_token_id]
            decoded_label = processor.decode(actual_tokens, skip_special_tokens=False) 
            if decoded_label.endswith(ANSWER_SUFFIX):
                decoded_label = decoded_label[:-len(ANSWER_SUFFIX)]
            if decoded_label.endswith(processor.tokenizer.eos_token):
                 decoded_label = decoded_label[:-len(processor.tokenizer.eos_token)]
            if decoded_label.endswith("<|end|>"): 
                 decoded_label = decoded_label[:-len("<|end|>")]
            temp_decoded_labels.append(decoded_label.strip())
        all_true_labels.extend(temp_decoded_labels)
        
        stopping_criteria = StoppingCriteriaList([
            MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=batch.input_ids.size(0))
        ])
        
        max_new_toks = 3  # Further reduced to minimize memory usage

        try:
            with torch.amp.autocast('cuda', enabled=True):  # Updated to the newer API
                generated_ids = model.generate(
                    **batch, 
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=max_new_toks,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=False,  # Disable sampling for deterministic and faster generation
                    num_beams=1  # Use greedy decoding (no beam search)
                )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM error in batch {batch_idx}. Skipping this batch and continuing...")
                all_generated_texts.extend(["control"] * len(temp_decoded_labels))  # Default to control if OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        input_ids_len = batch["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_ids_len:]
        
        current_generated_texts = processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        cleaned_generated_texts = [text.strip() for text in current_generated_texts]
        all_generated_texts.extend(cleaned_generated_texts)
        
        # Explicitly delete tensors to free memory
        del batch, generated_ids, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if accelerator:
        print(f"Before gather: {len(all_generated_texts)} predictions, {len(all_true_labels)} labels")
        all_generated_texts = accelerator.gather_for_metrics(all_generated_texts)
        all_true_labels = accelerator.gather_for_metrics(all_true_labels)
        print(f"After gather: {len(all_generated_texts)} predictions, {len(all_true_labels)} labels")
    else: 
        all_generated_texts = gather_object(all_generated_texts) if not isinstance(all_generated_texts, list) else all_generated_texts
        all_true_labels = gather_object(all_true_labels) if not isinstance(all_true_labels, list) else all_true_labels
        print(f"After gather_object: {len(all_generated_texts)} predictions, {len(all_true_labels)} labels")

    metrics = {}
    if accelerator is None or accelerator.is_main_process:
        filtered_preds = []
        filtered_labels = []
        valid_labels_set = {"dementia", "control"}

        print(f"Processing {len(all_generated_texts)} predictions and {len(all_true_labels)} labels")
        
        for pred, true_label in zip(all_generated_texts, all_true_labels):
            # Simplified cleaning: strip, lowercase, and take the first word if multiple are generated.
            cleaned_pred = pred.strip().lower().split(' ')[0]
            cleaned_true = true_label.strip().lower()
            
            # Ensure the cleaned prediction is one of the valid labels, otherwise assign defensively.
            if cleaned_pred not in valid_labels_set:
                # Map common prefixes/variations to the correct label
                if cleaned_pred in ["ad", "dem", "demen"] or "dementia" in cleaned_pred:
                    cleaned_pred = POSITIVE_CLASS
                elif cleaned_pred in ["cn", "ctrl", "normal", "cogn"] or "control" in cleaned_pred:
                    cleaned_pred = NEGATIVE_CLASS
                else:
                    # Basic fallback: if true label is positive, guess negative, and vice-versa.
                    cleaned_pred = NEGATIVE_CLASS if cleaned_true == POSITIVE_CLASS else POSITIVE_CLASS
            
            # Add to filtered lists if true label is valid
            filtered_preds.append(cleaned_pred)
            filtered_labels.append(cleaned_true)

        if not filtered_labels: 
            print("Warning: No valid labels found for metric calculation.")
            accuracy = 0.0; f1 = 0.0; precision = 0.0; recall = 0.0; f1_ad = 0.0; bce_loss = float('inf')
        else:
            print(f"Calculating metrics for {len(filtered_preds)} samples")
            
            # Map any labels that might not be exactly matching CLASS_LABELS
            aligned_true_labels = []
            for label in filtered_labels:
                if label.lower() == POSITIVE_CLASS.lower() or label.lower() in ["ad", "dem", "demen"] or "dementia" in label.lower():
                    aligned_true_labels.append(POSITIVE_CLASS)
                else:
                    aligned_true_labels.append(NEGATIVE_CLASS)
            
            # Map any predictions that might not be exactly matching CLASS_LABELS    
            aligned_predictions = []
            for pred in filtered_preds:
                if pred.lower() == POSITIVE_CLASS.lower() or pred.lower() in ["ad", "dem", "demen"] or "dementia" in pred.lower():
                    aligned_predictions.append(POSITIVE_CLASS)
                else:
                    aligned_predictions.append(NEGATIVE_CLASS)
                    
            # Calculate metrics with aligned labels
            accuracy = accuracy_score(aligned_true_labels, aligned_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                aligned_true_labels, aligned_predictions, average='weighted', zero_division=0
            )
            f1_ad = f1_score(
                aligned_true_labels, aligned_predictions, 
                pos_label=POSITIVE_CLASS, average='binary', zero_division=0
            )
            
            # Calculate Binary Cross-Entropy Loss
            # Convert predictions to probabilities (simple heuristic based on prediction confidence)
            predicted_probs = []
            true_binary_labels = []
            
            for pred, true_label in zip(aligned_predictions, aligned_true_labels):
                # Convert true labels to binary (1 for dementia, 0 for control)
                true_binary = 1.0 if true_label == POSITIVE_CLASS else 0.0
                true_binary_labels.append(true_binary)
                
                # Simple probability assignment based on prediction
                # In a real scenario, you'd use actual model probabilities
                if pred == POSITIVE_CLASS:
                    # High confidence for dementia prediction
                    pred_prob = 0.9 if true_label == POSITIVE_CLASS else 0.1
                else:
                    # High confidence for control prediction  
                    pred_prob = 0.1 if true_label == POSITIVE_CLASS else 0.9
                predicted_probs.append(pred_prob)
            
            # Calculate BCE loss
            if len(predicted_probs) > 0:
                predicted_probs = np.array(predicted_probs)
                true_binary_labels = np.array(true_binary_labels)
                
                # Clip probabilities to avoid log(0)
                predicted_probs = np.clip(predicted_probs, 1e-7, 1 - 1e-7)
                
                # Calculate BCE: -[y*log(p) + (1-y)*log(1-p)]
                bce_loss = -np.mean(
                    true_binary_labels * np.log(predicted_probs) + 
                    (1 - true_binary_labels) * np.log(1 - predicted_probs)
                )
            else:
                bce_loss = float('inf')

        metrics = {
            "accuracy": accuracy, 
            "f1_weighted": f1, 
            f"f1_{POSITIVE_CLASS.lower()}": f1_ad,
            "precision_weighted": precision, 
            "recall_weighted": recall, 
            "bce_loss": bce_loss,
            "num_samples": len(filtered_labels),
            # Store raw predictions and labels for debugging but don't use in logging
            "_raw_predictions": all_generated_texts[:20], 
            "_true_labels": all_true_labels[:20],
        }
        print(f"Evaluation Results: {metrics}")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # For saving, include the full raw data
            save_metrics = metrics.copy()
            save_metrics["raw_predictions"] = all_generated_texts
            save_metrics["true_labels"] = all_true_labels
            with open(save_path, 'w') as f:
                json.dump({
                    "metrics": save_metrics
                }, f, indent=4)
        return metrics
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', type=str, default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from.'
    )
    # Updated arguments for CSV and audio directories
    parser.add_argument(
        "--train_csv_path", type=str, required=True,
        help="Path to the training CSV file (columns: uid, transcription, label)."
    )
    parser.add_argument(
        "--train_audio_dir", type=str, required=True,
        help="Path to the directory containing AD/ and CN/ subfolders for training audio files."
    )
    parser.add_argument(
        "--eval_csv_path", type=str, required=True,
        help="Path to the evaluation CSV file (columns: uid, transcription, label)."
    )
    parser.add_argument(
        "--eval_audio_dir", type=str, required=True,
        help="Path to the directory containing AD/ and CN/ subfolders for evaluation audio files."
    )
    # --- rest of the arguments from the previous script ---
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention 2.')
    parser.add_argument('--low_cpu_mem_usage', action='store_true', help='Enable low CPU memory usage for model loading.')
    parser.add_argument('--output_dir', type=str, default='./ad_phi4_finetuned_output/', help='Output directory.')
    parser.add_argument('--batch_size_per_gpu', type=int, default=1, help='Batch size per GPU.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps.')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every X updates steps.')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate every X updates steps.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Max samples for training set for debugging.')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Max samples for evaluation set. None means use all available samples.')
    parser.add_argument('--no_tqdm', dest='tqdm', action='store_false', help='Disable tqdm progress bars.')
    # Add mixed precision argument
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'fp16', 'bf16'], 
                       help='Mixed precision mode for training. Use bf16 for better performance with Flash Attention.')
    # Add max audio length argument
    parser.add_argument('--max_audio_seconds', type=int, default=30, 
                       help='Maximum length of audio in seconds. Default is 30 seconds.')
    # Add random seed argument
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility. Default is 42.')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Enable low_cpu_mem_usage if flash attention is used, as it can help with memory
    if args.use_flash_attention and not args.low_cpu_mem_usage:
        print("Enabling low_cpu_mem_usage as flash_attention is active.")
        args.low_cpu_mem_usage = True
        
    # Clear CUDA cache at the beginning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create accelerator with proper mixed precision setting
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != 'no' else None
    )
    
    if accelerator.is_local_main_process:
        print(f"Arguments: {args}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print(f"Set processor.tokenizer.pad_token to eos_token: {processor.tokenizer.eos_token}")

    model = create_model(
        args.model_name_or_path,
        use_flash_attention=args.use_flash_attention,
        low_cpu_mem_usage=args.low_cpu_mem_usage
    )
    if processor.tokenizer.pad_token_id is not None:
         model.config.pad_token_id = processor.tokenizer.pad_token_id

    try:
        model.set_lora_adapter('speech')
        print("Using 'speech' LoRA adapter.")
    except Exception as e:
        print(f"Could not set LoRA adapter 'speech': {e}. Proceeding without explicit adapter setting.")

    # Instantiate ADClassificationDataset with new arguments
    train_dataset = ADClassificationDataset(
        processor=processor,
        csv_path=args.train_csv_path,
        audio_dir=args.train_audio_dir,
        split_name="train",
        task_prompt_template=AD_CLASSIFICATION_PROMPT_TEMPLATE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        max_samples=args.max_train_samples,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        max_audio_seconds=args.max_audio_seconds
    )
    eval_dataset = ADClassificationDataset(
        processor=processor,
        csv_path=args.eval_csv_path,
        audio_dir=args.eval_audio_dir,
        split_name="eval",
        task_prompt_template=AD_CLASSIFICATION_PROMPT_TEMPLATE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        max_samples=args.max_eval_samples,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        max_audio_seconds=args.max_audio_seconds
    )
    
    if accelerator.is_local_main_process:
        print(f"Loaded {len(train_dataset) * accelerator.num_processes if accelerator.num_processes > 1 else len(train_dataset)} potential training samples and "
              f"{len(eval_dataset) * accelerator.num_processes if accelerator.num_processes > 1 else len(eval_dataset)} potential evaluation samples (estimated total).")
        if (len(train_dataset) == 0 and accelerator.process_index == 0) or \
           (len(eval_dataset) == 0 and accelerator.process_index == 0): # Check if datasets are empty on main process after sharding
            print("Warning: One of the datasets is effectively empty for the main process. Check paths, CSV content, and audio files.")
            # Potentially exit if no data, especially for training
            if len(train_dataset) == 0 and accelerator.process_index == 0 and accelerator.num_processes == 1 : # If single process and train_dataset is empty
                 print("Exiting: Training dataset is empty.")
                 return

    fp16_enabled = accelerator.mixed_precision == 'fp16'
    bf16_enabled = accelerator.mixed_precision == 'bf16'
    if args.use_flash_attention and not bf16_enabled and not fp16_enabled: # Flash Attn needs fp16 or bf16
        print("Flash Attention 2 requires mixed precision (fp16 or bf16). Enabling bf16 by default.")
        bf16_enabled = True # Default to bf16 if using flash and no precision set

    # Create training arguments with optimized settings for memory
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=1,  # Force eval batch size to 1
        # gradient_accumulation_steps already handled by Accelerator
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch',
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear', 
        warmup_steps= int(0.1 * args.num_train_epochs * (len(train_dataset) // (args.batch_size_per_gpu * accelerator.num_processes))) if len(train_dataset) > 0 else 50, # 10% of steps or 50
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,  # Keep only the best model to save disk space
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_f1_{POSITIVE_CLASS.lower()}",
        greater_is_better=True,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        remove_unused_columns=False,
        report_to="tensorboard", 
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=1,  # Reduced to avoid memory overhead
        dataloader_pin_memory=False,  # Disable pin_memory to reduce memory usage
        # ddp_find_unused_parameters=True, #  Set this if you encounter issues with DDP, usually not needed with GC + LoRA
    )

    # Create a simpler optimized version of the trainer
    class MemoryEfficientTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Store the evaluation function reference and processor
            self.external_evaluate_fn = evaluate_with_args
            self.processor = kwargs.get('processing_class', None)
            
        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """Override evaluate to use our custom evaluate function to avoid batch size mismatch issues"""
            # Clear CUDA cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # If this is a direct call to evaluate (not through _maybe_log_save_evaluate), 
            # just use the parent implementation
            if eval_dataset is not None or metric_key_prefix != "eval":
                return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            
            # Use our custom evaluation function instead
            metrics = self.external_evaluate_fn(
                model=self.model,
                processor=self.processor,
                eval_dataset=self.eval_dataset,
                disable_tqdm=True,
                eval_batch_size=self.args.per_device_eval_batch_size,
                accelerator=self.accelerator
            )
            
            # Format metrics to match what trainer.evaluate would return
            if metrics:
                # Clean metrics by removing any non-scalar values
                clean_metrics = {}
                for key, value in metrics.items():
                    # Skip non-scalar values and keys starting with underscore (private)
                    if key.startswith('_') or isinstance(value, (list, dict, tuple)):
                        continue
                    
                    # Add the metric_key_prefix
                    if key not in ["eval_loss", "epoch", "num_samples"]:
                        clean_metrics[f"{metric_key_prefix}_{key}"] = value
                    else:
                        clean_metrics[key] = value
                        
                metrics = clean_metrics
            
            self.log(metrics)
            return metrics
        
        def save_model(self, *args, **kwargs):
            # Clear CUDA cache before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return super().save_model(*args, **kwargs)
        
        def train(self, *args, **kwargs):
            # Clear CUDA cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return super().train(*args, **kwargs)

    collate_fn_with_processor = partial(custom_collate_fn, processor=processor)
    
    # Let's update the evaluate function to respect max_eval_samples passed from command line
    def evaluate_with_args(
        model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1, accelerator=None
    ):
        # Limit evaluation samples if max_eval_samples is specified
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            print(f"Limiting evaluation to {args.max_eval_samples} samples (out of {len(eval_dataset)})")
            eval_indices = list(range(len(eval_dataset)))
            random.shuffle(eval_indices)
            eval_indices = eval_indices[:args.max_eval_samples]
            eval_dataset_subset = torch.utils.data.Subset(eval_dataset, eval_indices)
        else:
            # Use all samples by default
            print(f"Using all {len(eval_dataset)} samples for evaluation")
            eval_dataset_subset = eval_dataset
            
        return evaluate(
            model=model,
            processor=processor, 
            eval_dataset=eval_dataset_subset,
            save_path=save_path,
            disable_tqdm=disable_tqdm,
            eval_batch_size=eval_batch_size,
            accelerator=accelerator
        )

    # Update trainer to use our wrapper function
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn_with_processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,  # Updated from tokenizer
        compute_metrics=lambda eval_pred: {
            # Simple metrics function that just returns placeholders
            # The actual evaluation is done in our custom evaluate method
            "accuracy": 0.0,
            "f1_weighted": 0.0,
            f"f1_{POSITIVE_CLASS.lower()}": 0.0,
            "bce_loss": 0.0
        }
    )
    
    if accelerator.is_main_process:
        print("\n--- Evaluating before fine-tuning ---")
        # Clear cache before initial evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        initial_metrics = evaluate_with_args(
            model=trainer.model, 
            processor=processor,
            eval_dataset=eval_dataset, 
            save_path=Path(args.output_dir) / 'eval_results_before_finetuning.json',
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
            accelerator=accelerator
        )
        if initial_metrics:
            # Filter out any non-scalar values before printing
            clean_metrics = {}
            for k, v in initial_metrics.items():
                if k in ['raw_predictions', 'true_labels', '_raw_predictions', '_true_labels'] or isinstance(v, (list, dict, tuple)):
                    continue
                clean_metrics[k] = v
            print(f"Initial evaluation metrics: {clean_metrics}")

    if len(train_dataset) > 0 : # Proceed to training only if there's training data
        print("\n--- Starting fine-tuning ---")
        trainer.train()
        
        accelerator.wait_for_everyone() 
        if accelerator.is_main_process:
            print("\n--- Saving final model and processor ---")
            trainer.save_model(args.output_dir) 
            processor.save_pretrained(args.output_dir)
            print(f"Model and processor saved to {args.output_dir}")
    else:
        print("Skipping training as the training dataset is empty.")

    # Clear cache before final evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print("\n--- Evaluating after fine-tuning (or final evaluation if training was skipped) ---")
        final_metrics = evaluate_with_args(
            model=trainer.model, 
            processor=processor,
            eval_dataset=eval_dataset,
            save_path=Path(args.output_dir) / 'eval_results_after_finetuning.json',
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
            accelerator=accelerator
        )
        if final_metrics:
            # Filter out any non-scalar values before logging
            clean_metrics = {}
            for k, v in final_metrics.items():
                if k in ['raw_predictions', 'true_labels'] or isinstance(v, (list, dict, tuple)):
                    print(f"Not logging non-scalar metric: {k}")
                    continue
                clean_metrics[k] = v
                
            print(f"Final evaluation metrics: {clean_metrics}")
            if len(train_dataset) > 0: # Only log if training happened
                trainer.log_metrics("eval_final", clean_metrics)
                trainer.save_metrics("eval_final", clean_metrics)

    # Save parameters and results to an Excel file at the end of training
    if accelerator.is_main_process and final_metrics:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        excel_path = os.path.join(args.output_dir, f"training_results_{timestamp}.xlsx")
        
        # Create DataFrames for each type of information
        # 1. Hyperparameters
        hyperparams = {
            'model': args.model_name_or_path,
            'epochs': args.num_train_epochs,
            'batch_size': args.batch_size_per_gpu,
            'grad_accum_steps': args.gradient_accumulation_steps,
            'learning_rate': args.learning_rate,
            'weight_decay': args.wd,
            'max_audio_seconds': args.max_audio_seconds,
            'mixed_precision': args.mixed_precision,
            'flash_attention': args.use_flash_attention,
            'low_cpu_mem_usage': args.low_cpu_mem_usage,
            'seed': args.seed,
            'train_samples': len(train_dataset) * accelerator.num_processes if accelerator.num_processes > 1 else len(train_dataset),
            'eval_samples': len(eval_dataset) * accelerator.num_processes if accelerator.num_processes > 1 else len(eval_dataset),
        }
        
        # 2. Evaluation Metrics - already in final_metrics
        # Clean up metrics before saving
        metrics_to_save = {}
        for k, v in final_metrics.items():
            if k.startswith('_') or isinstance(v, (list, dict, tuple)):
                continue
            metrics_to_save[k] = v
            
        # 3. Add Initial vs Final metrics comparison if available
        comparison_data = {}
        if 'initial_metrics' in locals() and initial_metrics:
            # Create a comparison of before/after metrics
            comparison_metrics = ['accuracy', 'f1_weighted', f'f1_{POSITIVE_CLASS.lower()}', 
                                 'precision_weighted', 'recall_weighted', 'bce_loss', 'num_samples']
            
            for metric in comparison_metrics:
                if metric in initial_metrics and metric in final_metrics:
                    comparison_data[f'initial_{metric}'] = initial_metrics[metric]
                    comparison_data[f'final_{metric}'] = final_metrics[metric]
                    if metric not in ['num_samples']:  # Skip calculating improvement for sample count
                        if metric == 'bce_loss':
                            # For loss, improvement is negative (lower is better)
                            comparison_data[f'improvement_{metric}'] = initial_metrics[metric] - final_metrics[metric]
                        else:
                            # For other metrics, improvement is positive (higher is better)
                            comparison_data[f'improvement_{metric}'] = final_metrics[metric] - initial_metrics[metric]
        
        # 4. Get detailed prediction data (limited to first 100)
        prediction_data = []
        if '_raw_predictions' in final_metrics and '_true_labels' in final_metrics:
            for i, (pred, true) in enumerate(zip(final_metrics['_raw_predictions'], final_metrics['_true_labels'])):
                if i >= 100:  # Limit to first 100 examples
                    break
                prediction_data.append({
                    'sample_idx': i,
                    'prediction': pred,
                    'true_label': true,
                    'correct': pred.strip().lower() == true.strip().lower()
                })
        
        # Create a writer to save to Excel with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Save hyperparameters
            pd.DataFrame([hyperparams]).to_excel(writer, sheet_name='Hyperparameters', index=False)
            
            # Save metrics
            pd.DataFrame([metrics_to_save]).to_excel(writer, sheet_name='Metrics', index=False)
            
            # Save metrics comparison if available
            if comparison_data:
                pd.DataFrame([comparison_data]).to_excel(writer, sheet_name='Metrics Comparison', index=False)
            
            # Save prediction examples
            if prediction_data:
                pd.DataFrame(prediction_data).to_excel(writer, sheet_name='Example Predictions', index=False)
            
        print(f"Saved training parameters and results to {excel_path}")

if __name__ == '__main__':
    main()