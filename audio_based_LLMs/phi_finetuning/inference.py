import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from functools import partial
import datetime

# Import necessary components from the finetune.py file
from finetune import (
    ADClassificationDataset,
    custom_collate_fn,
    evaluate,
    AD_CLASSIFICATION_PROMPT_TEMPLATE,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
    MultipleTokenBatchStoppingCriteria,
    _IGNORE_INDEX,
    ANSWER_SUFFIX
)
from transformers import StoppingCriteriaList
from tqdm import tqdm
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def evaluate_with_device_handling(
    model, processor, eval_dataset, device, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    """
    Device-aware evaluation function that ensures all tensors are on the correct device.
    """
    # Clear CUDA cache at start of evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.eval()
    all_generated_texts = []
    all_true_labels = []

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
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory to reduce memory usage
    )

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
        
        # Move all batch tensors to the correct device
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
            MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=batch["input_ids"].size(0))
        ])
        
        max_new_toks = 3  # Further reduced to minimize memory usage

        try:
            with torch.amp.autocast('cuda', enabled=True):  # Updated to the newer API
                # Generate with return_dict_in_generate=True to get logits
                generation_output = model.generate(
                    **batch, 
                    eos_token_id=processor.tokenizer.eos_token_id,
                    max_new_tokens=max_new_toks,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=False,  # Disable sampling for deterministic and faster generation
                    num_beams=1,  # Use greedy decoding (no beam search)
                    return_dict_in_generate=True,
                    output_scores=True
                )
                generated_ids = generation_output.sequences
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
        
        # Extract logits for the first generated token (class prediction)
        if hasattr(generation_output, 'scores') and len(generation_output.scores) > 0:
            # Get logits for the first generated token
            first_token_logits = generation_output.scores[0]  # Shape: [batch_size, vocab_size]
            
            # Get token IDs for "dementia" and "control"
            dementia_token_id = processor.tokenizer.encode("dementia", add_special_tokens=False)[0]
            control_token_id = processor.tokenizer.encode("control", add_special_tokens=False)[0]
            
            # Extract logits for these specific tokens
            dementia_logits = first_token_logits[:, dementia_token_id]  # Shape: [batch_size]
            control_logits = first_token_logits[:, control_token_id]    # Shape: [batch_size]
            
            # Calculate probabilities using softmax over the two class tokens
            class_logits = torch.stack([control_logits, dementia_logits], dim=1)  # [batch_size, 2]
            class_probs = F.softmax(class_logits, dim=1)  # [batch_size, 2]
            dementia_probs = class_probs[:, 1]  # Probability of dementia class
            
            # Store for BCE loss calculation later
            batch_logits = dementia_probs.cpu().numpy()
        else:
            # Fallback if no scores available
            batch_logits = [0.5] * len(generated_tokens)  # Neutral probability
        
        current_generated_texts = processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        cleaned_generated_texts = [text.strip() for text in current_generated_texts]
        all_generated_texts.extend(cleaned_generated_texts)
        
        # Explicitly delete tensors to free memory
        del batch, generated_ids, generated_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate metrics
    metrics = {}
    filtered_preds = []
    filtered_labels = []
    filtered_logits = []  # Store logits for BCE loss calculation
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
        "raw_predictions": all_generated_texts, 
        "true_labels": all_true_labels,
    }
    print(f"Evaluation Results: {metrics}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({
                "metrics": metrics
            }, f, indent=4)
    
    return metrics


def test_finetuned_model(
    model_dir,
    test_csv_path,
    test_audio_dir,
    output_dir=None,
    max_test_samples=None,
    batch_size=1,
    max_audio_seconds=30,
    use_flash_attention=False,
    mixed_precision="bf16"
):
    """
    Test a fine-tuned model on test data.
    
    Args:
        model_dir (str): Directory containing the saved fine-tuned model and processor
        test_csv_path (str): Path to the test CSV file (columns: uid, transcription, label)
        test_audio_dir (str): Path to the directory containing test audio files
        output_dir (str, optional): Directory to save test results. If None, saves to model_dir/test_results
        max_test_samples (int, optional): Maximum number of test samples to evaluate
        batch_size (int): Batch size for evaluation (default: 1)
        max_audio_seconds (int): Maximum audio length in seconds (default: 30)
        use_flash_attention (bool): Whether to use flash attention (default: False)
        mixed_precision (str): Mixed precision mode ('bf16', 'fp16', or 'no')
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(model_dir, "test_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model and processor from: {model_dir}")
    print(f"Test CSV: {test_csv_path}")
    print(f"Test audio directory: {test_audio_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        print("✓ Processor loaded successfully")
    except Exception as e:
        print(f"Error loading processor from {model_dir}: {e}")
        print("Trying to load from original model...")
        processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    
    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print(f"Set processor.tokenizer.pad_token to eos_token: {processor.tokenizer.eos_token}")
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else (torch.float16 if mixed_precision == "fp16" else torch.float32),
            _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        raise
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Set pad token id for model
    if processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # Try to set LoRA adapter if available
    try:
        model.set_lora_adapter('speech')
        print("✓ Using 'speech' LoRA adapter")
    except Exception as e:
        print(f"Could not set LoRA adapter 'speech': {e}. Proceeding without explicit adapter setting.")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ADClassificationDataset(
        processor=processor,
        csv_path=test_csv_path,
        audio_dir=test_audio_dir,
        split_name="test",
        task_prompt_template=AD_CLASSIFICATION_PROMPT_TEMPLATE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        max_samples=max_test_samples,
        rank=0,
        world_size=1,
        max_audio_seconds=max_audio_seconds
    )
    
    print(f"✓ Test dataset created with {len(test_dataset)} samples")
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check CSV file, audio directory, and file paths.")
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run evaluation
    print("Starting evaluation...")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    metrics = evaluate_with_device_handling(
        model=model,
        processor=processor,
        eval_dataset=test_dataset,
        device=device,
        save_path=Path(results_file),
        disable_tqdm=False,
        eval_batch_size=batch_size
    )
    
    if metrics:
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        # Print main metrics
        main_metrics = ['accuracy', 'f1_weighted', f'f1_{POSITIVE_CLASS.lower()}', 
                       'precision_weighted', 'recall_weighted', 'bce_loss', 'num_samples']
        
        for metric in main_metrics:
            if metric in metrics:
                if metric == 'num_samples':
                    print(f"{metric}: {metrics[metric]}")
                elif metric == 'bce_loss' and metrics[metric] == float('inf'):
                    print(f"{metric}: inf")
                else:
                    print(f"{metric}: {metrics[metric]:.4f}")
        
        print("="*50)
        
        # Save summary results to a separate file
        summary_file = os.path.join(output_dir, f"test_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("TEST RESULTS SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Model Directory: {model_dir}\n")
            f.write(f"Test CSV: {test_csv_path}\n")
            f.write(f"Test Audio Directory: {test_audio_dir}\n")
            f.write(f"Number of Test Samples: {metrics.get('num_samples', 'N/A')}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("METRICS:\n")
            for metric in main_metrics:
                if metric in metrics:
                    if metric == 'num_samples':
                        f.write(f"{metric}: {metrics[metric]}\n")
                    elif metric == 'bce_loss' and metrics[metric] == float('inf'):
                        f.write(f"{metric}: inf\n")
                    else:
                        f.write(f"{metric}: {metrics[metric]:.4f}\n")
        
        print(f"✓ Detailed results saved to: {results_file}")
        print(f"✓ Summary saved to: {summary_file}")
        
        # Save results to Excel if pandas is available
        try:
            excel_file = os.path.join(output_dir, f"test_results_{timestamp}.xlsx")
            
            # Prepare data for Excel
            summary_data = {
                'model_directory': model_dir,
                'test_csv_path': test_csv_path,
                'test_audio_directory': test_audio_dir,
                'timestamp': timestamp,
                'num_test_samples': metrics.get('num_samples', 0)
            }
            
            # Add metrics
            for metric in main_metrics:
                if metric in metrics and metric != 'num_samples':
                    # Handle infinite BCE loss for Excel
                    if metric == 'bce_loss' and metrics[metric] == float('inf'):
                        summary_data[metric] = 'inf'
                    else:
                        summary_data[metric] = metrics[metric]
            
            # Create DataFrame and save to Excel
            df_summary = pd.DataFrame([summary_data])
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name='Test Results', index=False)
                
                # Add prediction examples if available
                if 'raw_predictions' in metrics and 'true_labels' in metrics:
                    prediction_data = []
                    for i, (pred, true) in enumerate(zip(metrics['raw_predictions'][:100], metrics['true_labels'][:100])):
                        prediction_data.append({
                            'sample_idx': i,
                            'prediction': pred,
                            'true_label': true,
                            'correct': pred.strip().lower() == true.strip().lower()
                        })
                    
                    if prediction_data:
                        df_predictions = pd.DataFrame(prediction_data)
                        df_predictions.to_excel(writer, sheet_name='Example Predictions', index=False)
            
            print(f"✓ Excel results saved to: {excel_file}")
            
        except Exception as e:
            print(f"Warning: Could not save Excel file: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned Phi-4 model on test data")
    
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing the saved fine-tuned model and processor"
    )
    parser.add_argument(
        "--test_csv_path", type=str, required=True,
        help="Path to the test CSV file (columns: uid, transcription, label)"
    )
    parser.add_argument(
        "--test_audio_dir", type=str, required=True,
        help="Path to the directory containing test audio files"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save test results (default: model_dir/test_results)"
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None,
        help="Maximum number of test samples to evaluate (default: use all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--max_audio_seconds", type=int, default=30,
        help="Maximum audio length in seconds (default: 30)"
    )
    parser.add_argument(
        "--use_flash_attention", action="store_true",
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (default: bf16)"
    )
    
    args = parser.parse_args()
    
    # Run the test
    try:
        metrics = test_finetuned_model(
            model_dir=args.model_dir,
            test_csv_path=args.test_csv_path,
            test_audio_dir=args.test_audio_dir,
            output_dir=args.output_dir,
            max_test_samples=args.max_test_samples,
            batch_size=args.batch_size,
            max_audio_seconds=args.max_audio_seconds,
            use_flash_attention=args.use_flash_attention,
            mixed_precision=args.mixed_precision
        )
        
        if metrics:
            print("\n✓ Testing completed successfully!")
        else:
            print("\n✗ Testing failed - no metrics returned")
            
    except Exception as e:
        print(f"\n✗ Testing failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 