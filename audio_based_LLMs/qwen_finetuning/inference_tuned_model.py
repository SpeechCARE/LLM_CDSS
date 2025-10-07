import warnings
import os
import sys

# Disable all warnings
warnings.filterwarnings("ignore")

# Suppress specific library warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress transformers warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import pandas as pd
import soundfile as sf
import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniModel
from qwen_omni_utils import process_mm_info
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

class AudioClassificationTester2Label:
    def __init__(self, model_path, test_csv_path="dementiabank_test.csv", audio_base_path="data/audio_files"):
        """
        Initialize the 2-label audio classification tester (AD vs CN)
        
        Args:
            model_path: Path to the fine-tuned model
            test_csv_path: Path to the CSV file containing test data
            audio_base_path: Base path to audio files
        """
        self.model_path = model_path
        self.test_csv_path = test_csv_path
        self.audio_base_path = audio_base_path
        
        # Load model and processor
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5OmniModel.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        
        # Load test data
        print(f"Loading test data from {test_csv_path}...")
        self.df = pd.read_csv(test_csv_path)
        print(f"Found {len(self.df)} test samples")
        
        # Define label mapping for 2-label classification
        # Label 1 = Dementia, Label 0 = Control
        self.label_mapping = {
            1: 'dementia',    # Dementia (Alzheimer's Disease or Related Dementia)
            0: 'control'      # Control (Cognitively Normal)
        }
        
        # Define class names for reporting
        self.class_names = ['dementia', 'control']
        self.class_display_names = ['Dementia', 'Control']
        
        # Define the classification prompt for 2-label classification
        self.classification_prompt = (
            "Based on the speech audio and its transcription, "
            "classify the speaker's cognitive status with a single word "
            "from the following options: 'dementia' (for speakers with dementia) or "
            "'control' (for healthy control speakers)."
        )
    
    def create_conversation(self, audio_path, transcription):
        """Create conversation format for the model"""
        conversation = [
            {
                'role': 'system', 
                'content': 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
            },
            {
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f'Transcription: "{transcription}"\n\n{self.classification_prompt}'}
                ]
            }
        ]
        return [conversation]
    
    def predict_single_sample(self, audio_path, transcription):
        """Predict cognitive status for a single audio sample"""
        try:
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return None
            
            # Create conversation
            conversations = self.create_conversation(audio_path, transcription)
            
            # Process input
            text = self.processor.apply_chat_template(
                conversations, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
            inputs = self.processor(
                text=text, 
                audios=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate prediction
            with torch.no_grad():
                text_ids, _ = self.model.generate(
                    **inputs, 
                    use_audio_in_video=False,
                    max_new_tokens=10,  # We only need a short response
                    do_sample=False,    # Use greedy decoding for consistency
                    temperature=0.1
                )
            
            # Decode response
            response = self.processor.batch_decode(
                text_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract prediction from response
            prediction = self.extract_prediction(response)
            return prediction
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def extract_prediction(self, response):
        """Extract the classification prediction from model response"""
        response_lower = response.lower()
        
        # Look for the prediction keywords in the response
        if 'dementia' in response_lower:
            return 'dementia'
        elif 'control' in response_lower:
            return 'control'
        elif 'normal' in response_lower or 'healthy' in response_lower:
            return 'control'
        elif 'alzheimer' in response_lower or 'cognitive' in response_lower:
            return 'dementia'
        else:
            # If no clear prediction found, return the last word
            words = response.strip().split()
            if words:
                last_word = words[-1].lower().strip('.,!?')
                if last_word in ['dementia', 'control']:
                    return last_word
        
        return 'unknown'
    
    def test_subset(self, num_samples=None, start_idx=0):
        """Test on a subset of the data"""
        if num_samples is None:
            test_df = self.df.iloc[start_idx:]
        else:
            end_idx = min(start_idx + num_samples, len(self.df))
            test_df = self.df.iloc[start_idx:end_idx]
        
        predictions = []
        true_labels = []
        results = []
        
        print(f"Testing on {len(test_df)} samples...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing samples"):
            sample_id = row['uid']
            transcription = row['transcription']
            true_label = self.label_mapping.get(row['label'], row['label'])
            
            # Construct audio path - assuming audio files are named with sample ID + .wav
            audio_path = os.path.join(self.audio_base_path, f"{sample_id}.wav")
            
            # Get prediction
            prediction = self.predict_single_sample(audio_path, transcription)
            
            # Handle None predictions by converting to 'unknown'
            if prediction is None:
                prediction = 'unknown'
            
            predictions.append(prediction)
            true_labels.append(true_label)
            
            # Store detailed results
            results.append({
                'id': sample_id,
                'true_label': true_label,
                'predicted_label': prediction,
                'transcription': transcription,
                'audio_path': audio_path
            })
            
            # Print progress for first few samples
            if len(results) <= 5:
                print(f"Sample {len(results)}: True={true_label}, Predicted={prediction}")
        
        return predictions, true_labels, results
    
    def calculate_detailed_metrics(self, true_labels, predictions):
        """Calculate detailed metrics for each class and weighted averages"""
        # Further filter to ensure only expected class names are included
        valid_indices = []
        for i in range(len(true_labels)):
            if (true_labels[i] in self.class_names and 
                predictions[i] in self.class_names):
                valid_indices.append(i)
        
        # Filter to only valid class names
        filtered_true_labels = [true_labels[i] for i in valid_indices]
        filtered_predictions = [predictions[i] for i in valid_indices]
        
        print(f"Debug - Filtered to {len(filtered_true_labels)} samples with valid class names")
        print(f"Debug - Filtered unique true labels: {set(filtered_true_labels)}")
        print(f"Debug - Filtered unique predictions: {set(filtered_predictions)}")
        
        if len(filtered_true_labels) == 0:
            print("Warning: No samples with valid class names found!")
            # Return zero metrics
            metrics = {}
            for class_name in self.class_names:
                metrics[f'f1_{class_name}'] = 0.0
                metrics[f'precision_{class_name}'] = 0.0
                metrics[f'recall_{class_name}'] = 0.0
            
            metrics['f1_weighted'] = 0.0
            metrics['precision_weighted'] = 0.0
            metrics['recall_weighted'] = 0.0
            metrics['f1_macro'] = 0.0
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0
            metrics['accuracy'] = 0.0
            return metrics
        
        metrics = {}
        
        # Calculate per-class metrics using filtered data
        f1_per_class = f1_score(filtered_true_labels, filtered_predictions, labels=self.class_names, average=None, zero_division=0)
        precision_per_class = precision_score(filtered_true_labels, filtered_predictions, labels=self.class_names, average=None, zero_division=0)
        recall_per_class = recall_score(filtered_true_labels, filtered_predictions, labels=self.class_names, average=None, zero_division=0)
        
        # Calculate weighted averages
        f1_weighted = f1_score(filtered_true_labels, filtered_predictions, average='weighted', zero_division=0)
        precision_weighted = precision_score(filtered_true_labels, filtered_predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(filtered_true_labels, filtered_predictions, average='weighted', zero_division=0)
        
        # Calculate macro averages
        f1_macro = f1_score(filtered_true_labels, filtered_predictions, average='macro', zero_division=0)
        precision_macro = precision_score(filtered_true_labels, filtered_predictions, average='macro', zero_division=0)
        recall_macro = recall_score(filtered_true_labels, filtered_predictions, average='macro', zero_division=0)
        
        # Store per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
        
        # Store weighted and macro averages
        metrics['f1_weighted'] = f1_weighted
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_macro'] = f1_macro
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        
        # Calculate accuracy
        metrics['accuracy'] = accuracy_score(filtered_true_labels, filtered_predictions)
        
        return metrics
    
    def evaluate_results(self, predictions, true_labels, results, output_path=None):
        """Evaluate and print results, save to Excel"""
        # Debug: Check unique values in predictions and true labels
        unique_predictions = set(predictions)
        unique_true_labels = set(true_labels)
        print(f"Debug - Unique predictions: {unique_predictions}")
        print(f"Debug - Unique true labels: {unique_true_labels}")
        print(f"Debug - Expected class names: {self.class_names}")
        
        # Filter out unknown predictions for accuracy calculation
        valid_indices = [i for i, pred in enumerate(predictions) if pred != 'unknown']
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_labels = [true_labels[i] for i in valid_indices]
        
        # Debug: Check if valid labels are all in expected class names
        valid_unique_predictions = set(valid_predictions)
        valid_unique_true_labels = set(valid_true_labels)
        print(f"Debug - Valid unique predictions: {valid_unique_predictions}")
        print(f"Debug - Valid unique true labels: {valid_unique_true_labels}")
        
        # Check if all valid labels are in expected class names
        unexpected_pred = valid_unique_predictions - set(self.class_names)
        unexpected_true = valid_unique_true_labels - set(self.class_names)
        if unexpected_pred:
            print(f"Warning: Unexpected prediction values not in class_names: {unexpected_pred}")
        if unexpected_true:
            print(f"Warning: Unexpected true label values not in class_names: {unexpected_true}")
        
        # Count predictions by type
        unknown_count = len(predictions) - len(valid_predictions)
        valid_rate = len(valid_predictions) / len(predictions) * 100 if len(predictions) > 0 else 0
        
        print(f"\n=== EVALUATION RESULTS (2-Label: Dementia vs Control) ===")
        print(f"Total samples: {len(predictions)}")
        print(f"Valid predictions: {len(valid_predictions)} ({valid_rate:.1f}%)")
        print(f"Unknown predictions: {unknown_count} ({100-valid_rate:.1f}%)")
        
        if unknown_count > 0:
            print(f"\nNote: All performance metrics below are calculated ONLY on the {len(valid_predictions)} valid predictions.")
            print("Unknown predictions are excluded from all metric calculations.")
        
        if len(valid_predictions) == 0:
            print("No valid predictions to evaluate!")
            return 0.0
        
        # Calculate detailed metrics (only on valid predictions)
        metrics = self.calculate_detailed_metrics(valid_true_labels, valid_predictions)
        
        # Apply same filtering logic as in calculate_detailed_metrics for consistency
        final_valid_indices = []
        for i in range(len(valid_true_labels)):
            if (valid_true_labels[i] in self.class_names and 
                valid_predictions[i] in self.class_names):
                final_valid_indices.append(i)
        
        final_true_labels = [valid_true_labels[i] for i in final_valid_indices]
        final_predictions = [valid_predictions[i] for i in final_valid_indices]
        
        print(f"Debug - Final filtered count: {len(final_true_labels)} for reporting")
        
        # Print overall metrics
        print(f"\n=== OVERALL METRICS (Valid Predictions Only) ===")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        
        # Print per-class metrics
        print(f"\n=== PER-CLASS METRICS (Valid Predictions Only) ===")
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)
        for i, class_name in enumerate(self.class_names):
            display_name = self.class_display_names[i]
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            print(f"{display_name:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        # Print macro averages
        print(f"\n=== MACRO AVERAGES (Valid Predictions Only) ===")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print(f"Macro Precision: {metrics['precision_macro']:.4f}")
        print(f"Macro Recall: {metrics['recall_macro']:.4f}")
        
        # Print classification report (using final filtered data)
        if len(final_true_labels) > 0:
            print(f"\n=== DETAILED CLASSIFICATION REPORT (Valid Predictions Only) ===")
            print(classification_report(final_true_labels, final_predictions, 
                                      target_names=self.class_display_names, zero_division=0))
            
            # Print confusion matrix (using final filtered data)
            print(f"\n=== CONFUSION MATRIX (Valid Predictions Only) ===")
            cm = confusion_matrix(final_true_labels, final_predictions, labels=self.class_names)
            print("Predicted:  Dementia  Control")
            for i, label in enumerate(self.class_display_names):
                print(f"True {label}:     {cm[i][0]:3d}      {cm[i][1]:3d}")
        else:
            print(f"\n=== CLASSIFICATION REPORT AND CONFUSION MATRIX ===")
            print("No valid samples for detailed reporting.")
            cm = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
        
        # Save results to files (use final filtered data for metrics)
        self.save_results(results, metrics, cm, final_true_labels, final_predictions, output_path)
        
        return metrics['accuracy']
    
    def save_results(self, results, metrics, confusion_matrix, true_labels, predictions, output_path=None):
        """Save results to CSV and Excel files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"test_results_2label_{timestamp}"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Add validity flag to results
        for i, result in enumerate(results):
            result['is_valid_prediction'] = result['predicted_label'] != 'unknown'
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_path, 'detailed_results.csv')
        excel_path = os.path.join(output_path, 'test_results_2label.xlsx')
        
        results_df.to_csv(csv_path, index=False)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Detailed results
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Sheet 2: Valid predictions only (filtered results)
            valid_results_df = results_df[results_df['is_valid_prediction'] == True]
            valid_results_df.to_excel(writer, sheet_name='Valid_Predictions_Only', index=False)
            
            # Sheet 3: Summary metrics
            metrics_data = []
            
            # Add sample counts
            total_samples = len(results)
            valid_samples = len(valid_results_df)
            unknown_samples = total_samples - valid_samples
            
            metrics_data.append(['Total Samples', total_samples])
            metrics_data.append(['Valid Predictions', valid_samples])
            metrics_data.append(['Unknown Predictions', unknown_samples])
            metrics_data.append(['Valid Prediction Rate', valid_samples / total_samples if total_samples > 0 else 0])
            metrics_data.append(['', ''])  # Empty row
            
            # Overall metrics (calculated on valid predictions only)
            metrics_data.append(['Overall Accuracy', metrics['accuracy']])
            metrics_data.append(['Weighted F1-Score', metrics['f1_weighted']])
            metrics_data.append(['Weighted Precision', metrics['precision_weighted']])
            metrics_data.append(['Weighted Recall', metrics['recall_weighted']])
            metrics_data.append(['Macro F1-Score', metrics['f1_macro']])
            metrics_data.append(['Macro Precision', metrics['precision_macro']])
            metrics_data.append(['Macro Recall', metrics['recall_macro']])
            metrics_data.append(['', ''])  # Empty row
            
            # Per-class metrics
            for i, class_name in enumerate(self.class_names):
                display_name = self.class_display_names[i]
                metrics_data.append([f'{display_name} Precision', metrics[f'precision_{class_name}']])
                metrics_data.append([f'{display_name} Recall', metrics[f'recall_{class_name}']])
                metrics_data.append([f'{display_name} F1-Score', metrics[f'f1_{class_name}']])
                metrics_data.append(['', ''])  # Empty row
            
            metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
            metrics_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
            
            # Sheet 4: Confusion Matrix
            cm_df = pd.DataFrame(confusion_matrix, 
                               index=[f'True_{name}' for name in self.class_display_names],
                               columns=[f'Pred_{name}' for name in self.class_display_names])
            cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
            
            # Sheet 5: Per-class breakdown
            class_breakdown = []
            for i, class_name in enumerate(self.class_names):
                display_name = self.class_display_names[i]
                class_breakdown.append({
                    'Class': display_name,
                    'Precision': metrics[f'precision_{class_name}'],
                    'Recall': metrics[f'recall_{class_name}'],
                    'F1-Score': metrics[f'f1_{class_name}'],
                    'Support': sum(1 for label in true_labels if label == class_name)
                })
            
            class_df = pd.DataFrame(class_breakdown)
            class_df.to_excel(writer, sheet_name='Per_Class_Metrics', index=False)
        
        print(f"\nResults saved to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - Excel: {excel_path}")
        print(f"  - Valid prediction rate: {valid_samples}/{total_samples} ({valid_samples/total_samples*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Test 2-label audio classification model (Dementia vs Control)')
    parser.add_argument('--model_path', type=str, default='/workspace/qwen_seed0',
                       help='Path to the fine-tuned model')
    parser.add_argument('--test_csv', type=str, default='/workspace/phi4/csv_files/test.csv',
                       help='Path to test CSV file')
    parser.add_argument('--audio_path', type=str, 
                       default='data/audio_files',
                       help='Base path to audio files')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to test (default: all)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for testing')
    parser.add_argument('--output_path', type=str, default="/workspace/qwen_results/",
                       help='Output directory path for results (default: auto-generated with timestamp)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = AudioClassificationTester2Label(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        audio_base_path=args.audio_path
    )
    
    # Run tests
    predictions, true_labels, results = tester.test_subset(
        num_samples=args.num_samples,
        start_idx=args.start_idx
    )
    
    # Evaluate results
    accuracy = tester.evaluate_results(predictions, true_labels, results, args.output_path)
    
    print(f"\n=== TESTING COMPLETED ===")
    print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    main() 