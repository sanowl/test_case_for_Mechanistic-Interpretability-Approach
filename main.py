from analyzer import SimpleMechanisticAnalyzer
import logging
import json
import os
from datetime import datetime
import argparse
from typing import List, Dict

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log'),
            logging.StreamHandler()
        ]
    )

def load_test_cases(file_path: str = None) -> List[Dict]:
    """Load test cases from file or return default cases."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    return [
        {"text": "The capital of France is", "target": "Paris"},
        {"text": "Water boils at", "target": "100"},
        {"text": "The speed of light is approximately", "target": "300000"},
        {"text": "The chemical symbol for gold is", "target": "Au"},
        {"text": "The largest planet in our solar system is", "target": "Jupiter"}
    ]

def save_results(results: Dict, case: Dict, output_dir: str):
    """Save analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}_{case['target']}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Saved results to {filepath}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run mechanistic analysis on language model")
    parser.add_argument("--model", default="gpt2", help="Model to analyze")
    parser.add_argument("--test-cases", help="JSON file containing test cases")
    parser.add_argument("--output-dir", default="analysis_results", help="Directory for saving results")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    args = parser.parse_args()

    # Set up logging and output directory
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Starting mechanistic analysis demo...")
    
    try:
        # Initialize analyzer
        analyzer = SimpleMechanisticAnalyzer(
            model_name=args.model,
            output_dir=args.output_dir
        )
        
        # Load test cases
        test_cases = load_test_cases(args.test_cases)
        
        # Analyze each test case
        for case in test_cases:
            try:
                logging.info(f"\nAnalyzing: {case['text']} -> {case['target']}")
                
                results = analyzer.analyze_single_fact(case["text"], case["target"])
                
                if results:
                    analyzer.visualize_results(results, save=args.save_plots)
                    save_results(results, case, args.output_dir)
                else:
                    logging.error(f"Analysis failed for case: {case}")
                
                print("\n" + "="*50 + "\n")
                
            except Exception as e:
                logging.error(f"Error processing case {case}: {e}")
                continue
        
        logging.info("Analysis complete!")
        
    except Exception as e:
        logging.error(f"Fatal error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()