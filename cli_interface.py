import os
import sys
import argparse
from typing import Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predictor import FakeNewsPredictor, find_latest_model


class FakeNewsDetectionCLI:
    """
    Command Line Interface for fake news detection.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.predictor = FakeNewsPredictor()
        self.is_model_loaded = False
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model.
        
        Args:
            model_path (Optional[str]): Path to model file, or None to auto-find latest
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if model_path is None:
                # Try to find the latest model
                model_path = find_latest_model()
                if model_path is None:
                    print("❌ No trained models found!")
                    print("Please run 'python main.py' to train a model first.")
                    return False
                print(f"🔍 Auto-detected latest model: {os.path.basename(model_path)}")
            
            # Load the model
            self.predictor.load_model(model_path)
            self.is_model_loaded = True
            
            # Display model info
            model_info = self.predictor.get_model_info()
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model: {model_info['model_name']}")
            print(f"📈 F1-Score: {model_info['performance_metrics'].get('f1_score', 'N/A'):.4f}")
            print(f"🎯 Accuracy: {model_info['performance_metrics'].get('accuracy', 'N/A'):.4f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_single_text(self, text: str) -> None:
        """
        Predict and display result for a single text.
        
        Args:
            text (str): Text to classify
        """
        if not self.is_model_loaded:
            print("❌ No model loaded. Please load a model first.")
            return
        
        try:
            # Get prediction
            if hasattr(self.predictor.model, 'predict_proba'):
                prediction, confidence = self.predictor.predict_with_confidence(text)
                
                # Display result with confidence
                if prediction == 'FAKE':
                    emoji = "🚨"
                    color = "red"
                else:
                    emoji = "✅"
                    color = "green"
                
                print(f"\n{emoji} PREDICTION: {prediction}")
                print(f"🎯 CONFIDENCE: {confidence:.3f} ({confidence*100:.1f}%)")
                
                # Add interpretation
                if confidence > 0.8:
                    certainty = "Very High"
                elif confidence > 0.6:
                    certainty = "High"
                elif confidence > 0.5:
                    certainty = "Moderate"
                else:
                    certainty = "Low"
                
                print(f"📊 CERTAINTY: {certainty}")
                
            else:
                prediction = self.predictor.predict_news(text)
                emoji = "🚨" if prediction == 'FAKE' else "✅"
                print(f"\n{emoji} PREDICTION: {prediction}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
    
    def interactive_mode(self) -> None:
        """
        Run interactive mode for continuous text input.
        """
        print("🔄 FACTFORGE INTERACTIVE MODE")
        print("=" * 50)
        print("Enter news text to forge the truth (or 'quit' to exit)")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help message")
        print("  'info' - Show model information")
        print("  'clear' - Clear the screen")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                print("\n📝 Enter news text:")
                user_input = input("> ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📚 HELP:")
                    print("  Enter any news text to get a FAKE/REAL prediction")
                    print("  Commands: quit, exit, help, info, clear")
                    continue
                
                elif user_input.lower() == 'info':
                    if self.is_model_loaded:
                        model_info = self.predictor.get_model_info()
                        print(f"\n📊 MODEL INFORMATION:")
                        print(f"  Model: {model_info['model_name']}")
                        print(f"  Type: {model_info['model_type']}")
                        print(f"  Saved: {model_info['save_date']}")
                        print(f"  Performance: {model_info['performance_metrics']}")
                    else:
                        print("❌ No model loaded")
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                elif not user_input:
                    print("⚠️  Please enter some text to classify")
                    continue
                
                # Make prediction
                self.predict_single_text(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def batch_mode(self, input_file: str, output_file: Optional[str] = None) -> None:
        """
        Process a batch of texts from a file.
        
        Args:
            input_file (str): Path to input file (one text per line)
            output_file (Optional[str]): Path to output file for results
        """
        if not self.is_model_loaded:
            print("❌ No model loaded. Please load a model first.")
            return
        
        try:
            # Read input file
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"📁 Processing {len(texts)} texts from {input_file}")
            
            # Make predictions
            predictions = self.predictor.batch_predict(texts)
            
            # Prepare results
            results = []
            for i, (text, prediction) in enumerate(zip(texts, predictions), 1):
                results.append(f"{i}. {text[:50]}... -> {prediction}")
            
            # Display results
            print("\n📊 BATCH RESULTS:")
            print("=" * 50)
            for result in results:
                print(result)
            
            # Save to output file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("Text,Prediction\n")
                    for text, prediction in zip(texts, predictions):
                        f.write(f'"{text}",{prediction}\n')
                print(f"\n💾 Results saved to {output_file}")
            
            # Show summary
            fake_count = predictions.count('FAKE')
            real_count = predictions.count('REAL')
            print(f"\n📈 SUMMARY:")
            print(f"  Total texts: {len(texts)}")
            print(f"  FAKE: {fake_count} ({fake_count/len(texts)*100:.1f}%)")
            print(f"  REAL: {real_count} ({real_count/len(texts)*100:.1f}%)")
            
        except FileNotFoundError:
            print(f"❌ Input file not found: {input_file}")
        except Exception as e:
            print(f"❌ Error processing batch: {str(e)}")


def main():
    """
    Main function for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="FactForge - AI News Authenticity Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_interface.py                           # Interactive mode
  python cli_interface.py --text "Breaking news!"  # Single prediction
  python cli_interface.py --batch input.txt        # Batch processing
  python cli_interface.py --model path/to/model.joblib  # Use specific model
        """
    )
    
    parser.add_argument('--model', '-m', type=str, help='Path to model file')
    parser.add_argument('--text', '-t', type=str, help='Text to classify')
    parser.add_argument('--batch', '-b', type=str, help='Input file for batch processing')
    parser.add_argument('--output', '-o', type=str, help='Output file for batch results')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = FakeNewsDetectionCLI()
    
    # Load model
    if not cli.load_model(args.model):
        return
    
    # Handle different modes
    if args.text:
        # Single text prediction
        print(f"📝 Input text: {args.text}")
        cli.predict_single_text(args.text)
        
    elif args.batch:
        # Batch processing
        cli.batch_mode(args.batch, args.output)
        
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
