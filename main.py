# main.py

from analyzer import SimpleMechanisticAnalyzer


def main():
    print("Starting mechanistic analysis demo...")
    
    # Initialize analyzer
    analyzer = SimpleMechanisticAnalyzer()
    
    # Test cases
    test_cases = [
        {"text": "The capital of France is", "target": "Paris"},
        {"text": "Water boils at", "target": "100"},
        {"text": "The speed of light is approximately", "target": "300000"},
        {"text": "The chemical symbol for gold is", "target": "Au"},
        {"text": "The largest planet in our solar system is", "target": "Jupiter"}
    ]
    
    # Analyze each test case
    for case in test_cases:
        print(f"\nAnalyzing: {case['text']} -> {case['target']}")
        results = analyzer.analyze_single_fact(case["text"], case["target"])
        analyzer.visualize_results(results)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
