#!/usr/bin/env python3
"""
Test runner script for BERTopic Desktop Application.

This script provides convenient ways to run different types of tests:
- All tests
- Unit tests only
- Integration tests only
- UI tests only (requires display)
- Performance benchmarks
- Coverage reports

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --ui               # Run UI tests only
    python run_tests.py --benchmark        # Run performance benchmarks
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --verbose          # Verbose output
"""
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for BERTopic Desktop Application")
    
    # Test type options
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--ui", action="store_true", help="Run UI tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    # Test modifiers
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    # Output options
    parser.add_argument("--html-coverage", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture output (useful for debugging)")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Determine what tests to run
    test_paths = []
    if args.unit:
        test_paths.append("tests/unit")
    if args.integration:
        test_paths.append("tests/integration")
    if args.ui:
        test_paths.append("tests/ui")
    if args.benchmark:
        test_paths.append("tests/performance")
    
    # If no specific test type selected, run all except benchmarks by default
    if not any([args.unit, args.integration, args.ui, args.benchmark]):
        test_paths = ["tests/unit", "tests/integration", "tests/ui"]
    
    cmd.extend(test_paths)
    
    # Add pytest options
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Skip slow tests if requested
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Coverage options
    if args.coverage or args.html_coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        if args.html_coverage:
            cmd.extend(["--cov-report=html"])
    
    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Output capture
    if args.no_capture:
        cmd.append("-s")
    
    # Additional pytest options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Error on unknown markers
    ])
    
    # Run the tests
    description = f"Running tests: {', '.join(test_paths)}"
    success = run_command(cmd, description)
    
    # Additional actions based on results
    if success:
        print("\n🎉 All tests passed!")
        
        if args.html_coverage:
            print("\n📊 HTML coverage report generated in: htmlcov/index.html")
            print("Open it in your browser to view detailed coverage information.")
        
        # Show quick stats
        if args.benchmark:
            print("\n📈 Performance benchmark results are shown above.")
            print("Use these as baseline measurements for performance regression testing.")
    
    else:
        print("\n💥 Some tests failed!")
        print("\nTroubleshooting tips:")
        print("- Check the error messages above")
        print("- Run with --verbose for more detailed output")
        print("- Run specific test files: pytest tests/unit/test_specific.py -v")
        print("- Use --no-capture to see print statements: python run_tests.py --no-capture")
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())