#!/usr/bin/env python
"""
Test runner for the Quantum System Solver.
Runs all test suites and generates a coverage report.
"""

import sys
import os
import pytest

def run_tests(verbose=True, coverage=False):
    """
    Run all tests with optional coverage reporting.
    
    Args:
        verbose: If True, show detailed test output
        coverage: If True, generate coverage report
    """
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Base pytest arguments
    args = [
        test_dir,  # Test directory
        "-v" if verbose else "",  # Verbose output
        "--tb=short",  # Short traceback format
    ]
    
    if coverage:
        # Add coverage arguments
        args.extend([
            "--cov=src",  # Coverage for src directory
            "--cov-report=html",  # HTML coverage report
            "--cov-report=term-missing",  # Terminal report with missing lines
            "--cov-branch",  # Branch coverage
        ])
    
    # Filter out empty strings
    args = [arg for arg in args if arg]
    
    # Run pytest
    return pytest.main(args)

def run_specific_test(test_file, test_name=None, verbose=True):
    """
    Run a specific test file or test.
    
    Args:
        test_file: Name of the test file
        test_name: Optional specific test name
        verbose: If True, show detailed output
    """
    args = [test_file]
    
    if test_name:
        args.extend(["-k", test_name])
        
    if verbose:
        args.append("-v")
        
    args.extend(["--tb=short"])
    
    return pytest.main(args)

def main():
    """Main function to run tests based on command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Quantum System Solver")
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Minimal output"
    )
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file"
    )
    parser.add_argument(
        "--test", "-t",
        help="Run specific test (use with --file)"
    )
    parser.add_argument(
        "--markers", "-m",
        action="store_true",
        help="Show available test markers"
    )
    
    args = parser.parse_args()
    
    if args.markers:
        print("Available test markers:")
        print("  - slow: Marks tests as slow running")
        print("  - integration: Integration tests")
        print("  - unit: Unit tests")
        return 0
    
    if args.file:
        # Run specific test file
        return run_specific_test(
            args.file,
            args.test,
            verbose=not args.quiet
        )
    else:
        # Run all tests
        return run_tests(
            verbose=not args.quiet,
            coverage=args.coverage
        )

if __name__ == "__main__":
    sys.exit(main())
