#!/usr/bin/env python
"""
Verification script to check that tests can be properly imported and run.
"""

import sys
import os
import importlib
import traceback

def verify_imports():
    """Verify that all test modules can be imported."""
    print("=" * 60)
    print("VERIFYING TEST IMPORTS")
    print("=" * 60)
    
    # Add source path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_path = os.path.join(parent_dir, 'inputs', 'Quantum-System-Solver', 'src')
    
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"✓ Added source path: {src_path}")
    else:
        print(f"⚠ Source path not found: {src_path}")
        print("  Tests will need the source code to be available")
    
    print()
    
    # Test modules to verify
    test_modules = [
        'test_linear_operator_system',
        'test_momentum_coordinate_change',
        'test_integration',
        'test_examples'
    ]
    
    results = []
    
    for module_name in test_modules:
        try:
            module = importlib.import_module(module_name)
            classes = [m for m in dir(module) if m.startswith('Test')]
            print(f"✓ {module_name}: Successfully imported ({len(classes)} test classes)")
            results.append((module_name, True, len(classes)))
        except ImportError as e:
            print(f"✗ {module_name}: Import failed - {e}")
            results.append((module_name, False, 0))
        except Exception as e:
            print(f"✗ {module_name}: Unexpected error - {e}")
            traceback.print_exc()
            results.append((module_name, False, 0))
    
    return results

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60 + "\n")
    
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'pytest': 'pytest',
        'sympy': 'SymPy'
    }
    
    available = []
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            available.append(name)
            print(f"✓ {name}: Available")
        except ImportError:
            missing.append(name)
            print(f"✗ {name}: Not installed")
    
    return available, missing

def count_tests():
    """Count the total number of tests."""
    print("\n" + "=" * 60)
    print("TEST STATISTICS")
    print("=" * 60 + "\n")
    
    try:
        import ast
        
        test_files = [
            'test_linear_operator_system.py',
            'test_momentum_coordinate_change.py',
            'test_integration.py',
            'test_examples.py'
        ]
        
        total_tests = 0
        total_classes = 0
        
        for filename in test_files:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    tree = ast.parse(f.read())
                
                classes = [node for node in ast.walk(tree) 
                          if isinstance(node, ast.ClassDef) and node.name.startswith('Test')]
                test_methods = 0
                
                for cls in classes:
                    methods = [node for node in cls.body 
                              if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
                    test_methods += len(methods)
                
                print(f"{filename}:")
                print(f"  Classes: {len(classes)}")
                print(f"  Tests: {test_methods}")
                
                total_classes += len(classes)
                total_tests += test_methods
            else:
                print(f"{filename}: Not found")
        
        print(f"\nTOTAL: {total_classes} test classes, {total_tests} test methods")
        
    except Exception as e:
        print(f"Could not count tests: {e}")

def main():
    """Main verification routine."""
    print("QUANTUM SYSTEM SOLVER - TEST VERIFICATION")
    print()
    
    # Check dependencies
    available, missing = check_dependencies()
    
    if missing:
        print(f"\n⚠ Warning: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r test_requirements.txt")
    
    # Verify imports
    results = verify_imports()
    
    # Count tests
    count_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60 + "\n")
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if successful == total:
        print(f"✓ All {total} test modules can be imported successfully!")
        print("\nTo run tests:")
        print("  python run_tests.py           # Run all tests")
        print("  python run_tests.py --coverage # Run with coverage")
        print("  pytest -v                      # Using pytest directly")
        return 0
    else:
        print(f"⚠ {successful}/{total} test modules imported successfully")
        print("\nPlease ensure the source code is available in the correct location")
        return 1

if __name__ == "__main__":
    sys.exit(main())
