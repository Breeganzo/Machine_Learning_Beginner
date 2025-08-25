"""
Machine Learning Environment Test Script
Run this to verify all packages are working correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("ðŸ” Testing package imports...\n")
    
    packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('sklearn', None),
        ('scipy', None),
        ('jupyter', None),
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package, alias in packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"âœ… {package:<20} - OK")
            success_count += 1
        except ImportError as e:
            print(f"âŒ {package:<20} - FAILED: {e}")
    
    print(f"\nðŸ“Š Results: {success_count}/{total_packages} packages imported successfully")
    
    if success_count == total_packages:
        print("ðŸŽ‰ All packages are working correctly!")
        return True
    else:
        print("âš ï¸ Some packages failed to import. Please check installation.")
        return False

def test_basic_functionality():
    """Test basic functionality of key packages"""
    print("\nðŸ§ª Testing basic functionality...\n")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Test NumPy
        arr = np.array([1, 2, 3, 4, 5])
        print(f"âœ… NumPy array creation: {arr}")
        
        # Test Pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"âœ… Pandas DataFrame creation:\n{df}")
        
        # Test Matplotlib (without displaying)
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot")
        plt.close()  # Close without showing
        print("âœ… Matplotlib plot creation: OK")
        
        # Test Scikit-learn
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        print("âœ… Scikit-learn model creation: OK")
        
        print("\nðŸŽ‰ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"âŒ Functionality test failed: {e}")
        return False

def show_versions():
    """Show versions of installed packages"""
    print("\nðŸ“‹ Package Versions:")
    print("-" * 40)
    
    packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy']
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"{package:<15}: {version}")
        except ImportError:
            print(f"{package:<15}: Not installed")

def main():
    """Main test function"""
    print("=" * 50)
    print("   MACHINE LEARNING ENVIRONMENT TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()
        
        # Show versions
        show_versions()
        
        if functionality_ok:
            print("\nðŸŒŸ Your machine learning environment is ready!")
            print("\nYou can now open and run your Jupyter notebooks:")
            print("  â€¢ Simple Linear Regression")
            print("  â€¢ Multiple Linear Regression") 
            print("  â€¢ Logistic Regression")
            print("  â€¢ Polynomial Regression")
            print("  â€¢ Ridge & Lasso Regression")
            
            print("\nTo start Jupyter Lab, run:")
            print("  jupyter lab")
        else:
            print("\nâš ï¸ Environment setup incomplete. Please check errors above.")
    else:
        print("\nâŒ Package installation incomplete. Please run setup script again.")

if __name__ == "__main__":
    main()
