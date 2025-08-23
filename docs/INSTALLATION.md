# NeuralScript Installation Guide

This guide walks you through installing and setting up NeuralScript on your system.

## System Requirements

### Operating Systems
- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+, or similar
- **macOS**: macOS 10.14 (Mojave) or later
- **Windows**: Windows 10 or Windows 11

### Hardware Requirements
- **CPU**: x86-64 architecture (Intel/AMD 64-bit)
- **Memory**: At least 2GB RAM (4GB+ recommended for large projects)
- **Storage**: 500MB for basic installation, 2GB+ for full development setup

### Software Dependencies
- **Python**: 3.8 or later (3.9+ recommended)
- **Git**: For version control and repository cloning
- **C/C++ Compiler**: GCC 7+, Clang 9+, or MSVC 2019+ (for LLVM backend)

## Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install neuralscript

# Verify installation
neuralscript --version
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/xwest/neuralscript.git
cd neuralscript

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "import neuralscript; print('NeuralScript installed successfully!')"
```

## Detailed Installation Instructions

### Linux/macOS Installation

1. **Install Python 3.8+**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.9 python3.9-venv python3.9-dev
   
   # macOS with Homebrew
   brew install python@3.9
   
   # CentOS/RHEL
   sudo dnf install python39 python39-devel
   ```

2. **Install build tools**:
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential git cmake
   
   # macOS
   xcode-select --install
   brew install cmake
   
   # CentOS/RHEL
   sudo dnf groupinstall "Development Tools"
   sudo dnf install cmake
   ```

3. **Clone and install NeuralScript**:
   ```bash
   git clone https://github.com/xwest/neuralscript.git
   cd neuralscript
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

### Windows Installation

1. **Install Python**:
   - Download Python 3.9+ from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
   - Verify: Open Command Prompt and run `python --version`

2. **Install Git**:
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default installation settings

3. **Install Visual Studio Build Tools** (for LLVM backend):
   - Download "Build Tools for Visual Studio 2022" from Microsoft
   - Install with C++ build tools workload

4. **Install NeuralScript**:
   ```cmd
   git clone https://github.com/xwest/neuralscript.git
   cd neuralscript
   python -m venv venv
   venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Optional Components

### LLVM Backend (Recommended)

For full compilation to native machine code:

```bash
# Install LLVM Python bindings
pip install llvmlite

# Verify LLVM installation
python -c "import llvmlite; print('LLVM version:', llvmlite.binding.llvm_version_info)"
```

If `llvmlite` installation fails:

**Linux**:
```bash
# Install LLVM development packages
sudo apt install llvm-12-dev  # Ubuntu/Debian
sudo dnf install llvm-devel    # CentOS/RHEL
```

**macOS**:
```bash
brew install llvm
export LLVM_CONFIG=/usr/local/opt/llvm/bin/llvm-config
pip install llvmlite
```

**Windows**:
- Download pre-built LLVM from [llvm.org](https://llvm.org/builds/)
- Add LLVM bin directory to PATH
- Install: `pip install llvmlite`

### GPU Support (Advanced)

For GPU acceleration with CUDA:

```bash
# Install CUDA toolkit (NVIDIA GPUs)
# Follow instructions at https://developer.nvidia.com/cuda-downloads

# Install Python CUDA bindings
pip install cupy-cuda11x  # Adjust version for your CUDA

# For OpenCL support
pip install pyopencl
```

### Development Tools

For contributing to NeuralScript:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install testing tools
pip install pytest pytest-cov black mypy
```

## Verification

### Basic Installation Test

```bash
# Test basic functionality
python -c "
import neuralscript
from neuralscript.lexer import Lexer
from neuralscript.parser import Parser

# Test tokenization
lexer = Lexer('let x = 42', '<test>')
tokens = lexer.tokenize()
print(f'✓ Lexer working: {len(tokens)} tokens generated')

# Test parsing
parser = Parser(tokens)
ast = parser.parse()
print('✓ Parser working: AST generated successfully')
print('✓ NeuralScript installation verified!')
"
```

### Interactive REPL Test

```bash
# Start the interactive REPL
python repl.py

# You should see:
# 🧠 NeuralScript Interactive REPL v0.1.0
# ==================================================
# Features:
#   • Live compilation and execution
#   • Unicode math operators (×, ÷, ², ≤, ≥)
#   • Complex numbers (3+4i)
#   • Unit literals (100.0_m, 5_kg)
#   • Type inference and checking
```

Try some basic expressions:
```
ns:1> let x = 42
ns:2> let y = 3.14
ns:3> let complex_num = 3.0+4.0i
ns:4> let distance = 100.0_m
```

### Compilation Test

```bash
# Create a test file
echo 'fn main() -> i32 { 
    let x = 42
    let y = 3.14
    println!("Hello from NeuralScript!")
    0
}' > test.ns

# Compile it
python -m neuralscript compile test.ns

# Should output compilation success message
```

## Troubleshooting

### Common Issues

**Problem**: `python` command not found
```bash
# Solution: Use python3 explicitly
python3 -m venv venv
```

**Problem**: Permission denied during installation
```bash
# Solution: Install in user space
pip install --user -r requirements.txt
```

**Problem**: LLVM installation fails
```bash
# Solution: Use mock backend temporarily
export NEURALSCRIPT_USE_MOCK_LLVM=1
pip install -r requirements.txt
```

**Problem**: Unicode characters not displaying
- **Windows**: Use Windows Terminal or PowerShell 7+
- **Linux/macOS**: Ensure UTF-8 locale: `export LC_ALL=en_US.UTF-8`

### Getting Help

1. **Check the FAQ**: `docs/FAQ.md`
2. **Search existing issues**: [GitHub Issues](https://github.com/xwest/neuralscript/issues)
3. **Create a new issue** with:
   - Operating system and version
   - Python version (`python --version`)
   - Full error message and stack trace
   - Steps to reproduce the problem

## IDE Integration

### VS Code Setup

1. **Install the NeuralScript extension** (when available):
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "NeuralScript"
   - Install the extension

2. **Manual setup**:
   ```json
   // Add to settings.json
   {
     "files.associations": {
       "*.ns": "neuralscript"
     },
     "editor.unicodeHighlight.ambiguousCharacters": false
   }
   ```

### Other Editors

**Vim/Neovim**:
```vim
" Add to .vimrc
au BufNewFile,BufRead *.ns set filetype=neuralscript
syntax on
```

**Emacs**:
```lisp
;; Add to .emacs
(add-to-list 'auto-mode-alist '("\\.ns\\'" . neuralscript-mode))
```

## Performance Optimization

### Compilation Performance

```bash
# Use parallel compilation
export NEURALSCRIPT_PARALLEL=4

# Enable optimization
export NEURALSCRIPT_OPTIMIZE=1
```

### Memory Usage

```bash
# For large projects, increase memory limits
export NEURALSCRIPT_MAX_MEMORY=8G
```

## Next Steps

1. **Read the Language Guide**: `docs/LANGUAGE_GUIDE.md`
2. **Try the Examples**: Explore the `examples/` directory
3. **Run the Test Suite**: `python run_tests.py`
4. **Join the Community**: Check out GitHub Discussions
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Uninstallation

To remove NeuralScript:

```bash
# If installed from source
pip uninstall neuralscript

# Remove virtual environment
rm -rf venv

# Remove cloned repository
cd .. && rm -rf neuralscript
```

---

Welcome to NeuralScript! If you encounter any issues during installation, please don't hesitate to ask for help in our GitHub Discussions or file an issue.
