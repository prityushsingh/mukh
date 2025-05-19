# Documentation Guide

This directory contains the documentation for the mukh project using Sphinx. Follow these steps to update and build the documentation.

## Prerequisites

Make sure you have the required dependencies installed:
  
```bash
pip install -e ".[dev]" --use-pep517  
```
  
## Steps to Update Documentation
  
1. **Update Source Files**
   - Documentation source files are in reStructuredText format (`.rst`)
   - Main configuration is in `conf.py`
   - Main index file is `index.rst`
  
2. **Auto-generate API Documentation**
   ```bash
   # From the docs directory
   sphinx-apidoc -o . ../mukh -f
   ```
   This will update the module documentation based on docstrings in the code.
  
3. **Build HTML Documentation**
   ```bash
   # If you're on Unix/Linux/MacOS
   make html

   # If you're on Windows
   make.bat html
   ```
  
4. **View Documentation**
   - Open `_build/html/index.html` in your web browser
   - Check that your changes appear correctly
  
## Common Tasks
  
### Adding a New Module
1. Add docstrings to your Python code
2. Run `sphinx-apidoc` command (step 2 above)
3. Build documentation (step 3 above)
  
### Updating Existing Documentation
1. Make changes to `.rst` files or Python docstrings
2. Build documentation (step 3 above)
3. View changes in browser
  
### Clean Build
To do a clean rebuild:
  
```bash
# Unix/Linux/MacOS
make clean
make html

# Windows
make.bat clean
make.bat html
```
  
## Tips

- Write docstrings in Google format
- Keep docstrings up to date with code changes
- Preview changes locally before committing
- Use cross-references when referring to other parts of the documentation

