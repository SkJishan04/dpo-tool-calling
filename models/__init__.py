"""Models module for DPO tool calling project."""# ============================================================================
# CELL X: Create Missing __init__.py Files
# ============================================================================

import os

# Create __init__.py in all directories
directories = [
    '/content/dpo-tool-calling',
    '/content/dpo-tool-calling/data',
    '/content/dpo-tool-calling/models',
    '/content/dpo-tool-calling/config',
    '/content/dpo-tool-calling/evaluation',
    '/content/dpo-tool-calling/utils',
    '/content/dpo-tool-calling/scripts',
    '/content/dpo-tool-calling/tests',
]

for directory in directories:
    init_file = os.path.join(directory, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('')
        print(f"✅ Created: {init_file}")
    else:
        print(f"✓ Already exists: {init_file}")

print("\n✅ All __init__.py files created!")