# Branch merging
- [X] Merge 'visualizations' into main
- [X] If there's any other branches with non conflicting code that need to be merged we should do that as well
- [ ] Find a way to remove Jupyter metadata from git so that branch merges and pulling isn't so painful

# Project architecture
- [X] Move files such as 'workflow.md', 'process-outline.md', 'mllm-questions', 'ground-truth-guidelines.md', etc into a folder named something like 'notes', or move them to another location such as OneDrive. Possibly combine some of these files into one. 
    - [X] Similarly, examine the tesseract folder and see which of those files should be moved, renamed, or removed
- [X] Delete old pipeline file, rename new pipeline file to just 'pipeline'
- [X] 'file_retrieval' is the only file in the 'tools' folder, are there other things we should move there? 
- [X] Should 'requirements.txt' be in the config folder? What about '.gitignore'?
- [X] 'prompts' folder should either be deleted, or our new prompts should be put in there

# benchmarking-results
- [X] Make sure either NONE of our results are on the github version, or that our final results are on there

# Config
- [X] Sort requirements.txt in alphabetical order 
- [ ] Double-check that every requirement is there, and that we don't have requirements listed that we didn't end up using
- [ ] Make sure environment.yaml is up to date as well

# Benchmarking
- For all:
    - [ ] Make sure comments/attributions are up to date
    - [X] Make sure import statements at the top have unused imports removed
    - [X] Remove old commented-out code
- [X] Consistent naming scheme between 'json_accuracy_example.ipynb' and 'txt_accuracy.ipynb'

# Tools
- [X] If 'file_retrieval.py' is still in use, update comments

# Workflow
- json_creation.py
    - [X] Update comments and attributions
    - [X] Remove unused imports
    - [X] Remove or fix nonworking code
    - [X] Remove commented out code
    - [ ] If time, figure out Gemini async
- txt_creation.py
    - [ ] Update comments and attributions
    - [X] Remove unused imports
- pipeline.ipynb
    - [X] Update comments and attributions
        - [X] Write more detailed instructions for running on Windows with anaconda
        - [X] Make sure each cell is properly labled and described
    - [X] Remove or fix nonworking cells
    - [X] Remove unused imports

# README.md
- [X] Make sure everything is up to date, with instructions for how to get started
- [X] Make sure the correct attributions are made
- [X] Remove artifacts from Greif et al code that don't apply to us








