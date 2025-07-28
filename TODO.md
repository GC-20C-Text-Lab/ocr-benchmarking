# Branch merging
- [ ] Merge 'visualizations' into main
- [ ] If there's any other branches with non conflicting code that need to be merged we should do that as well
- [ ] Find a way to remove Jupyter metadata from git so that branch merges and pulling isn't so painful

# Project architecture
- [ ] Move files such as 'workflow.md', 'process-outline.md', 'mllm-questions', 'ground-truth-guidelines.md', etc into a folder named something like 'notes', or move them to another location such as OneDrive. Possibly combine some of these files into one. 
    - [ ] Similarly, examine the tesseract folder and see which of those files should be moved, renamed, or removed
- [ ] Delete old pipeline file, rename new pipeline file to just 'pipeline'
- [ ] 'file_retrieval' is the only file in the 'tools' folder, are there other things we should move there? 
- [ ] Should 'requirements.txt' be in the config folder? What about '.gitignore'?
- [ ] 'prompts' folder should either be deleted, or our new prompts should be put in there

# benchmarking-results
- [ ] Make sure either NONE of our results are on the github version, or that our final results are on there

# Config
- [ ] Sort requirements.txt in alphabetical order? Is that something people do? We also should double-check that every requirement is there, and that we don't have requirements listed that we didn't end up using
- [ ] Make sure environment.yaml is up to date as well

# Benchmarking
- [ ] For all:
    - [ ] Make sure comments/attributions are up to date
    - [ ] Make sure import statements at the top have unused imports removed
    - [ ] Remove old commented-out code
- [ ] Consistent naming scheme between 'json_accuracy_example.ipynb' and 'txt_accuracy.ipynb'

# Tools
- [ ] If 'file_retrieval.py' is still in use, update comments

# Workflow
- [ ] json_creation.py
    - [ ] Update comments and attributions
    - [ ] Remove unused imports
    - [ ] Remove or fix nonworking code
    - [ ] Remove commented out code
    - [ ] If time, figure out Gemini async
- [ ] txt_creation.py
    - [ ] Update comments and attributions
    - [ ] Remove unused imports
- [ ] txt_pipeline.ipynb
    - [ ] Update comments and attributions
        - [ ] Write more detailed instructions for running on Windows with anaconda
        - [ ] Make sure each cell is properly labled and described
    - [ ] Remove or fix nonworking cells
    - [ ] Remove unused imports

# README.md
- [ ] Make sure everything is up to date, with instructions for how to get started
- [ ] Make sure the correct attributions are made
- [ ] Remove artifacts from Greif et al code that don't apply to us








