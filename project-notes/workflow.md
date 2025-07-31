# Our proposed workflow

## Ground truth guidelines:
[Ground Truth Guidelines](./ground-truth-guidelines.md)

# Other examples
## Greif et al.

Corpus size: 30 pages
Two leading LLMs: Gemini 2.0 Flash and GPT-4o-2024-08-06
OCR models: Transkibus Text Titan I and Print M1, Tesseract 5.5.0

Traditional pipline:
- Pre-processing
    - Remove excess visual information
    - De-skew the image
    - Grayscaling/increasing contrast
    - Upsampling degraded images
- layout recognition
    - image segmentation
    - line recognition
    - baseline detection
    - reading order determination
- character recognition
    - can fine tune a model to the specific historical font of your corpus
- post-processing
    - manual post-correction
    - rules-based (dictionary) post-correction
    - probabalistic models
- Named Entity Recognition 
    - BERT

LLM experimentation
- Set temperature parameter to 0.0
- PNG and text prompt uploaded together via the API 
- Output concatenated into a txt file

OCR experimentation
- No pre or post processing of the transcription
- Transkribus API fed individual images and convert XML files into TXT files
- No separate layout recognition that's not inherent to the model

OCR post-correction
- Best transcription by OCR model fed with the prompt and the original PNG into the mLLM one page at a time
- Plain text in a TXT file returned

NER with mLLMs
- Identify variables 
- Prompt mLLM to return structured output in JSON
- Prompt included examples of format
- convert JSON to CSV
- First test on ground truth document
- Second test on noisy transcription
- Third test on image itself

Prompt development
- Ask AI to generate prompt after describing the problem
- Test the prompt
- Ask AI to amend the prompt to fix specific issues noticed
- AI written prompt includes loaded language and high stakes

Evaluation metrics
- Levenshtein Distance (insertions, deletions, or substitutions to match ground truth)
- Character Error Rate (Levenshtein distance divided by total number of characters in ground truth)
- Word Error Rate (sum of insertions, deletions, and substitutions at a word level divided the total number of words)
- Normalized (limited to ASCII characters, no punctuation or capitalization) vs non normalized
- NER matching: strict (exact string comparison binary result) vs fuzzy (Jaro-Winkler similarity with common scaling factor of 0.9)

Price/time according to this article
- Gemini 2.0 Flash is free for up to 15 requests per minute, up to 1 million tokens per minute, and up to 1,500 requests per day, 11.5 "s/page" (seconds per page? the unit is never exactly stated)
- GPT-4o is $10.84 for 1,000 pages, at 18.23 s/page
- Tesseract is free, at 13.77 s/page (but apparently its transcriptions had too many errors for the authors of the paper to include it in the table). They found Transkribus to be more accurate (but it is not free, at $54.25 per 1,000 pages)

Results
- Gemini 2.0 Flash was the most accurate with no pre-processing, post-processing, or model-specific fine tuning
- For post-correction, the most accurate method was using Gemini 2.0 Flash on the "noisy transcription" 
- The LLMs were very succesful in identifying named entities in the "ground truth" documents, and were mostly successful in identifying named entities in the OCR transcripts and the original PNGs

## questions

### for Tesseract
- do we need to do layout recognition?
- is there a way to think about an entry by entry approach that doesn't require crazy scanning?
- would that approach be an improvement?



