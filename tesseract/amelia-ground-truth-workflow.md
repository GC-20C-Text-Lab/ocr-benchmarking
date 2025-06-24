# OCR
- Put all tif images into the same directory (scans-tif)
- Create blank text files for the raw output and put them in a separate directory (raw-ocr)
- Run python script (amelia-tesseract-workflow) 

# LLM
- Save all tif images as PNGs 
- Enter this prompt into Gemini 2.5 Flash:
    - Please correct the Tesseract transcription of this bibliography page using the scanned image of the page and provide the output in a txt file. Where index numbers in square brackets appear, they should be moved to the end of the entry. Put each entry on a separate line. Only transcribe text that appears on the page and do not attempt to predict missing information or complete cut off entries.
- Attach PNG and raw ocr text file for one page to the prompt
- Copy and paste the gemini output into a text file in a separate directory (gemini-output)

# Ground truth
- Also paste that same gemini output into a text file in another directory (ground-truth)
- Manually look back and forth between the original image and the gemini output, and correct any errors
- Optionally, record the types of errors that appear (I would put the incorrect word (or a word with an incorrect character) on the left and then an arrow -> to the corrected output. If it was a deletion of an entire word or punctuation mark, I would type 'delete' and then what was deleted. If it was an insertion of a missing word or punctuation, I would type 'insert' and then what was inserted)

- Repeat LLM and ground truth process for each image you want the ground truth for