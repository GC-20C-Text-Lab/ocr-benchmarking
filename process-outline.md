## steps:
1) scan tiffs of selected pages
2) OCR with tesseract
3) correct to create ground truth
4) while correcting, track errors and time
5) see if we can script JSON transformation
6) use MLLM to OCR same pages
7) track cost 
8) correct against ground truth, tracking errors and time
9) analyze the errors for each tool: what types of mistakes (eg wrong letter in a word vs. hallucination), how many, how consequential, come up with an overall accuracy metric
10) cost out scanning and OCRing the whole book with both methods


## questions:
- which MLLM to use
- how to set it up: cloud, local
- where to set it up: computer power, not timing out or getting disrupted
- are we fine-tuning, and if so, what is the process for that and how many additional pages would we need to scan for the actual comparison testing phase?


## tasks:
- scanning
- determine shared and accessible storage for scans
- create notebook for tesseract workflow

## file naming conventions
kbaa-p#
save as tiffs

Amelia: pp 3-33
Muhammad: pp 34-64
Shayak: pp 65-95
Tim: 96-126




