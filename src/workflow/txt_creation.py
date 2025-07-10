import base64
from pydantic import BaseModel, Field
import instructor
from typing import List
from openai import OpenAI, AsyncOpenAI
from instructor import from_openai, from_genai, Mode
import anthropic

# from google import genai
from google.genai import Client
import aiofiles
import asyncio

prompt_llm = """
You are an expert historian. Your task is to transcribe the provided image into text. The image
is a 20th century bibliographic entry. Because this is a historical document, try to preserve 
archaic spelling or formatting where clearly intended. Put the dates associated with each entry at the end of the line.
Return the text only, nothing else.
"""

prompt_template_ocr_llm = """
You are a text correction assistant. Your task is to clean up and correct errors from raw OCR output.
The text may contain misrecognized characters, broken words, or incorrect formatting.
Carefully read the provided OCR output and produce a corrected version that is grammatically accurate 
and as faithful to the original content as possible. Because this is a historical document, try to 
preserve archaic spelling or formatting where clearly intended. Only correct obvious OCR errors.
Put the dates associated with each entry at the end of the line.

Input (Raw OCR Text):
{input}
"""

async def openai_img2txt_async(input_img_path, output_path):
    client = AsyncOpenAI()
    # Read and base64-encode image
    async with aiofiles.open(input_img_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img}"
        },
    }
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": prompt_llm,
                    },
                ],
            }
        ],
    )
    # Async file write
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.choices[0].message.content)

async def openai_img_txt2txt_async(input_img_path, input_txt_path, output_path):
    print(input_txt_path)
    client = AsyncOpenAI()
    input = ""
    async with aiofiles.open(input_txt_path, "r") as f:
        input = await f.read()

    # Read and base64-encode image
    async with aiofiles.open(input_img_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_img}"
        },
    }
    prompt_ocr_llm = prompt_template_ocr_llm.format(input=input).strip()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": prompt_ocr_llm,
                    },
                ],
            }
        ],
    )
    # Async file write
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.choices[0].message.content)

async def process_single_async(input_img_paths, output_dir, processor, doc_format, model):
    # Array to hold all the tasks to be completed including writing to files
    tasks = []
    n = len(input_img_paths)
    for i in range(n//2):
        output_path = str(
            output_dir
            / model
            / (input_img_paths[i].stem + f".{doc_format}")
        )
        # Append the tasks to be executed outside the for loop
        tasks.append(processor(input_img_paths[i], output_path))
    await asyncio.gather(*tasks)
    await asyncio.sleep(60)
    tasks = []
    for i in range(n//2, n):
        output_path = str(
            output_dir
            / model
            / (input_img_paths[i].stem + f".{doc_format}")
        )
        # Append the tasks to be executed outside the for loop
        tasks.append(processor(input_img_paths[i], output_path))
    await asyncio.gather(*tasks)



async def process_double_async(input_img_paths, input_txt_paths, output_dir, processor, doc_format, model):
    # Array to hold all the tasks to be completed including writing to files
    tasks = []
    n = len(input_img_paths)
    # count = 0  # THis is just to test out # TODO: remove in final pipeline
    for i in range(n//2):
        output_path = str(
            output_dir
            / model
            / (input_img_paths[i].stem + f".{doc_format}")
        )

        # Append the tasks to be executed outside the for loop
        tasks.append(processor(input_img_paths[i], input_txt_paths[i], output_path))
    await asyncio.gather(*tasks)
    await asyncio.sleep(60)
    tasks = []
    for i in range(n//2, n):
        output_path = str(
            output_dir
            / model
            / (input_img_paths[i].stem + f".{doc_format}")
        )

        # Append the tasks to be executed outside the for loop
        tasks.append(processor(input_img_paths[i], input_txt_paths[i], output_path))
    await asyncio.gather(*tasks)