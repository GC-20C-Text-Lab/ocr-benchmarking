import base64
import random
import PIL
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
Your task is to transcribe this image of a historical bibliography page as faithfully as possible.
Only transcribe typed text that appears on the page and do not attempt to predict missing information or complete cut off entries. 
Put each entry on a separate line. When an entry has an index number in square brackets, place it at the end of the entry. 
"""

prompt_template_ocr_llm = """
You are a text correction assistant. Your task is to clean up and correct errors from raw OCR output.
The text may contain misrecognized characters, broken words, or incorrect formatting.
Carefully read the provided OCR output, compare it to the original image, and produce a corrected version that is  
as faithful to the original content as possible. Only correct obvious OCR errors, and do not attempt to complete
cut-off entries or predict missing information. Put each entry on a separate line.
When an entry has an index number in square brackets, place it at the end of the entry.
Input (Raw OCR Text):
{input}
"""


async def encode_image_to_base64(image_path):
    async with aiofiles.open(image_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
    }


async def openai_img2txt_async(input_img_path, output_path):
    client = AsyncOpenAI()
    # Read and base64-encode image
    async with aiofiles.open(input_img_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
    }

    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
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


async def gemini_img2txt_async(input_img_path, output_path):
    client = Client()
    img = PIL.Image.open(input_img_path)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", contents=[prompt_llm, img]
    )
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.text)


async def openai_img_txt2txt_async(input_img_path, input_txt_path, output_path):
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
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
    }
    prompt_ocr_llm = prompt_template_ocr_llm.format(input=input).strip()
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
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


async def retry_with_backoff(fn, *args):
    retries, base_delay = 5, 2

    for attempt in range(retries):
        try:
            return await fn(*args)
        except Exception as e:
            if attempt < retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f}s after error: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"Failed after {retries} attempts: {e}")
                raise


async def limited_processor(semaphore, processor, *args):
    async with semaphore:
        return await retry_with_backoff(processor, *args)


async def process_single_async(input_img_paths, output_dir, processor, model):
    # Array to hold all the tasks to be completed including writing to files
    tasks = []
    n = len(input_img_paths)
    max_concurrency = 4
    semaphore = asyncio.Semaphore(max_concurrency)

    for i in range(n):
        output_path = str(output_dir / model / (input_img_paths[i].stem + ".txt"))

        # Append the tasks to be executed outside the for loop
        task = limited_processor(semaphore, processor, input_img_paths[i], output_path)
        tasks.append(task)
    await asyncio.gather(*tasks)


async def process_double_async(
    input_img_paths, input_txt_paths, output_dir, processor, model
):
    n = len(input_img_paths)
    max_concurrency = 4  # Safe limit for Tier 1
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    for i in range(n):
        output_path = str(output_dir / model / (input_img_paths[i].stem + ".txt"))
        task = limited_processor(
            semaphore, processor, input_img_paths[i], input_txt_paths[i], output_path
        )
        tasks.append(task)

    await asyncio.gather(*tasks)
