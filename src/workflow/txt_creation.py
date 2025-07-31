"""
This code provides asynchronous functions for OCR and text transcription using mLLMs.
It supports image-to-text and OCR post-processing workflows using OpenAI, Google's Gemini, and OpenRouter APIs (not implemented yet).
This code also implements retry mechanisms and concurrency control for processing multiple files quickly.

Main features:
- Direct image-to-text transcription
- OCR post-processing with image context
- Support for multiple LLM providers (OpenAI, Gemini, OpenRouter)
- Asynchronous processing with concurrency control
- Automatic retries with exponential backoff
"""

import base64
import os
from pathlib import Path
import random
import PIL
from typing import List
from openai import AsyncOpenAI

# from google import genai
from google.genai import Client
from google.genai import types
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

opensource_llms = {"qwen2.5-vl-72b-instruct": "qwen", "llama-4-maverick": "meta-llama"}


async def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string asynchronously.

    Args:
        image_path: Path to the image file to encode

    Returns:
        dict: Image content dictionary in the format required by LLM APIs
    """
    async with aiofiles.open(image_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")
    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
    }


async def openai_img2txt_async(input_img_path, output_path):
    """
    Process an image using OpenAI's GPT-4 Vision API and save the transcribed text.

    Args:
        input_img_path: Path to the input image file
        output_path: Path where the transcribed text will be saved
    """
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
    """
    Process an image using Google's Gemini Vision API and save the transcribed text.

    Args:
        input_img_path: Path to the input image file
        output_path: Path where the transcribed text will be saved
    """
    client = Client()
    img = PIL.Image.open(input_img_path)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(temperature=0),
        contents=[prompt_llm, img],
    )
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.text)


async def gemini_img_txt2txt_async(input_img_path, input_txt_path, output_path):
    """
    Process an image and OCR text using Gemini Vision API for post-processing.

    Args:
        input_img_path: Path to the input image file
        input_txt_path: Path to the input OCR text file
        output_path: Path where the processed text will be saved
    """
    client = Client()
    input = ""
    async with aiofiles.open(input_txt_path, "r") as f:
        input = await f.read()
    prompt_ocr_llm = prompt_template_ocr_llm.format(input=input).strip()
    img = PIL.Image.open(input_img_path)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(temperature=0),
        contents=[prompt_ocr_llm, img],
    )
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.text)


async def openai_img_txt2txt_async(input_img_path, input_txt_path, output_path):
    """
    Process an image and OCR text using OpenAI's GPT-4 Vision API for post-processing.

    Args:
        input_img_path: Path to the input image file
        input_txt_path: Path to the input OCR text file
        output_path: Path where the processed text will be saved
    """
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


async def openrouter_img2txt_async(input_img_path, output_path):
    """
    Process an image using OpenRouter API with various open-source LLMs.

    Args:
        input_img_path: Path to the input image file
        output_path: Path where the transcribed text will be saved
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    # Read and base64-encode image
    async with aiofiles.open(input_img_path, "rb") as f:
        img_bytes = await f.read()
    base64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Create image content in correct format
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
    }
    model = (
        opensource_llms[str(Path(output_path).parent.name)]
        + "/"
        + str(Path(output_path).parent.name)
    )
    response = await client.chat.completions.create(
        model=model,
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
            },
        ],
    )
    # Async file write
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.choices[0].message.content)


async def openrouter_img_txt2txt_async(input_img_path, input_txt_path, output_path):
    """
    Process an image and OCR text using OpenRouter API with open-source LLMs.

    Args:
        input_img_path: Path to the input image file
        input_txt_path: Path to the input OCR text file
        output_path: Path where the processed text will be saved
    """
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
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
    model = (
        opensource_llms[str(Path(output_path).parent.name)]
        + "/"
        + str(Path(output_path).parent.name)
    )
    response = await client.chat.completions.create(
        model=model,
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
            },
        ],
    )
    # Async file write
    async with aiofiles.open(output_path, "w") as f:
        await f.write(response.choices[0].message.content)


async def retry_with_backoff(fn, *args):
    """
    Execute an async function with exponential backoff retry logic.

    Args:
        fn: Async function to execute
        *args: Arguments to pass to the function

    Returns:
        The result of the successful function execution

    Raises:
        Exception: If all retry attempts fail
    """
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
    """
    Execute a processor function with concurrency control using a semaphore.

    Args:
        semaphore: Asyncio Semaphore for concurrency control
        processor: Async function to execute
        *args: Arguments to pass to the processor
    """
    async with semaphore:
        return await retry_with_backoff(processor, *args)


async def process_single_async(input_img_paths, output_dir, processor, model):
    """
    Process multiple images asynchronously with concurrency control.

    Args:
        input_img_paths: List of paths to input image files
        output_dir: Directory where output files will be saved
        processor: Async function to process each image
        model: Name of the model/directory for organizing outputs
    """
    # Array to hold all the tasks to be completed including writing to files
    tasks = []
    n = len(input_img_paths)
    max_concurrency = 4
    semaphore = asyncio.Semaphore(max_concurrency)

    for i in range(n):
        output_path = str(output_dir / model / (input_img_paths[i].stem + ".txt"))
        # Append the tasks to be executed outside the for loop
        print("Here")
        task = limited_processor(semaphore, processor, input_img_paths[i], output_path)
        tasks.append(task)
    await asyncio.gather(*tasks)


async def process_double_async(
    input_img_paths, input_txt_paths, output_dir, processor, model
):
    """
    Process multiple image-text pairs asynchronously with concurrency control.

    Args:
        input_img_paths: List of paths to input image files
        input_txt_paths: List of paths to input OCR text files
        output_dir: Directory where output files will be saved
        processor: Async function to process each image-text pair
        model: Name of the model/directory for organizing outputs

    Raises:
        NameError: If image and text file names don't match
    """
    input_img_paths = sorted(input_img_paths)
    input_txt_paths = sorted(input_txt_paths)
    n = len(input_img_paths)
    max_concurrency = 4  # Safe limit for Tier 1
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    # Validate inputs
    for i in range(n):
        if input_img_paths[i].stem != input_txt_paths[i].stem:
            raise NameError(
                f"Your image '{input_img_paths[i]}' and text '{input_txt_paths[i]}' files don't match"
            )
        print(
            f"Matched image file '{input_img_paths[i]}' with text file {input_txt_paths[i]}"
        )

    # Call function asynchronously
    for i in range(n):
        output_path = str(output_dir / model / (input_img_paths[i].stem + ".txt"))
        task = limited_processor(
            semaphore, processor, input_img_paths[i], input_txt_paths[i], output_path
        )
        tasks.append(task)
        print(
            f"Added task with image file '{input_img_paths[i]}' and text file {input_txt_paths[i]}"
        )

    await asyncio.gather(*tasks)
