from pydantic import BaseModel, Field
import instructor
from typing import List
from openai import OpenAI
import anthropic
#from google import genai
from google.genai import Client
#import os
#need to install json-ref even though it's not listed in imports

#script_dir = os.path.dirname(os.path.realpath(__file__))
#project_root = os.path.abspath(os.path.join(script_dir, ".."))

# JSON schema for mLLMs to strictly follow
class Entry(BaseModel):
    lastname: str = Field(default = "", description = "The author's last name. In the original entries, names are written in Last Name, First Name format. If the name is followed by \"(pseud.)\", this indicates that the name is a pseudonym, but it should still be stored in this field. If this information is not found, assign the empty string to this field.")
    firstname: str = Field(default = "", description = "The author's first name(s). In the orginal entries, names are written in Last Name, First Name format. If this information is not found, assign the empty string to this field.")
    maidenname: str = Field(default = "", description = "The author's maiden name. If they have one, it will be found in parentheses after the rest of the name. If this information is not found, assign the empty string to this field.")
    birthyear: int = Field(default = 0, description = "The year the author was born. In the original entries, it will be written either as b.YEAR or YEAR-YEAR where the first year is the birth year.If this information is not found, assign 0 to this field.")
    deathyear: int = Field(default = 0, description = "The year the author died. In the original entries, it will be written as YEAR-YEAR where the second year is the death year. If this information is not found, assign 0 to this field.")
    title: str = Field(default = "", description = "The title of the book. In the original entries, it should appear after the author's name and birth/death dates. If this information is not found, assign the empty string to this field.")
    city: str = Field(default = "", description = "The city the book was published in. In the original entries, it is often followed by a colon. It may occasionally take the format CITY, STATE. It is acceptable to include the state in this field. In entries that include the words \"No imprint\", this information is often unavailable. If this information is not found, assign the empty string to this field.")
    publisher: str = Field(default = "", description = "The name of the publisher which published the book. In the original entries, it often follows a colon. In entries that include the words \"No imprint\", this information is often unavailable. If this information is not found, assign the empty string to this field.")
    publishyear: int = Field(default = 0, description = "The year the book was published. Usually seperated from the name of the publisher by a comma, although occasionally a period was used instead. In entries that include the words \"No imprint\", this information is often unavailable. If this information is not found, assign 0 to this field.")
    pagecount: int = Field(default = 0, description = "The number of pages in the book. Found in the format NUMBER p. If this information is not found, assign 0 to this field.")
    library: str = Field(default = "", description = "The abbreviated name of the library that the book was found in. In the original entries, this information is found after the page count. If this information is not found, assign the empty string to this field.")
    description: str = Field(default = "",description = "A description of the book, or the author's occupation. After the library field, the description consists of whatever is left of the entry (unless the index number is treated as occuring at the end of the entry). In entries where the author uses a pseudonym, the information after \"(pseud.)\" should be included in this field. If this information is not found, assign the empty string to this field.")
    index: int = Field(default = 0, description = "The index number of the bibliography entry. In the original entries, this information is enclosed in square brackets. Entries that redirect elsewhere due to the author using a pseudonym do not have an index number. If this information is not found, assign 0 to this field.")

# Container so that multiple entries are contained in a single JSON object
class Entries(BaseModel):
    entries: List[Entry]

#Takes as input the path to a text file and returns formatted JSON following the Entries schema
# OpenAI version
def openai_txt2json(path):
    # Convert contents of the file to a string
    text = ""
    with open(path,"r") as file:
        text = file.read()
    # Create OpenAI client
    client = instructor.from_openai(OpenAI())
    # Call the API
    entries = client.chat.completions.create(
        model = "gpt-4o",
        response_model=Entries,
        messages=
        [
            {
                "role": "user", 
                "content": 
                [
                    {
                        "type" : "text",
                        "text" : "Convert each entry in this bibliography into structured JSON:\n" + text
                    }
                ]
            }
        ],
    )
    # Return JSON, with 2 spaces of indentation and default values excluded 
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(path, "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

# Takes as input the path to an image and returns formatted JSON following the Entries schema
# OpenAI version
def openai_img2json(path):
    # Create OpenAI client
    client = instructor.from_openai(OpenAI())
    # Call the API
    entries = client.chat.completions.create(
        model = "gpt-4o",
        response_model=Entries,
        messages=
        [
            {
                "role": "user", 
                "content": 
                [
                    instructor.Image.from_path(path),
                    {
                        "type" : "text",
                        "text" : "Using this scanned image of the page, convert each entry in this bibliography into structured JSON.\n"
                    }
                ]
            }
        ],
    )
    # Return JSON, with 2 spaces of indentation and default values excluded
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(path, "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

# Takes as input the path to a text file and returns formatted JSON following the Entries schema
# Google version
def gemini_txt2json(path):
    # Convert contents of the file to a string
    text = ""
    with open(path,"r",) as file:
        text = file.read()
    # Create the Google GenAI client
    client = instructor.from_genai(Client(), instructor.Mode.GENAI_STRUCTURED_OUTPUTS)
    # Call the API
    entries = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_model=Entries,
        messages=[
            {
                "role": "user",
                "content": "Convert each entry in this bibliography into structured JSON:\n"
                + text,
            }
        ],
    )
    # Return JSON, with 2 spaces of indentation and default values excluded
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(path, "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

# Takes as input the path to an image and returns formatted JSON following the Entries schema
def gemini_img2json(path):
    # Create the Google GenAI client
    client = instructor.from_genai(Client(), instructor.Mode.GENAI_STRUCTURED_OUTPUTS)
    # Call the API
    entries = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_model=Entries,
        messages=[
            {
                "role": "user",
                "content": [
                    instructor.Image.from_path(path),
                    "Using this scanned image of the page, convert each entry in this bibliography into structured JSON.",
                ],
            }
        ],
    )
    # Return JSON, with 2 spaces of indentation and default values excluded
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(path, "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

# Takes as input the path to a text file and returns formatted JSON following the Entries schema
# Anthropic version
def anthropic_txt2json(path):
    # Convert contents of the file to a string
    text = ""
    with open(path,"r") as file:
        text = file.read()
    # Create the Anthropic client
    client = instructor.from_anthropic(anthropic.Anthropic())
    # Call the API
    entries = client.chat.completions.create(
        model="claude-sonnet-4-20250514",
        response_model=Entries,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert each entry in this bibliography into structured JSON:\n"
                        + text,
                    }
                ],
            }
        ],
        max_tokens=10000,
    )
    # Return JSON, with 2 spaces of indentation and default values excluded
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(os.path.join(project_root, "json", "anthropictxt2json_kbaa-p003.json"), "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

# Takes as input the path to an image file and returns formatted JSON following the Entries schema
# Anthropic version
# Max image size: 5MB
def anthropic_img2json(path):
    # Create the Anthropic client
    client = instructor.from_anthropic(anthropic.Anthropic())
    # Call the API
    entries = client.chat.completions.create(
        model="claude-sonnet-4-20250514",
        response_model=Entries,
        messages=[
            {
                "role": "user",
                "content": [
                    instructor.Image.from_path(path),
                    {
                        "type": "text",
                        "text": "Using this scanned image of the page, convert each entry in this bibliography into structured JSON.\n",
                    },
                ],
            }
        ],
        max_tokens=10000,
    )
    # Return JSON, with 2 spaces of indentation and default values excluded
    return entries.model_dump_json(indent=2, exclude_defaults=True)
    #with open(os.path.join(project_root, "json", "anthropicimg2json_kbaa-p003.json"), "w") as file:
        #file.write(entries.model_dump_json(indent=2, exclude_defaults=True))

