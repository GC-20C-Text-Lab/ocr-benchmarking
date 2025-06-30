import sys
from pydantic import BaseModel, Field
import instructor
from typing import List
from openai import OpenAI
import os
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))


client = instructor.from_openai(OpenAI())

class Entry(BaseModel):
    lastname: str = Field(default = "", description = "The author's last name. In the original entries, names are written in Last Name, First Name format. If the name is followed by \"(pseud.)\", this indicates that the name is a pseudonym, but it should still be stored in this field.")
    firstname: str = Field(default = "", description = "The author's first name(s). In the orginal entries, names are written in Last Name, First Name format.")
    maidenname: str = Field(default = "", description = "The author's maiden name. If they have one, it will be found in parentheses after the rest of the name.")
    birthyear: int = Field(default = 0, description = "The year the author was born. In the original entries, it will be written either as b.YEAR or YEAR-YEAR where the first year is the birth year.")
    deathyear: int = Field(default = 0, description = "The year the author died. In the original entries, it will be written as YEAR-YEAR where the second year is the death year.")
    title: str = Field(default = "", description = "The title of the book. In the original entries, it should appear after the author's name and birth/death dates.")
    city: str = Field(default = "", description = "The city the book was published in. In the original entries, it is often followed by a colon. It may occasionally take the format CITY, STATE. It is acceptable to include the state in this field. In entries that include the words \"No imprint\", this information is often unavailable.")
    publisher: str = Field(default = "", description = "The name of the publisher which published the book. In the original entries, it often follows a colon. In entries that include the words \"No imprint\", this information is often unavailable.")
    publishyear: int = Field(default = 0, description = "The year the book was published. Usually seperated from the name of the publisher by a comma, although occasionally a period was used instead. In entries that include the words \"No imprint\", this information is often unavailable.")
    pagecount: int = Field(default = 0, description = "The number of pages in the book. Found in the format NUMBER p.")
    library: str = Field(default = "", description = "The abbreviated name of the library that the book was found in. In the original entries, this information is found after the page count.")
    description: str = Field(default = "",description = "A description of the book, or the author's occupation. After the library field, the description consists of whatever is left of the entry (unless the index number is treated as occuring at the end of the entry). In entries where the author uses a pseudonym, the information after \"(pseud.)\" should be included in this field.")
    index: int = Field(default = 0, description = "The index number of the bibliography entry. In the original entries, this information is enclosed in square brackets. Entries that redirect elsewhere due to the author using a pseudonym do not have an index number.")

    

class Entries(BaseModel):
    entries: List[Entry]

text = ""
with open (os.path.join(project_root, "ignore", "gt_kbaa-p003.txt"), "r") as file:
    text = file.read()

entries = client.chat.completions.create(
    model = "gpt-4o-mini",
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

with open (os.path.join(project_root, "json", "instructor-test-output.json"), "w") as file:
    file.write(entries.model_dump_json(indent=2, exclude_defaults=True))