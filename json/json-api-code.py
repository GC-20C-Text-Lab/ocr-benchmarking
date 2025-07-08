# To run this code you need to install the following dependencies:
# pip install google-genai

# The above comment was included in the code directly copied from AI studio. 
# Is this something that we need to do with Anaconda instead?

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                #Might be the place to enter the prompt?
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "entries": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        properties = {
                            "last-name": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The author's last name. In the original entries, names are written in Last Name, First Name format.",
                            ),
                            "first-name": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The author's first name(s). In the orginal entries, names are written in Last Name, First Name format.",
                            ),
                            "maiden-name": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The author's maiden name. If they have one, it will be found in parentheses after the rest of the name.",
                            ),
                            "birth-year": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "The year the author was born. In the original entries, it will be written either as b.YEAR or YEAR-YEAR where the first year is the birth year.",
                            ),
                            "death-year": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "The year the author died. In the original entries, it will be written as YEAR-YEAR where the second year is the death year.",
                            ),
                            "title": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The title of the book. In the original entries, it should appear after the author's name and birth/death dates.",
                            ),
                            "city": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The city the book was published in. In the original entries, it is often followed by a colon. It may occasionally take the format CITY, STATE. It is acceptable to include the state in this field. In entries that include the words \"No imprint\", this information is often unavailable.",
                            ),
                            "publisher": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The name of the publisher which published the book. In the original entries, it often follows a colon. In entries that include the words \"No imprint\", this information is often unavailable.",
                            ),
                            "publish-year": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "The year the book was published. Usually seperated from the name of the publisher by a comma, although occasionally a period was used instead. In entries that include the words \"No imprint\", this information is often unavailable.",
                            ),
                            "page-count": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "The number of pages in the book. Found in the format NUMBER p.",
                            ),
                            "library": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "The abbreviated name of the library that the book was found in. In the original entries, this information is found after the page count.",
                            ),
                            "description": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "A description of the book, or the author's occupation. After the library field, the description consists of whatever is left of the entry (unless the index number is treated as occuring at the end of the entry).",
                            ),
                            "index": genai.types.Schema(
                                type = genai.types.Type.INTEGER,
                                description = "The index number of the bibliography entry. In the original entries, this information is enclosed in square brackets.",
                            ),
                        },
                    ),
                ),
            },
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()