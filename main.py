"""
GEN AI APPLICATION:
AI Love Quote Generator ❤️

COMPONENTS USED:
1. Model I/O
2. Prompts
3. Chains
4. Output Parsers

NOT USED:
1. Tools
2. Agents
3. Memory
4. Retriever / VectorStore
"""

# =========================================================
# IMPORTS
# =========================================================

import os

# Load .env variables
from dotenv import load_dotenv

# MODEL I/O COMPONENT
from langchain_groq import ChatGroq

# PROMPT COMPONENT
from langchain_core.prompts import ChatPromptTemplate

# OUTPUT PARSER COMPONENT
from langchain_core.output_parsers import JsonOutputParser

# Used for structured JSON schema
from pydantic import BaseModel, Field


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


# =========================================================
# MODEL I/O COMPONENT
# =========================================================
"""
MODEL I/O:
Responsible for connecting and communicating
with the Large Language Model.
"""

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.9
)


# =========================================================
# OUTPUT PARSER COMPONENT
# =========================================================
"""
OUTPUT PARSERS:
Convert raw LLM output into structured JSON.
"""

class LoveQuotes(BaseModel):

    quote_1: str = Field(
        description="First romantic love quote"
    )

    quote_2: str = Field(
        description="Second romantic love quote"
    )


# Create parser
parser = JsonOutputParser(
    pydantic_object=LoveQuotes
)


# =========================================================
# PROMPT COMPONENT
# =========================================================
"""
PROMPTS:
Prompt templates dynamically control
the behavior of the LLM.
"""

prompt = ChatPromptTemplate.from_template(
    """
You are a romantic AI assistant.

The user's lover name is:
{lover_name}

The flower is:
{flower_name}

The favorite food is:
{food_name}

Generate:
1. Two romantic love quotes
2. Blend all three elements naturally
3. Make the quotes emotional, poetic, and sweet
4. Keep them short

{format_instructions}
"""
)


# =========================================================
# CHAIN COMPONENT
# =========================================================
"""
CHAINS:
Chains connect multiple LangChain components.

FLOW:
Prompt -> LLM -> Output Parser
"""

chain = (
    prompt
    | llm
    | parser
)


# =========================================================
# MAIN FUNCTION
# =========================================================

def generate_love_quotes(
    lover_name: str,
    flower_name: str,
    food_name: str
):
    """
    Generates romantic quotes using LangChain.
    """

    response = chain.invoke({

        "lover_name": lover_name,

        "flower_name": flower_name,

        "food_name": food_name,

        "format_instructions":
            parser.get_format_instructions()
    })

    return response


# =========================================================
# TEST RUN
# =========================================================

if __name__ == "__main__":

    result = generate_love_quotes(

        lover_name="Sophia",

        flower_name="Rose",

        food_name="Chocolate Cake"
    )

    print("\n========== LOVE QUOTES ==========\n")

    print("Quote 1:")
    print(result["quote_1"])

    print("\nQuote 2:")
    print(result["quote_2"])