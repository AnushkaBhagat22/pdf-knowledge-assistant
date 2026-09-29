import os
import time

from dotenv import load_dotenv
from google import genai


# ============================================================
# GEMINI CLIENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(
    api_key=api_key
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    chat_history=None
):

    # --------------------------------------------------------
    # Build retrieved context
    # --------------------------------------------------------

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Build conversation context
    # --------------------------------------------------------

    conversation = ""

    if chat_history:

        conversation_parts = []

        for message in chat_history:

            role = message["role"]

            # Don't include previous error messages
            if role not in ["user", "assistant"]:
                continue

            conversation_parts.append(
                f"{role.upper()}: "
                f"{message['content']}"
            )

        conversation = "\n".join(
            conversation_parts
        )


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document-based question answering assistant.

Your job is to answer questions using ONLY the
information contained in the retrieved document context.

Do not use outside knowledge.

The user may ask follow-up questions. Use the
conversation history to understand references such as:

- it
- they
- this
- that
- its
- the above
- the previous method

However, the actual answer must still be based ONLY
on the retrieved document context.

If the answer cannot be determined from the document,
say:

"I could not find enough information in the document
to answer this question."

Keep the answer clear and concise.

--------------------------------------------------
CONVERSATION HISTORY
--------------------------------------------------

{conversation}

--------------------------------------------------
RETRIEVED DOCUMENT CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
CURRENT QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""


    # --------------------------------------------------------
    # Generate answer with retry handling
    # --------------------------------------------------------

    response = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            break

        except Exception as e:

            print(
                f"Generation attempt "
                f"{attempt + 1} failed: {e}"
            )

            if attempt < 2:

                print(
                    "Retrying in 5 seconds..."
                )

                time.sleep(5)

            else:

                raise


    return response.text

def generate_comparison(topic, document_chunks):
    """
    Generate a comparison between multiple documents
    using only the retrieved document chunks.
    """

    context_parts = []

    for source_name, chunks in document_chunks.items():

        context_parts.append(
            f"\n===== DOCUMENT: {source_name} ====="
        )

        for chunk in chunks:

            context_parts.append(
                f"[Page {chunk['page_number']}]\n"
                f"{chunk['text']}"
            )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document comparison assistant.

Compare the information about the following topic:

{topic}

Use ONLY the information provided in the documents below.

Do not use outside knowledge.

If a document does not contain enough information
about a particular aspect, explicitly say:

"Not discussed in this document."

Do not invent information.

Create a clear comparison.

Start with a short overall explanation.

Then provide a markdown table with useful comparison
aspects such as definition, structure, advantages,
limitations, memory, processing, or other aspects that
are actually supported by the documents.

After the table, mention important similarities or
differences that are supported by the documents.

Always identify which document supports the information.

DOCUMENTS:
----------------------------

{context}

----------------------------

Topic:
{topic}

Comparison:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text