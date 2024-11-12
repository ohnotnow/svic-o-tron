import os
import re
import sys
import json
import chromadb
from openai import AsyncOpenAI
from gepetto import gpt
import asyncio
from dataclasses import dataclass

default_model = gpt.Model.GPT_4_OMNI_MINI.value[0]
@dataclass
class SearchResult():
    id: str
    document: str
    metadata: dict
    distance: float

@dataclass
class RagResponse():
    message: str
    cost: float
    model = default_model

    def __str__(self):
        return f"{self.message}\n_[Model: {self.model}, Cost: ${self.cost:.6f}]_"

async def get_collection(name="svic-transcripts") -> chromadb.Collection:
    """
    Get the Chroma collection for the transcripts.
    """
    name = os.getenv("CHROMA_COLLECTION", name)
    client = await chromadb.AsyncHttpClient(os.getenv("CHROMA_HOST", "localhost"))
    return await client.get_or_create_collection(name=name)

def get_confidence_emoji(distance: float) -> str:
    """
    Get an emoji to indicate the closeness of the RAG vector distance (ie, how similar the chunk is to the query).
    """
    if distance < 1.3:
        return "💯"  # High confidence
    elif 1.3 <= distance <= 1.5:
        return "🤔"  # Medium confidence
    else:
        return "❓"  # Low confidence

def get_cost(openai_response, model=default_model):
    """
    Get the cost of the OpenAI response.
    """
    input_tokens = openai_response.usage.prompt_tokens
    output_tokens = openai_response.usage.completion_tokens
    bot = gpt.GPTModelSync(model=model)
    output_cost = bot.get_token_price(output_tokens, "output", model)
    input_cost = bot.get_token_price(input_tokens, "input", model)
    cost = input_cost + output_cost
    return cost

def timestamp_to_seconds(timestamp: str) -> int:
    """
    Convert the HH:MM:SS timestamp to seconds.
    """
    seconds = int(timestamp.split(':')[2])
    minutes = int(timestamp.split(':')[1])
    hours = int(timestamp.split(':')[0])
    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds

def normalize_timestamp(timestamp: str) -> str:
    """
    Normalize the timestamp to HH:MM:SS.
    """
    if len(timestamp.split(':')) == 2:
        timestamp = '00:' + timestamp
    return f"{timestamp}"

def replace_timestamps_in_text(transcript):
    """
    Replace the timestamps in the transcript so they are always HH:MM:SS.
    """
    pattern = r'\((\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\)'
    updated_transcript = re.sub(pattern, lambda x: f"({normalize_timestamp(x.group(1))})", transcript)
    return updated_transcript

def parse_local_transcript(transcript):
    updated_transcript = replace_timestamps_in_text(transcript)
    return updated_transcript

def split_transcript(transcript, max_chars=8000, overlap_chars=500) -> list[str]:
    """
    Split the full transcript into smaller chunks so they fit in the context window of the LLM.
    """
    plain_text_content = parse_local_transcript(transcript)
    pattern = r'(\w+)\s*\((\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\)\n(.*?)(?=\n\w+\s*\(\d{1,2}:\d{2}|\Z)'
    matches = re.findall(pattern, plain_text_content, re.DOTALL)
    chunks = []
    combined_text = ""
    for match in matches:
        speaker, timestamp, text = match
        chunk = f"{speaker} ({timestamp}) {text.strip()}\n\n"
        if len(combined_text + chunk) > max_chars:
            chunks.append(combined_text)
            combined_text = ""
        combined_text += chunk
    if combined_text != chunk:
        chunks.append(combined_text)

    return chunks

async def chunk_transcript(transcript: str, show_title: str, url: str, chunk_words=150, model=default_model) -> tuple[list[dict], float]:
    """
    Chunk the transcript into smaller 'logical' chunks using an LLM for use in a Retrieval-Augmented Generation (RAG) system.
    """
    prompt = f"""
Please divide the following transcript into chunks for use in a Retrieval-Augmented Generation (RAG) system. The chunks should:

1. Be contextually coherent, maintaining complete sentences and thoughts, as they will be used for accurate information retrieval and generation.
2. Be approximately {chunk_words - 50}-{chunk_words + 50} words in length, allowing for slight variations to preserve coherence.
3. Align with natural breaks such as topic changes or speaker transitions, ensuring that each chunk can stand alone with a clear and complete message.
"""
    prompt += '''
Your response should be in JSON format as follows :
{
    "chunks": [
        {
            "show_title": "Title of the show",
            "url": "URL of the show",
            "chunk": "First chunk of transcript",
            "start_timestamp": "00:00:00",
            "start_seconds": 0,
            "summary": "Very short summary of the first chunk"
        },
        {
            "show_title": "Title of the show",
            "url": "URL of the show",
            "chunk": "Second chunk of transcript",
            "start_timestamp": "00:02:31",
            "start_seconds": 151,
            "summary": "Very short summary of the second chunk"
        },
        ...
    ]
}
'''
    prompt += f"""
<transcript>
Show Title: {show_title}
Show URL: {url}
{transcript}
</transcript>

Remember: You must keep the original text intact as it is for a RAG search.  Do not summarize the chunk text itself outside of the summary field.  Just break it into chunks.
"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that divides a transcript into chunks for use in a Retrieval-Augmented Generation (RAG) system.  You will be given a transcript, a show title, and a show URL.  You should divide the transcript into chunks that are approximately {chunk_words - 50}-{chunk_words + 50} words in length, maintaining complete sentences and thoughts, and aligning with natural breaks such as topic changes or speaker transitions.  You should return a JSON object in the format specified.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    cost = get_cost(response, model)
    message = str(response.choices[0].message.content)
    return json.loads(message)["chunks"], cost

async def add_to_chroma(chunks):
    collection = await get_collection()
    for i, chunk in enumerate(chunks):
        await collection.add(
            documents=[chunk['chunk']], # we embed for you, or bring your own
            metadatas=[{"source": chunk['show_title'], "seconds": chunk["start_seconds"], "url": chunk["url"], "timestamp": chunk["start_timestamp"], "summary": chunk["summary"]}],
            ids=[f"{i}-{chunk['show_title']}"], # must be unique for each doc
        )

async def query_to_rag_terms(query: str) -> tuple[list[str], float]:
    """
    Take a natural language query and return a list of terms that can be used to retrieve relevant results via a RAG search.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    system_prompt = """
    You are a helpful AI assistant who is an expert in Retrieval-Augmented Generation (RAG) systems. You are
    tasked with identifying key terms in a user query that can be used to retrieve relevant information from a large
    corpus of video transcripts. You should provide a list of terms that are likely to be present in the text
    that the user is looking for to give them the very best results when used in a RAG search.  Your response
    should be a JSON object in the format :

    {"terms": ["first term", "second", "third", "fourth", ...]}

    Please remember that the terms are used in a RAG search, so do not add too many terms of your own.  You should
    be quite conservative about adding your own terms - only add terms that you are confident will help give specific
    results to the user.

    Example of good terms:
    User query: "What did they say about Elon?"
    RAG terms: ["elon", "musk", "tesla", "spacex"]

    Example of bad terms:
    User query: "What did they say about Elon?"
    RAG terms: ["elon musk", "tesla", "spacex", "ai", "technology", "business", "startups", "venture capital", "silicon valley", "twitter", "entrepreneur", "innovation"]
    """
    response = await client.chat.completions.create(
        model=default_model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f'Could you give me a JSON object containing an array of RAG terms based on my query? The terms should assume the query is about IT, AI, technology, business, and startups if it is not clear from the query. <query>{query}</query>',
            }
        ],
    )
    cost = get_cost(response, model=default_model)
    message = str(response.choices[0].message.content)
    return json.loads(message)["terms"], cost

async def search(query: str, results_limit=5, should_rerank: bool = True, should_autorag: bool = True) -> list[SearchResult]:
    """
    Search for the query in the RAG system and return a list of show SearchResult objects.
    """
    if should_autorag:
        terms, cost = await query_to_rag_terms(query)
    else:
        cleaned_query = re.sub(r'\W+', ' ', query)
        terms = cleaned_query.split()
    collection = await get_collection()
    results = await collection.query(query_texts=[query.strip()], n_results=results_limit*2, include=["metadatas", "documents", 'distances'])
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadata = results['metadatas'][0]
    distances = results['distances'][0]
    id_document_pairs = list(zip(ids, documents, metadata, distances))
    # reorder the list based on distance - closest match to worst
    id_document_pairs.sort(key=lambda x: x[3])
    result_list = []
    counter = 0
    for id, document, meta, distance in id_document_pairs:
        if distance < 1.55:
            result_list.append(SearchResult(id, document, meta, distance))
            counter += 1
            if counter >= results_limit:
                break
    if should_rerank and should_autorag:
        result_list, cost = await rerank_results(query, result_list)
    return result_list, terms

def results_to_discord_message(results: list[SearchResult]) -> str:
    """
    Convert the list of SearchResult objects to a Discord markdown-formatted message.
    """
    if len(results) == 0:
        return ""
    message = ""
    for result in results:
        timestamp = result.metadata['timestamp']
        timestamp_parts = timestamp.split(":")
        if timestamp_parts[0] == "00":
            timestamp = f"{timestamp_parts[1]}:{timestamp_parts[2]}"
        url = result.metadata['url']
        remove_timestamp_pattern = r'\b[A-Za-z]+\s\(\d{2}:\d{2}:\d{2}\)'
        cleaned_text = re.sub(remove_timestamp_pattern, '', result.metadata['summary'])
        cleaned_text = cleaned_text.replace("\n", " ")
        cleaned_text = " ".join(cleaned_text.split()[:10])
        if '?' in url:
            url += f'&t={result.metadata["seconds"]}s'
        else:
            url += f"?t={result.metadata['seconds']}s"
        message += f"- {get_confidence_emoji(result.distance)} [{result.metadata['source']} {timestamp}](<{url}>) : _{cleaned_text}_\n"
    return message

async def query(query: str, model=default_model, should_rerank: bool = True, should_autorag: bool = True) -> RagResponse:
    """
    Perform a general query of the RAG system and return a response from the LLM.
    """
    results, terms = await search(query, results_limit=20, should_rerank=should_rerank, should_autorag=should_autorag)
    context = ""
    for result in results:
        context += f"<context-item>{result.metadata['source']} @ {result.metadata['seconds'] // 60}:{result.metadata['seconds'] % 60}, Link: <{result.metadata['url']}&t={result.metadata['seconds']}> : Distance: {result.distance} : Text: {result.document}</context-item>\n\n"
    prompt = f"""
    <context>
    {context}
    </context>

    {query}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": os.getenv("DISCORD_BOT_DEFAULT_PROMPT") + "\nThe users will sometimes give you some additional local context based on the transcripts from a podcast. The context will be ordered from most relevant to least relevant. You should use that context over your own knowledge if it is provided as the context is what the user expects to be asking about.  If you use information from the context please reference the show title and show url as this will help the user explore further (please put the show url inside angle-brackets (eg, <https://www.youtube.com/watch?v=f5ZQVg-SmWI>) so that discord does not generate a full preview of the link).  Please keep your reply fairly concise and to the point.  Remember - you do not need to mention that you are reading from the context provided - the user will assume that is the case.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    cost = get_cost(response, model)
    message = str(response.choices[0].message.content)
    return RagResponse(f"{message}", cost)

async def remove_existing_chunks(url: str):
    collection = await get_collection()
    await collection.delete(where={"url": url})

async def rerank_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """
    Rerank the results using a further LLM call based on the original user query.
    """
    prompt = f"""
    Original user query: "{query}"

    Below are {len(results)} vector database chunks from a transcript of a podcast about AI, technology and business.
    Rank these chunks based on their relevance to the user's query.
    Consider context, implied meaning, and overall relevance.  If the chunk does not
    seem relevant or is a duplicate, you should not include it in the rankings.

    Output the rankings as JSON in the format using the chunk ids:

    {{
        "rankings": [id1, id3, id4, id2, ...]
    }}
    """
    for result in results:
        prompt += f"<rag-chunk id=\"{result.id.split('-')[0]}\">{result.document}</rag-chunk>\n\n"

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1/")
    response = await client.chat.completions.create(
        model=default_model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={"type": "json_object"},
    )
    cost = get_cost(response, model=default_model)
    message = str(response.choices[0].message.content)
    try:
        decoded_response = json.loads(message)["rankings"]
    except json.JSONDecodeError:
        return results, cost
    ranked_results = []
    for i, document_id in enumerate(decoded_response):
        for result in results:
            numeric_id = result.id.split("-")[0].strip()
            if int(numeric_id) == int(document_id):
                ranked_results.append(result)
                break
    return ranked_results, cost

async def export_collection_to_json(collection_name: str = "svic-transcripts") -> dict:
    collection = await get_collection(collection_name)
    data = await collection.get()
    return data

def transcript_looks_valid(transcript: str) -> bool:
    """
    Check if the transcript looks valid.
    """
    plain_text_content = parse_local_transcript(transcript)
    pattern = r'(\w+)\s*\((\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\)\n(.*?)(?=\n\w+\s*\(\d{1,2}:\d{2}|\Z)'
    matches = re.findall(pattern, plain_text_content, re.DOTALL)
    if len(matches) < 2 or len(plain_text_content) < 1000:
        return False
    return True

async def process_transcript(transcript: str, show_title: str = "", url: str = "", chunk_words: int = 150, transcript_format: str = "svic") -> RagResponse:
    """
    Process the transcript and add it to the RAG system.
    """
    if not transcript_looks_valid(transcript):
        return RagResponse("Transcript does not look valid", 0)

    await remove_existing_chunks(url)

    parts = split_transcript(transcript)
    all_chunks = []
    total_cost = 0

    chunk_tasks = [chunk_transcript(part, show_title, url, chunk_words) for part in parts]
    results = await asyncio.gather(*chunk_tasks)

    for chunks, cost in results:
        total_cost += cost
        all_chunks.extend(chunks)

    await add_to_chroma(all_chunks)
    return RagResponse(f"Added {len(all_chunks)} chunks to the RAG system.", total_cost)

async def main(path_to_transcript_files: str):
    for filename in os.listdir(path_to_transcript_files):
        with open(os.path.join(path_to_transcript_files, filename), "r") as f:
            transcript = f.read()
            print(f"Processing {filename}")
            lines = transcript.split("\n")
            show_title = lines[0]
            url = lines[1]
            if not url.startswith("https://"):
                print(f"Skipping {filename} as it does not have a valid URL")
                continue
            stats, cost = await process_transcript("\n".join(lines[2:]), show_title, url)
            print(stats, cost)
    # response = await query("What did they say about the new AirPods?")
    # print(response)

if __name__ == "__main__":
    # path to transcript files should be passed in as an argument
    if len(sys.argv) != 2:
        print("Usage: python gepetto/rag.py <path_to_transcript_files>")
        sys.exit(1)
    path_to_transcript_files = sys.argv[1]
    asyncio.run(main(path_to_transcript_files))
