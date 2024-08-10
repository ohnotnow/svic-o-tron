import io
import os
import re
import requests
import json
import chromadb
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from gepetto import gpt

class SearchResult():
    def __init__(self, id, document, metadata, distance):
        self.id = id
        self.document = document
        self.metadata = metadata
        self.distance = distance

def get_collection(name="svic-transcripts"):
    name = os.getenv("CHROMA_COLLECTION", name)
    client = chromadb.HttpClient(os.getenv("CHROMA_HOST", "localhost"))
    return client.get_or_create_collection(name=name)

def get_confidence_emoji(distance: float) -> str:
    if distance < 1.3:
        return "💯"  # High confidence
    elif 1.3 <= distance <= 1.5:
        return "🤔"  # Medium confidence
    else:
        return "❓"  # Low confidence

def get_cost(openai_response, model="gpt-4o-mini"):
    input_tokens = openai_response.usage.prompt_tokens
    output_tokens = openai_response.usage.completion_tokens
    tokens = input_tokens + output_tokens
    bot = gpt.GPTModelSync(model=model)
    output_cost = bot.get_token_price(output_tokens, "output", model)
    input_cost = bot.get_token_price(input_tokens, "input", model)
    cost = input_cost + output_cost
    return cost

def parse_whisper_transcript(transcript):
    structured_transcript = json.loads(transcript)
    plain_text_content = ""
    for part in structured_transcript['transcription']:
        plain_text_content += f"start_seconds: {int(part['offsets']['from'] / 1000)}\n{part['text']}\n\n"
    return plain_text_content, structured_transcript['title'], structured_transcript['youtube_url']

def parse_api_transcript(transcript):
    parsed_transcript = json.loads(transcript)
    title = parsed_transcript["data"]["transcripts"][0]["name"]
    url = parsed_transcript["youtube_url"]
    parsed_content = json.loads(parsed_transcript["data"]["transcripts"][0]["text"])
    plain_text_content = ""
    for content in parsed_content:
        plain_text_content += f"start_seconds: {int(content['start'])}\n{content['text']}\n\n"
    return plain_text_content, title, url

def split_transcript(transcript, max_chars=10000, overlap_chars=500, transcript_format="svic"):
    if transcript_format == "svic":
        plain_text_content, title, url = parse_api_transcript(transcript)
    else:
        plain_text_content, title, url = parse_whisper_transcript(transcript)
    pattern = r'start_seconds: (\d+)\n(.+?)(?=\nstart_seconds|\Z)'
    matches = re.findall(pattern, plain_text_content, re.DOTALL)
    chunks = []
    current_chunk = ''
    current_length = 0

    for i, (start_seconds, text) in enumerate(matches):
        entry = f'start_seconds: {start_seconds}\n{text}\n\n'
        entry_length = len(entry)

        if current_length + entry_length > max_chars:
            chunks.append(current_chunk)
            # Start new chunk with overlap from the previous one
            current_chunk = current_chunk[-overlap_chars:] + entry
            current_length = len(current_chunk)
        else:
            current_chunk += entry
            current_length += entry_length

    # Append the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks, title, url

def chunk_transcript(transcript, show_title, chunk_words=150, model="gpt-4o-mini"):
    chunks = []
    prompt = f"""
Please divide the following transcript into chunks for use in a Retrieval-Augmented Generation (RAG) system. The chunks should:

1. Be contextually coherent, maintaining complete sentences and thoughts, as they will be used for accurate information retrieval and generation.
2. Be approximately {chunk_words - 25}-{chunk_words + 25} words in length, allowing for slight variations to preserve coherence.
3. Align with natural breaks such as topic changes or speaker transitions, ensuring that each chunk can stand alone with a clear and complete message.
"""
    prompt += '''
Your response should be in JSON format as follows :
{
    "chunks": [
        {
            "show_title": "Title of the transcript",
            "chunk": "First chunk of transcript",
            "start_seconds": 0,
        },
        {
            "show_title": "Title of the transcript",
            "chunk": "Second chunk of transcript",
            "start_seconds": 151,
        },
        ...
    ]
}
'''
    prompt += f"""
<transcript>
Show Title: {show_title}
{transcript}
</transcript>
"""
    print(prompt)
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    cost = get_cost(response, model)
    print(f"Cost: {cost}")
    message = str(response.choices[0].message.content)
    return json.loads(message)["chunks"]

def add_to_chroma(chunks):
    print(f"Processing {len(chunks)} chunks")
    collection = get_collection()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}")
        collection.add(
            documents=[chunk['chunk']], # we embed for you, or bring your own
            metadatas=[{"source": chunk['show_title'], "seconds": chunk["start_seconds"], "url": chunk["url"]}], # filter on arbitrary metadata!
            ids=[f"{i}-{chunk['show_title']}"], # must be unique for each doc
        )

def query_to_rag_terms(query):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 'You are a helpful AI assistant who is an expert in Retrieval-Augmented Generation (RAG) systems. You are tasked with identifying key terms in a user query that can be used to retrieve relevant information from a large corpus of video transcripts. You should provide a list of terms that are likely to be present in the text that the user is looking for to give them the very best results when used in a RAG search.  Your response should be a JSON object in the format The format should be `{"terms": ["first term", "second", "third", "fourth", ...]}`',
            },
            {
                "role": "user",
                "content": f'Could you give me a JSON object containing an array of RAG terms based on this query? . <query>{query}</query>',
            }
        ],
    )
    cost = get_cost(response, model="gpt-4o-mini")
    message = str(response.choices[0].message.content)
    return json.loads(message)["terms"]

def search(query, results_limit=5):
    terms = query_to_rag_terms(query)
    collection = get_collection()
    results = collection.query(query_texts=terms, n_results=results_limit, include=["metadatas", "documents", 'distances'])
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadata = results['metadatas'][0]
    distances = results['distances'][0]
    id_document_pairs = list(zip(ids, documents, metadata, distances))
    results = ""
    result_list = []
    for id, document, meta, distance in id_document_pairs:
        seconds = meta['seconds']
        # convert seconds to mm:ss
        minutes = seconds // 60
        seconds = seconds % 60
        result_list.append(SearchResult(id, document, meta, distance))
    return result_list

def results_to_discord_message(results):
    message = ""
    for result in results:
        seconds = result.metadata['seconds']
        url = result.metadata['url']
        # convert seconds to mm:ss
        minutes = seconds // 60
        seconds = seconds % 60
        timestamp = f"{minutes:02}:{seconds:02}"
        message += f"- {result.metadata['source']} @ ~{timestamp}, Link: <{url}&t={result.metadata['seconds']}> : Confidence: {get_confidence_emoji(result.distance)} : Text: {result.document[:50]}\n"
    return message

def query(query, model="gpt-4o-mini"):
    results = search(query, results_limit=15)
    context = ""
    for result in results:
        context += f"{result.metadata['source']} @ {result.metadata['seconds'] // 60}:{result.metadata['seconds'] % 60}, Link: <{result.metadata['url']}&t={result.metadata['seconds']}> : Distance: {result.distance} : Text: {result.document}\n\n"
    prompt = f"""
    <context>
    {context}
    </context>

    {query}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = "https://api.openai.com/v1/"
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant that tries to help users with their queries.  They will sometimes give you some additional local context based on the transcripts from a video.  You should use that context where appropriate to answer their query.  If you use information from the context please reference the show title as this will help the user explore further.  Please keep your reply fairly concise and to the point.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    cost = get_cost(response, model)
    message = str(response.choices[0].message.content)
    return message

def process_transcript(transcript, show_title=None, chunk_words=150, transcript_format="svic"):
    parts, title, url = split_transcript(transcript, max_chars=10000, transcript_format=transcript_format)
    all_chunks = []
    if not show_title:
        show_title = title
    for i, part in enumerate(parts):
        chunks = chunk_transcript(part, show_title, chunk_words)
        for chunk in chunks:
            chunk["url"] = url
            all_chunks.append(chunk)
    add_to_chroma(all_chunks)
    return all_chunks

if __name__ == "__main__":
    with open("transcripts/phpugly/phpugly391.json", "r") as f:
        transcript = f.read()
    # chunks = chunk_transcript(transcript, "Show 1234", chunk_words=150)
    results = split_transcript(transcript, max_chars=10000, transcript_format="phpugly")
    print(results)
    exit()
    client = chromadb.HttpClient(os.getenv("CHROMA_HOST", "localhost"))
    client.delete_collection("svic-transcripts")
    collection = client.get_or_create_collection("svic-transcripts")
    with open("response.json", "r") as f:
        chunks = json.load(f)['chunks']
    add_to_chroma(chunks)
    results = collection.query(query_texts=["apple, cupertino, graphics"], n_results=5, include=["metadatas", "documents", 'distances',])
    ids = results['ids'][0]
    documents = results['documents'][0]
    metadata = results['metadatas'][0]
    distances = results['distances'][0]
    id_document_pairs = list(zip(ids, documents, metadata, distances))
    for id, document, meta, distance in id_document_pairs:
        seconds = meta['seconds']
        # convert seconds to mm:ss
        minutes = seconds // 60
        seconds = seconds % 60
        print(f"- {meta['source']} @ {minutes}:{seconds}, Link: https://www.youtube.com/watch?v=f5ZQVg-SmWI&t={meta['seconds']} : Distance: {distance} : Text: {document}")
