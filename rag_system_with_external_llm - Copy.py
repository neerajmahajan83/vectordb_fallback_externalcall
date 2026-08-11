import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer
from groq import Groq

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

# ── Groq client ────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"  # free, fast, very capable

documents = [
    "To track your Amazon order, log into your account, go to 'Your Orders,' and click 'Track Package' for real-time updates.",
    "Amazon's return policy allows most items to be returned within 30 days of delivery for a full refund, provided they are in new condition with original packaging and accessories.",
    "To return an Amazon order, initiate a return through 'Your Orders,' ship the item back, and receive a refund once processed.",
    "To contact Amazon customer service, use the 'Help' section on the website or app to chat, call, or email support.",
    "Amazon Prime members receive free two-day shipping, exclusive deals, and access to Prime Video and Music.",
    "If your Amazon package is delayed, check the estimated delivery date in 'Your Orders' or contact customer service for assistance.",
    "To cancel an Amazon order, go to 'Your Orders,' select the order, and click 'Cancel Items' if it hasn't shipped yet.",
    "To purchase an Amazon gift card, visit the Amazon website, navigate to 'Gift Cards,' select a design and amount, add to cart, and complete the purchase at checkout; the gift card can be redeemed for eligible products.",
    "To update your Amazon payment method, go to 'Your Account,' select 'Your Payments,' and add or edit your card details.",
    "To log into your Amazon account, go to the Amazon website or app, click 'Sign In,' and enter your email or phone number and password."
]

# ── Build FAISS index ──────────────────────────────────────────────────────────
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

# ── Load local flan-t5-small ───────────────────────────────────────────────────
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')


def answer_with_local_t5(query, retrieved_doc):
    """Use local flan-t5-small with retrieved context."""
    prompt = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
    inputs = t5_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    output_ids = t5_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        min_new_tokens=5,
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True,
        forced_eos_token_id=t5_tokenizer.eos_token_id,
    )
    response = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return response if response else retrieved_doc


def answer_with_groq(query):
    """Fallback: send query directly to Groq LLM when vector DB has no match."""
    print("[INFO] No relevant document found in vector DB — calling Groq LLM...\n")
    try:
        chat_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "clearly and concisely in one or two sentences."
                    )
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"External LLM error: {e}"


def rag_answer(query, top_k=1, threshold=0.3):
    """
    RAG pipeline:
      score >= threshold  →  local flan-t5 answers using retrieved context
      score <  threshold  →  Groq LLM answers the query directly (no context)
    """
    # Step 1: embed query and search FAISS
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    score = distances[0][0]

    # Step 2: route based on retrieval score
    if score >= threshold:
        # Answer found in vector DB — use local model
        retrieved_doc = documents[indices[0][0]]
        response = answer_with_local_t5(query, retrieved_doc)
        source = "vector DB + local T5"
    else:
        # No relevant document — fall back to Groq
        retrieved_doc = None
        response = answer_with_groq(query)
        source = "Groq LLM (external)"

    return retrieved_doc, response, source


def run_qa_bot():
    print("Welcome to the RAG Q&A Bot!")
    print("  - Questions about Amazon → answered from local vector DB")
    print("  - Other questions        → answered by Groq LLM (external)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        _, answer, source = rag_answer(query)
        print(f"HelpBot [{source}]: {answer}\n")


run_qa_bot()
