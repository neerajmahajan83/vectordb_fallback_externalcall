from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import T5ForConditionalGeneration, T5Tokenizer


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

# Build FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Load T5 model directly (pipeline not used — incompatible with newer transformers)
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')


def rag_answer(query, top_k=1, threshold=0.3):
    # Embed and retrieve
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    if distances[0][0] < threshold:
        return None, "Sorry, I couldn't find a relevant answer to your question."

    retrieved_doc = documents[indices[0][0]]

    # Build prompt and generate
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

    # Fallback: return the retrieved document if model gives empty output
    if not response:
        response = retrieved_doc

    return retrieved_doc, response


def run_qa_bot():
    print("Welcome to the RAG Q&A Bot! Ask a question or type 'exit' to quit.")
    while True:
        query = input("User: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue
        _, answer = rag_answer(query)
        print(f"HelpBot: {answer}\n")


run_qa_bot()
