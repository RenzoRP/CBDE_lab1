from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import time

# Function to load phrases from a text file
def load_phrases_from_file(filename):
    phrases = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            phrases.append(line.strip())  # Remove any surrounding whitespace or newline characters
    return phrases

# Function to batch data and add to Chroma
def batch_add_to_chroma(collection, phrases, batch_size=2000):
    
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i + batch_size]
        # Add the batch to the Chroma collection
        collection.add(
            embeddings=[[0.0] * 384] * len(batch_phrases),
            metadatas=[{"phrase": phrase} for phrase in batch_phrases],
            ids=[str(j) for j in range(i, i + len(batch_phrases))]
        )

# Function to update embeddings in batches
def batch_update_embeddings(collection, phrases, model, batch_size=2000):
    times = []  # To store the time taken for each batch
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i + batch_size]

        # Start timing
        start_time = time.time()

        # Generate embeddings for the batch
        batch_embeddings = model.encode(batch_phrases, batch_size=batch_size)

        # Update the collection with the new embeddings
        collection.update(
            ids=[str(j) for j in range(i, i + len(batch_phrases))],
            embeddings=batch_embeddings
        )

        # End timing
        end_time = time.time()

        # Calculate duration and store it
        batch_time = end_time - start_time
        times.append(batch_time)

    return times

if __name__ == "__main__":
    #Load phrases from a text file
    phrases = load_phrases_from_file('bookcorpus_phrases.txt')

    #Initialize Chroma client and create collection
    client = chromadb.Client()
    collection = client.create_collection("bookcorpus_phrases")

    #Add phrases in batches and collect timings
    batch_add_to_chroma(collection, phrases, batch_size=2000)

    #Download the list of sentences from the collection
    data = collection.get()
    phrases = [entry['phrase'] for entry in data['metadatas']]

    #Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Update embeddings in batches and collect timings
    times = batch_update_embeddings(collection, phrases, model, batch_size=2000)

    #Calculate and display min, max, average, and standard deviation of batch times
    print(f"Minimum time: {np.min(times):.4f} seconds")
    print(f"Maximum time: {np.max(times):.4f} seconds")
    print(f"Average time: {np.mean(times):.4f} seconds")
    print(f"Standard deviation: {np.std(times):.4f} seconds")

    print("Embeddings have been updated in Chroma in batches.")