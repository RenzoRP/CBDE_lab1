import chromadb
import time
import numpy as np

# Function to load phrases from a text file
def load_phrases_from_file(filename):
    phrases = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            phrases.append(line.strip())  # Remove any surrounding whitespace or newline characters
    return phrases

# Function to batch data and add to Chroma, also tracks time taken for each batch
def batch_add_to_chroma(collection, phrases, batch_size=2000):
    times = []  # List to store the time taken for each batch
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i + batch_size]

        # Start timing
        start_time = time.time()

        # Add the batch to the Chroma collection
        collection.add(
            embeddings=[[0.0] * 384] * len(batch_phrases),
            metadatas=[{"phrase": phrase} for phrase in batch_phrases],
            ids=[str(j) for j in range(i, i + len(batch_phrases))]
        )

        # End timing
        end_time = time.time()

        # Calculate duration and store it
        batch_time = end_time - start_time
        times.append(batch_time)

    return times

if __name__ == "__main__":
    # Step 1: Load phrases from a text file
    phrases = load_phrases_from_file('bookcorpus_phrases.txt')

    # Step 2: Initialize Chroma client and create collection
    client = chromadb.Client()
    collection = client.create_collection("bookcorpus_phrases")

    # Step 3: Add phrases in batches and collect timings
    times = batch_add_to_chroma(collection, phrases, batch_size=2000)

    # Step 4: Calculate and display min, max, average, and standard deviation of batch times
    print(f"Minimum time: {np.min(times):.4f} seconds")
    print(f"Maximum time: {np.max(times):.4f} seconds")
    print(f"Average time: {np.mean(times):.4f} seconds")
    print(f"Standard deviation: {np.std(times):.4f} seconds")

    print("Data loaded into Chroma in batches")
