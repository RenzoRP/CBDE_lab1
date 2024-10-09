import chromadb
from sentence_transformers import SentenceTransformer
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
    all_embeddings = []
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i + batch_size]
        # Generate embeddings for the batch
        batch_embeddings = model.encode(batch_phrases, batch_size=batch_size)

        # Update the collection with the new embeddings
        collection.update(
            ids=[str(j) for j in range(i, i + len(batch_phrases))],
            embeddings=batch_embeddings
        )
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

# Function to compute top-2 most similar using Euclidean and Manhattan distances
def find_similarities_euclidean(embedding, all_embeddings):
    # Compute Euclidean distances
    euclidean_distances = np.linalg.norm(all_embeddings - embedding, axis=1)
    
    # Get indices of top-2 most similar (ignoring the first result which will be itself)
    top2_euclidean = np.argsort(euclidean_distances)[1:3]
    
    return top2_euclidean

# Function to compute top-2 most similar using Manhattan distance
def find_similarities_manhattan(embedding, all_embeddings):
    
    # Compute Manhattan distances
    manhattan_distances = np.sum(np.abs(all_embeddings - embedding), axis=1)
    top2_manhattan = np.argsort(manhattan_distances)[1:3]
    
    return top2_manhattan

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
    embeddings = batch_update_embeddings(collection, phrases, model, batch_size=2000)

    # Select 10 sentences
    selected_indices = [2535, 5701, 7832, 2893, 2952, 2544, 1722, 937, 8519, 7931]
    selected_phrases = [phrases[i] for i in selected_indices]
    selected_embeddings = [embeddings[i] for i in selected_indices]

    # Lists to store the times for each distance computation
    euclidean_times = []
    manhattan_times = []

    # Compute similarities for each of the 10 selected sentences
    for i, embedding in enumerate(selected_embeddings):
        # Time the Euclidean distance calculation
        start_time = time.time()
        top2_euclidean = find_similarities_euclidean(embedding, np.array(embeddings))
        end_time = time.time()
        euclidean_times.append(end_time - start_time)

        # Time the Manhattan distance calculation
        start_time = time.time()
        top2_manhattan = find_similarities_manhattan(embedding, np.array(embeddings))
        end_time = time.time()
        manhattan_times.append(end_time - start_time)

        # Display results
        print(f"Sentence: {selected_phrases[i]}")
        print(f"Top-2 Euclidean Similar Sentences: {[phrases[idx] for idx in top2_euclidean]}")
        print(f"Top-2 Manhattan Similar Sentences: {[phrases[idx] for idx in top2_manhattan]}")
        print()

    #Calculate and display min, max, average, and standard deviation of Euclidean and Manhattan times
    print("Euclidean Distance Times:")
    print(f"Minimum time: {np.min(euclidean_times):.4f} seconds")
    print(f"Maximum time: {np.max(euclidean_times):.4f} seconds")
    print(f"Average time: {np.mean(euclidean_times):.4f} seconds")
    print(f"Standard deviation: {np.std(euclidean_times):.4f} seconds")

    print("\nManhattan Distance Times:")
    print(f"Minimum time: {np.min(manhattan_times):.4f} seconds")
    print(f"Maximum time: {np.max(manhattan_times):.4f} seconds")
    print(f"Average time: {np.mean(manhattan_times):.4f} seconds")
    print(f"Standard deviation: {np.std(manhattan_times):.4f} seconds")