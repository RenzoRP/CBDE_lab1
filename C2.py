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

# Load your dataset
phrases = load_phrases_from_file('bookcorpus_phrases.txt')

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(phrases, batch_size=32)

# Initialize Chroma client and create collection
client = chromadb.Client()
collection = client.create_collection("bookcorpus_phrases")

# Add the individual embeddings to the Chroma collection
for i, (phrase, embedding) in enumerate(zip(phrases, embeddings)):
    collection.add(
        embeddings=[embedding],
        metadatas=[{"phrase": phrase}],
        ids=[str(i)]
    )

# Select 10 sentences (by their index, for example)
selected_indices = [2535, 5701, 7832, 2893, 2952, 2544, 1722, 937, 8519, 7931]
selected_phrases = [phrases[i] for i in selected_indices]
selected_embeddings = [embeddings[i] for i in selected_indices]

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
