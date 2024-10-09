from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import psycopg2
import time
from configparser import ConfigParser
import numpy as np

def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config

def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error during connection: {error}")
        return None
    
def create_table(conn):
    table_creation_query = '''
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        embedding FLOAT8[]
    );
    '''

    try:
        with conn.cursor() as cursor:
            start_time = time.time()  
            cursor.execute(table_creation_query)
            end_time = time.time() 
            elapsed_time = end_time - start_time
            conn.commit()
            print("Table 'embeddings' created successfully.")
            print(f"Time taken for table creation: {elapsed_time:.6f} seconds")
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error creating table: {error}")

def get_data():
    query = '''
        SELECT sentence FROM bookcorpus;
    '''

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

            conn.commit()  # Confirmar la transacción

            if data is not None:
                print("Data successfully recovered")
            else:
                print("Error getting data: 'NoneType' object is not iterable")

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error getting data: {error}")
    
    data = [record[0] for record in data] if data else []
    return data

def insert_data(conn, data):
    insert_query = """
    INSERT INTO embeddings (embedding)
    VALUES (%s);
    """

    try:

        with conn.cursor() as cursor:
             
            cursor.executemany(insert_query, [(embedding,) for embedding in data])

            conn.commit() 

            print("Embeddings batch inserted successfully!")
            print(f"Time taken for executemany: {get_data_time:.6f} seconds")

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error inserting data: {error}")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embedding(data):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    #print("Sentence embeddings:")
    #print(sentence_embeddings)

    return sentence_embeddings

def process_and_insert_data(conn, data, get_data_time):
    times = []
    batch_size = 2000
    get_sentence_time = get_data_time/len(data)
    for i in range(0, len(data), batch_size):
        start_time = time.time() 
        batch_data = data[i:i + batch_size]
        sentence_embeddings = embedding(batch_data)

        embeddings_list = [embedding.tolist() for embedding in sentence_embeddings]
        insert_data(conn, embeddings_list)
        end_time = time.time()  
        process_and_insert_time = end_time - start_time
        times.append(process_and_insert_time + get_sentence_time)
    
    return times

if __name__ == '__main__':
    config = load_config()
    conn = connect(config)

    if conn:
        print()

        # Create the embeddings table
        create_table(conn)

        # Get the sentences from the bookcorpus table
        start_time = time.time()
        data = get_data()
        end_time = time.time()  
        get_data_time = end_time - start_time
        print(f"get_data_time: {get_data_time}")

        # Process and insert the sentences in the embeddings table
        times = process_and_insert_data(conn, data, get_data_time)

        #Calculate and display min, max, average, and standard deviation of individual insert times
        print(f"Minimum time: {np.min(times):.4f} seconds")
        print(f"Maximum time: {np.max(times):.4f} seconds")
        print(f"Average time: {np.mean(times):.4f} seconds")
        print(f"Standard deviation: {np.std(times):.4f} seconds")

        conn.close()
        print("Connection closed.")