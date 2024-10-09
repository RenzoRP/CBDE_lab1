import psycopg2
import time
import re
from configparser import ConfigParser
from datasets import load_dataset

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
    
def download_dataset():
    ds = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)

    # Initialize an empty list to store phrases
    phrases = []

    # Regular expression to split by '. ' but ignore periods inside quotes or abbreviations
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.)\s+'

    # Iterate through the stream and process only a limited number of entries
    for i, entry in enumerate(ds):
        # Use the regular expression to split sentences by '. '
        sentences = re.split(pattern, entry['text'])

        # Add each sentence as a tuple to the list of phrases
        cleaned_phrases = []
        temp_sentence = ""

        for sentence in sentences:
            # Join sentences that are part of dialogues or incomplete
            if sentence.startswith("''") or sentence == '':
                temp_sentence += sentence
            else:
                # If we have accumulated part of a sentence, append it before moving to the next one
                if temp_sentence:
                    cleaned_phrases.append((temp_sentence.strip(),))
                    temp_sentence = ""
                cleaned_phrases.append((sentence.strip(),))

        # Store the cleaned phrases
        phrases.extend(cleaned_phrases)

        # Stop after processing 10,000 phrases
        if len(phrases) >= 10000:
            break

    # Limit the list to exactly 10,000 phrases
    phrases = phrases[:10000]
    return phrases

def create_table(conn):
    table_creation_query = '''
    CREATE TABLE IF NOT EXISTS bookcorpus (
        id SERIAL PRIMARY KEY,
        sentence TEXT
    )
    '''

    try:
        # Crear un cursor y ejecutar la consulta
        with conn.cursor() as cursor:
            start_time = time.time()  # Iniciar cronómetro
            cursor.execute(table_creation_query)
            end_time = time.time()  # Detener cronómetro
            elapsed_time = end_time - start_time
            conn.commit()
            print("Table 'bookcorpus' created successfully.")
            print(f"Time taken for table creation: {elapsed_time:.6f} seconds")
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error creating table: {error}")

def insert_data(conn, data):
    insert_query = """
    INSERT INTO bookcorpus (sentence)
    VALUES (%s);
    """

    try:
        # Create a cursor and execute the batch insertion
        with conn.cursor() as cursor:
            start_time = time.time()  
            cursor.executemany(insert_query, data)
            end_time = time.time()  
            elapsed_time = end_time - start_time

            conn.commit()

            print("Sentences from the bookCorpus inserted successfully!")
            print(f"Time taken for executemany: {elapsed_time:.6f} seconds")

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error inserting data: {error}")

if __name__ == '__main__':
    config = load_config()
    conn = connect(config)

    if conn:
        print()

        data = download_dataset()

        # Create the bookcorpus table to store the sentences
        create_table(conn)
        print()

        # Add the sentences in batch form (all at once)
        insert_data(conn, data)
        print()

        conn.close()
        print("Connection closed.")