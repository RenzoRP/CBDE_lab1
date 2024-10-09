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
    CREATE TABLE IF NOT EXISTS log (
        selected_id INT,
        target_id INT,
        euclidean_distance FLOAT,
        manhattan_distance FLOAT,
        euclidean_execution_time FLOAT,
        manhattan_execution_time FLOAT,
        PRIMARY KEY (selected_id, target_id)
    )
    '''

    try:
        with conn.cursor() as cursor:
            cursor.execute(table_creation_query)

            conn.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error creating table: {error}")
    
def get_data(query, data):
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, data)
            data = cursor.fetchall()

            return data

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error getting data: {error}")
        return None

def get_data_by_id_list(ids):
    placeholders = ', '.join(['%s'] * len(ids))
    query = f'''
        SELECT id, embedding FROM embeddings WHERE id IN ({placeholders});
    '''
    return get_data(query, tuple(ids))

def get_multiple_data_by_id_range(id1, id2):
    query = '''
        SELECT id, embedding FROM embeddings WHERE id >= %s and id <= %s;
    '''
    return get_data(query, (id1, id2))

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def calculate_and_save_distances(selected_lines, target_lines, batch_time):
    log = []
    recover_time = batch_time/len(target_lines) if target_lines else 0

    for selected in selected_lines:
        selected_id, selected_vec = selected[0], np.array(selected[1])
        for target in target_lines:
            target_id, target_vec = target[0], np.array(target[1])
            if selected_id == target_id:
                continue  # Saltar el cálculo si los IDs son iguales
            if len(selected_vec) != len(target_vec):
                raise ValueError(f"Los vectores deben tener la misma longitud (ID {selected_id} vs {target_id})")
            start_time = time.perf_counter()
            eucl_dist = euclidean_distance(selected_vec, target_vec)
            eucl_time = time.perf_counter() - start_time  + recover_time
            
            start_time = time.perf_counter()
            manh_dist = manhattan_distance(selected_vec, target_vec)
            manh_time = time.perf_counter() - start_time  + recover_time

            log.append((int(selected_id), int(target_id), float(eucl_dist), float(manh_dist), float(eucl_time), float(manh_time)))
    
    insert_data(log)

def insert_data(data):
    insert_query = """
    INSERT INTO log (selected_id, target_id, euclidean_distance, manhattan_distance, euclidean_execution_time, manhattan_execution_time)
    VALUES (%s, %s, %s, %s, %s, %s);
    """

    try:
        with conn.cursor() as cursor:
            cursor.executemany(insert_query, data)
            conn.commit()

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error inserting data: {error}")

def print_similar_sentences(similar_sentences):
    for sentence in similar_sentences:
        print(f"Sentence:{sentence[1]}, Distance:{sentence[2]}")

    

def get_similar_sentences(ids):
    manhattan_query ='''
        SELECT e2.sentence AS selected_sentence, e1.sentence AS target_sentence, l.manhattan_distance
        FROM log l
        JOIN bookcorpus e1 ON l.target_id = e1.id
        JOIN bookcorpus e2 ON l.selected_id = e2.id
        WHERE l.selected_id = %s
        ORDER BY l.manhattan_distance ASC
        LIMIT 2;
    '''
    euclidean_query ='''
        SELECT e2.sentence AS selected_sentence, e1.sentence AS target_sentence, l.manhattan_distance
        FROM log l
        JOIN bookcorpus e1 ON l.target_id = e1.id
        JOIN bookcorpus e2 ON l.selected_id = e2.id
        WHERE l.selected_id = %s
        ORDER BY l.euclidean_distance ASC
        LIMIT 2;
    '''
    print("Manhattan distance -->")
    start_time = time.perf_counter()
    for id in ids:
        print()
        manhattan_similar_sentences = get_data(manhattan_query, (id, ))
        selected_sentence = manhattan_similar_sentences[0][0]
        print(f"Similar sentences to '{selected_sentence}' ({id}):")
        print_similar_sentences(manhattan_similar_sentences)
    manh_time = time.perf_counter() - start_time

    print()
    print("Euclidean distance -->")
    start_time = time.perf_counter()
    for id in ids:
        print()
        euclidean_similar_sentences = get_data(euclidean_query, (id, ))
        selected_sentence = euclidean_similar_sentences[0][0]
        print(f"Similar sentences to '{selected_sentence}' ({id}):")
        print_similar_sentences(euclidean_similar_sentences)
    eucl_time = time.perf_counter() - start_time

    return manh_time, eucl_time

def get_times(manh_time, eucl_time):
    manhattan_query ='''
        SELECT manhattan_execution_time FROM log
    '''
    euclidean_query ='''
        SELECT euclidean_execution_time FROM log
    '''
    manhattan_times = get_data(manhattan_query, None)
    manhattan_times = [time[0] + manh_time for time in manhattan_times]

    euclidean_times = get_data(euclidean_query, None)
    euclidean_times = [time[0] + eucl_time for time in euclidean_times]

    print("Manhattan times")
    print(f"Minimum time: {np.min(manhattan_times):.4f} seconds")
    print(f"Maximum time: {np.max(manhattan_times):.4f} seconds")
    print(f"Average time: {np.mean(manhattan_times):.4f} seconds")
    print(f"Standard deviation: {np.std(manhattan_times):.4f} seconds")
    print()
    print("Euclidean times")
    print(f"Minimum time: {np.min(euclidean_times):.4f} seconds")
    print(f"Maximum time: {np.max(euclidean_times):.4f} seconds")
    print(f"Average time: {np.mean(euclidean_times):.4f} seconds")
    print(f"Standard deviation: {np.std(euclidean_times):.4f} seconds")
    

if __name__ == '__main__':
    config = load_config()
    conn = connect(config)

    if conn:
        # Create table logs in order to store distances and times
        create_table(conn)

        ids = [937, 1722, 2535, 2544, 2893, 2952, 5701, 7832, 7931, 8519]
        selected_lines = get_data_by_id_list(ids)

        if selected_lines:
            start_id, batch_size, max_id = 1, 500, 10000
            for current_start_id in range(start_id, max_id + 1, batch_size):
                current_end_id = current_start_id + batch_size - 1
                start_time = time.perf_counter()
                target_lines = get_multiple_data_by_id_range(current_start_id, current_end_id)
                batch_time = time.perf_counter() - start_time
                if target_lines:
                    calculate_and_save_distances(selected_lines, target_lines, batch_time)
        
        manh_time, eucl_time = get_similar_sentences(ids)
        get_times(manh_time, eucl_time)

        conn.close()
        print("Connection closed.")
