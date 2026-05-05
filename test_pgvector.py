import argparse
import psycopg2
import numpy as np
import sys
import time
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark pgvector HNSW search with substring constraints."
    )
    parser.add_argument("string_data_file")
    parser.add_argument("vector_data_file")
    parser.add_argument("string_query_file")
    parser.add_argument("vector_query_file")
    parser.add_argument("k_file")
    parser.add_argument("ground_truth_file")
    parser.add_argument(
        "--insertion-percentage",
        type=float,
        default=0.0,
        help="Percentage of tail data to insert after building the index.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the table and index.")
    return parser.parse_args()


def split_base_and_insertions(vectors, strings, insertion_percentage):
    if insertion_percentage < 0 or insertion_percentage > 100:
        raise ValueError("--insertion-percentage must be in [0, 100].")

    insertion_count = int(len(vectors) * insertion_percentage / 100)
    base_count = len(vectors) - insertion_count
    if vectors and base_count == 0:
        raise ValueError("--insertion-percentage leaves no base vectors to build the index.")

    return (
        vectors[:base_count],
        strings[:base_count],
        vectors[base_count:],
        strings[base_count:],
    )


args = parse_args()

conn = psycopg2.connect(
    dbname="postgres"
)
cur = conn.cursor()

str_file = args.string_data_file
vec_file = args.vector_data_file
vectors = []
strings = []
insertion_vectors = []
insertion_strings = []
n = 0
table_name = vec_file.replace('.', '_').replace('/', '_').replace('-', '_')

print("Loading data...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("SET temp_buffers = '128GB';")
conn.commit()
cur.execute("SHOW temp_buffers;")
conn.commit()
buff_size = cur.fetchall()
print(f"temp_buffers size set to: {buff_size[0][0]}")

batch = [(vectors[i], strings[i].strip()) for i in range(n)]
table_exists = True
# Create table, insert data and build index if it does not exist
if args.rebuild:
    print(f"Deleting existing index and table for rebuild: {table_name}")
    cur.execute("DROP INDEX IF EXISTS idx_vec_hnsw;")
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    conn.commit()

cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s);", (table_name,))
if not cur.fetchone()[0]:
    with open(vec_file) as vf, open(str_file) as sf:
        vectors = vf.readlines()
        strings = sf.readlines()
        for i in range(len(vectors)):
            vectors[i] = [float(x) for x in vectors[i].split()]
        n = min(len(vectors), len(strings))
        if len(vectors) > n:
            vectors = vectors[:n]
        if len(strings) > n:
            strings = strings[:n]
    vectors, strings, insertion_vectors, insertion_strings = split_base_and_insertions(
        vectors,
        strings,
        args.insertion_percentage,
    )
    if insertion_vectors:
        print(
            f"Using {len(vectors)} base vectors and "
            f"{len(insertion_vectors)} insertion vectors."
        )
    n = len(vectors)
    table_exists = False
    dim = len(vectors[0])

    sql = f"""
    CREATE TEMP TABLE {table_name} (
        id SERIAL PRIMARY KEY,
        embedding VECTOR({dim}),
        text TEXT
    );"""
    cur.execute(sql)
    conn.commit()
    sql = f"""
    INSERT INTO {table_name} (embedding, text) VALUES (%s, %s);"""
    cur.executemany(sql, [(vectors[i], strings[i].strip()) for i in range(n)])
    conn.commit()
else:
    print(f"Table {table_name} already exists. Skipping data insertion.")
print("Done loading data.")

print("Building HNSW index...")
if not table_exists:
    sql = f"""DROP INDEX IF EXISTS idx_vec_hnsw;"""
    cur.execute(sql)
    conn.commit()
    sql = f"""CREATE INDEX idx_vec_hnsw ON {table_name} USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);"""
    cur.execute(sql)
    conn.commit()
else:
    print(f"Index on table {table_name} may already exist. Skipping index creation.")
print("Done building index.")

if insertion_vectors:
    if not table_exists:
        print(f"Inserting additional {len(insertion_vectors)} rows into pgvector table...")
        start = time.time()
        sql = f"""
        INSERT INTO {table_name} (embedding, text) VALUES (%s, %s);"""
        cur.executemany(
            sql,
            [(insertion_vectors[i], insertion_strings[i].strip()) for i in range(len(insertion_vectors))]
        )
        conn.commit()
        print(f"Insertion took {(time.time() - start) * 1e6:.0f} us.")
    else:
        print("Skipping insertion because the pgvector table already exists.")

# Execute queries
str_query_file = args.string_query_file
vec_query_file = args.vector_query_file
k_query_file = args.k_file
vectors = []
strings = []
k = []
n = 0

with open(vec_query_file) as vf, open(str_query_file) as sf, open(k_query_file) as kf:
    vectors = vf.readlines()
    strings = sf.readlines()
    k = kf.readlines()
    for i in range(len(vectors)):
        vectors[i] = [float(x) for x in vectors[i].split()]
    n = min(len(vectors), len(strings), len(k))
    if len(vectors) > n:
        vectors = vectors[:n]
    if len(strings) > n:
        strings = strings[:n]
    if len(k) > n:
        k = k[:n]

ground_truth_file = args.ground_truth_file
ground_truth = []
with open(ground_truth_file) as gtf:
    for line in gtf:
        ids = [int(x) for x in line.strip().split()]
        ground_truth.append(ids)

print("Executing queries...")
recall = []
time_taken = []
ef_search_values = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
for ef_search in ef_search_values:
    print(f"ef_search={ef_search}")
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    total_time = 0.0
    total_recall = 0.0
    effective = 0
    for i, (qvec, qstr, qk) in enumerate(zip(vectors, strings, k)):
        start = time.time()

        # PostgreSQL query: substring filter + vector distance
        sql = f"""
            SELECT id
            FROM {table_name}
            WHERE text LIKE %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        cur.execute(
            sql,
            (f"%{qstr.strip()}%", qvec, int(qk.strip()))
        )

        rows = cur.fetchall()
        elapsed = time.time() - start
        total_time += elapsed

        neighbor_ids = [row[0] - 1 for row in rows]

        true_ids = set(ground_truth[i])
        if len(true_ids) > 0:
            effective += 1
            total_recall += len(set(neighbor_ids) & true_ids) / len(true_ids)

    avg_time = total_time / n * 1e6  # convert to microseconds
    avg_recall = total_recall / effective
    time_taken.append(avg_time)
    recall.append(avg_recall)
    print(f"Average Time: {avg_time} us, Average Recall: {avg_recall:.4f}")

print("Query done.")
print("Write statistics to file...")
df = pd.DataFrame({
    'ef_search': ef_search_values,
    'time_us': time_taken,
    'recall': recall
})
df.to_csv("pgvector_hnsw_stats.csv", index=False)
print("Done.")
