import psycopg2
import numpy as np
import sys
import time
import pandas as pd

if len(sys.argv) != 7:
    print("Usage: python test_pgvector.py <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k> <ground_truth_file>")
    sys.exit(1)

conn = psycopg2.connect(
    dbname="postgres"
)
cur = conn.cursor()

str_file = sys.argv[1]
vec_file = sys.argv[2]
vectors = []
strings = []
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

# Execute queries
str_query_file = sys.argv[3]
vec_query_file = sys.argv[4]
k_query_file = sys.argv[5]
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

ground_truth_file = sys.argv[6]
ground_truth = []
with open(ground_truth_file) as gtf:
    for line in gtf:
        ids = [int(x) for x in line.strip().split()]
        ground_truth.append(ids)

print("Executing queries...")
recall = []
time_taken = []
ef_search = 16
while ef_search <= 512:
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
    ef_search *= 2

print("Query done.")
print("Write statistics to file...")
df = pd.DataFrame({
    'ef_search': [16, 32, 64, 128, 256, 512],
    'time_us': time_taken,
    'recall': recall
})
df.to_csv("pgvector_hnsw_stats.csv", index=False)
print("Done.")