import psycopg2
import numpy as np
import sys
import time
import pandas as pd

if len(sys.argv) != 12:
    print("Usage: python test_pgvector.py <string_data_file> <vector_data_file> <string_query_file> <vector_query_file> <k> <dbname> <user> <host> <port> <ground_truth_file>")
    sys.exit(1)

conn = psycopg2.connect(
    dbname=sys.argv[6],
    user=sys.argv[7],
    host=sys.argv[8],
    port=int(sys.argv[9])
)
cur = conn.cursor()

str_file = sys.argv[1]
vec_file = sys.argv[2]
vectors = []
strings = []
n = 0

print("Loading data...")

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

batch = [(vectors[i], strings[i].strip()) for i in range(n)]
cur.executemany(
    "INSERT INTO items (embedding, text) VALUES (%s, %s)",
    batch
)
conn.commit()

cur.close()
conn.close()

print("Done loading data.")

print("Building HNSW index...")
cur.execute("CREATE INDEX idx_vec_hnsw IF NOT EXISTS ON items hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 200);")
conn.commit()

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

ground_truth_file = sys.argv[10]
ground_truth = []
with open(ground_truth_file) as gtf:
    for line in gtf:
        ids = [int(x) for x in line.strip().split()]
        ground_truth.append(ids)

print("Executing queries...")
recall = []
time_taken = []
for ef_search in range(20, 401, 20):
    print(f"ef_search={ef_search}")
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    total_time = 0.0
    total_recall = 0.0
    effective = 0
    for i, (qvec, qstr, qk) in enumerate(zip(vectors, strings, k)):
        start = time.time()

        # PostgreSQL query: substring filter + vector distance
        cur.execute(
            """
            SELECT id, embedding <-> %s AS dist
            FROM items
            WHERE text LIKE %s
            ORDER BY embedding <-> %s
            LIMIT %s
            """,
            (qvec, f"%{qstr}%", qvec, qk)
        )

        rows = cur.fetchall()
        elapsed = time.time() - start
        total_time += elapsed

        neighbor_ids = [row[0] for row in rows]

        true_ids = set(ground_truth[i])
        if len(true_ids) > 0:
            recall = len(set(neighbor_ids) & true_ids) / len(true_ids)
            effective += 1
            total_recall += recall

    avg_time = total_time / n * 1e6  # convert to microseconds
    avg_recall = total_recall / effective
    time_taken.append(avg_time)
    recall.append(avg_recall)
    print(f"Average Time: {avg_time} us, Average Recall: {avg_recall:.4f}")

print("Query done.")
print("Write statistics to file...")
df = pd.DataFrame({
    'ef_search': list(range(20, 401, 20)),
    'time_us': time_taken,
    'recall': recall
})
df.to_csv("pgvector_hnsw_stats.csv", index=False)
print("Done.")