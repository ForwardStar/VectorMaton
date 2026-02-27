import random
import os
import sys

def generate_queries(string_list, vector_size, s, num_queries, mixed_length=False):
    """
    Generate queries: (substring of length s, random vector).

    Args:
        string_list (list[str]): List of original strings.
        vector_size (int): Size of the original vectors.
        s (int): Desired substring length.
        num_queries (int): Number of queries to generate.
        mixed_length (bool): If true, query length is uniformly sampled from 2 to 4.

    Returns:
        list[tuple[str, list[float]]]: List of queries.
    """
    queries = []
    for _ in range(num_queries):
        curr_s = s
        min_s = s
        if mixed_length:
            min_s = 2
            curr_s = random.randint(2, 4)

        # pick a base string that is at least length min_s
        candidates = [st for st in string_list if len(st) >= min_s]
        if not candidates:
            raise ValueError(f"No strings long enough for substring length {min_s}.")
        
        base_string = random.choice(candidates)
        if mixed_length:
            curr_s = random.randint(2, min(4, len(base_string)))
        start = random.randint(0, len(base_string) - curr_s)
        substring = base_string[start:start+curr_s]

        # generate a random vector
        random_vector = [random.random() for _ in range(vector_size)]

        queries.append((substring, random_vector))
    
    return queries

# Example usage:
if __name__ == "__main__":
    # Sample string list and vector size
    string_list = ["cat", "dog", "bird", "fish", "lion", "tiger"]
    vector_size = 128
    s = 2
    num_queries = 1000
    k = 10

    mixed_length = "--mixed-length" in sys.argv
    argv = [arg for arg in sys.argv[1:] if arg != "--mixed-length"]

    if len(argv) != 0:
        string_file_path = argv[0]
        vector_file_path = argv[1]
        s = int(argv[2])
        num_queries = int(argv[3])
        k = int(argv[4])
        truncate_len = int(argv[5])
        if not os.path.exists(vector_file_path) or not os.path.exists(string_file_path):
            print("Vector or string file not found.")
            exit(1)
        with open(vector_file_path, "r") as vf:
            vector_size = len(vf.readline().strip().split())
        with open(string_file_path, "r") as sf:
            string_list = [line.strip() for line in sf.readlines()]
        if truncate_len != -1:
            string_list = string_list[:truncate_len]
    else:
        # Read datasets info
        if not os.path.exists("datasets"):
            print("Datasets directory not found. Please run generate_datasets.py first.")
            exit(1)
        
        # Read datasets name
        dataset_names = os.listdir("datasets")
        print("Available datasets:")
        i = 0
        for name in dataset_names:
            print(f"{i}: {name}")
            i += 1

        # Read the dataset to be chosen
        dataset_index = int(input("Enter the index of the dataset to use: "))
        if dataset_index < 0 or dataset_index >= len(dataset_names):
            print("Invalid index.")
            exit(1)
        dataset_name = dataset_names[dataset_index]

        # Read vectors and strings
        vector_file_path = os.path.join("datasets", dataset_name, "vectors.txt")
        string_file_path = os.path.join("datasets", dataset_name, "strings.txt")
        if not os.path.exists(vector_file_path) or not os.path.exists(string_file_path):
            print("Vectors or strings file not found in the selected dataset.")
            exit(1)
        with open(vector_file_path, "r") as vf:
            vector_size = len(vf.readline().strip().split())
        with open(string_file_path, "r") as sf:
            string_list = [line.strip() for line in sf.readlines()]
        
        # Generate queries
        s = int(input("Enter the desired string length for queries: "))
        num_queries = int(input("Enter the number of queries to generate: "))
        k = int(input("Enter value k for k-NN search: "))
        truncate_len = int(input("Enter the number of elements you want to select from the dataset (-1 for all): "))
        if truncate_len != -1:
            string_list = string_list[:truncate_len]

    result = generate_queries(string_list, vector_size, s, num_queries, mixed_length=mixed_length)

    suffix = ""
    if len(argv) > 6:
        suffix = "_" + argv[6]
    # Print the generated queries to "strings.txt", "vectors.txt", and "k.txt"
    with open(f"strings{suffix}.txt", "w") as sf, open(f"vectors{suffix}.txt", "w") as vf, open(f"k{suffix}.txt", "w") as kf:
        for (st, vec) in result:
            sf.write(st + "\n")
            vf.write(" ".join(map(str, vec)) + "\n")
            kf.write(str(k) + "\n")
