import sys
import datetime
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
import pandas as pd
import json

headers = {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'}


def _synthetic_alpha_string(seed: int, length: int = 50) -> str:
    # Deterministic pseudo-random lowercase string for vector-only datasets (e.g., SIFT).
    x = ((seed + 1) * 2654435761) & 0xFFFFFFFF
    chars = []
    for _ in range(length):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        chars.append(chr(ord('a') + (x % 26)))
    return "".join(chars)


def _load_fvecs_memmap(path: str):
    with open(path, "rb") as f:
        dim_arr = np.fromfile(f, dtype=np.int32, count=1)
        if dim_arr.size == 0:
            raise ValueError(f"Empty fvecs file: {path}")
        dim = int(dim_arr[0])

    record_bytes = 4 * (dim + 1)
    file_size = os.path.getsize(path)
    if file_size % record_bytes != 0:
        raise ValueError(f"Invalid fvecs file size for {path}.")

    nvecs = file_size // record_bytes
    data = np.memmap(path, dtype=np.float32, mode="r", shape=(nvecs, dim + 1))
    dims = data[:, 0].view(np.int32)
    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent dimensions in fvecs file: {path}")
    return data[:, 1:], dim, nvecs


def _find_existing_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the expected files were found: {candidates}")

def download(url, path):
    try:
        from pathlib import Path
        from tqdm import tqdm
    except:
        print("Installing dependencies...")
        from pip._internal import main
        main(['install', 'pathlib'])
        main(['install', 'tqdm'])
        from pathlib import Path
        from tqdm import tqdm
    from urllib.request import urlopen, Request
    print("Fetching from", url + "...")
    path = Path(path)
    blocksize = 1024 * 8
    blocknum = 0
    retry_times = 0
    while True:
        try:
            with urlopen(Request(url, headers=headers), timeout=3) as resp:
                total = resp.info().get("content-length", None)
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    unit_divisor=1024,
                    total=total if total is None else int(total),
                ) as t, path.open("wb") as f:
                    block = resp.read(blocksize)
                    while block:
                        f.write(block)
                        blocknum += 1
                        t.update(len(block))
                        block = resp.read(blocksize)
            break
        except KeyboardInterrupt:
            if path.is_file():
                path.unlink()
            raise
        except:
            retry_times += 1
            if retry_times >= 20:
                break
            print("Timed out, retrying...")
    if retry_times >= 20:
        if path.is_file():
            path.unlink()
        raise ConnectionError("bad internet connection, check it and retry.")
    
class Logger:
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40
    }
    
    def __init__(self, level="INFO", stream=sys.stdout):
        self.level = self.LEVELS.get(level.upper(), 20)  # Default INFO
        self.stream = stream

    def _log(self, level_name, message):
        if self.LEVELS[level_name] >= self.level:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.stream.write(f"[{timestamp}] {level_name}: {message}\n")
            self.stream.flush()

    def debug(self, message):
        self._log("DEBUG", message)

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

def format_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:  # 1 hour
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h {m}min {s}s"
    elif seconds >= 60:  # 1 minute
        m = seconds // 60
        s = seconds % 60
        return f"{m}min {s}s"
    else:
        return f"{seconds}s"

if __name__ == "__main__":
    log = Logger(level="INFO")  # set threshold level
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    # Spam
    if not os.path.exists("datasets/spam"):
        os.makedirs("datasets/spam")
        log.info("Downloading and generating spam dataset...")
        start = time.perf_counter()
        # Load dataset
        import tarfile
        download("https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2", "datasets/spam/20021010_spam.tar.bz2")
        archive = tarfile.open("datasets/spam/20021010_spam.tar.bz2", "r:bz2")
        archive.extractall("datasets/spam")
        archive.close()
        import re

        from email import policy
        from email.parser import BytesParser
        
        def parse_email(file_path):
            with open(file_path, 'rb') as f:
                # Use default policy to get EmailMessage if possible
                try:
                    msg = BytesParser(policy=policy.default).parse(f)
                except Exception:
                    # fallback for older Message objects
                    msg = BytesParser(policy=policy.compat32).parse(f)

            # Extract title/subject
            title = msg['subject'] or "No Subject"

            # Extract body content
            content = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            try:
                                content += payload.decode(charset, errors='replace')
                            except LookupError:
                                content += payload.decode('utf-8', errors='replace')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    try:
                        content = payload.decode(charset, errors='replace')
                    except LookupError:
                        content = payload.decode('utf-8', errors='replace')

            # Clean whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            title = title.strip()

            return title, content
        
        def clean_content(text):
            # Remove quoted replies and non-alphanumeric
            text = re.sub(r">.*", "", text)
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.lower().strip()
        
        from sentence_transformers import SentenceTransformer

        # Load model
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Process
        with open("datasets/spam/strings.txt", "w") as str_file, open("datasets/spam/vectors.txt", "w") as vec_file:
            for file_path in os.listdir("datasets/spam/spam"):
                title, content = parse_email(os.path.join("datasets/spam/spam", file_path))
                clean_body = clean_content(content)
                vector = embed_model.encode(clean_body).tolist()  # convert to list for JSON
                title = ''.join(c for c in title if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                title = title.lower()
                if title != "":
                    str_file.write(title + "\n")
                    vec_file.write(" ".join(map(str, vector)) + "\n")
    else:
        log.info("Spam dataset already exists. Skipped.")

    # Words
    if not os.path.exists("datasets/words"):
        os.makedirs("datasets/words")
        log.info("Downloading and generating Words dataset...")
        start = time.perf_counter()
        # Load dataset
        download("https://huggingface.co/datasets/efarrall/word_embeddings/resolve/main/8000words_large.pkl?download=true", "datasets/words/8000words_large.pkl")
        download("https://huggingface.co/datasets/efarrall/word_embeddings/resolve/main/word_list?download=true", "datasets/words/word_list.txt")
        # Convert dataset to required format
        df_word_embeds = pd.read_pickle("datasets/words/8000words_large.pkl")
        words = json.load(open("datasets/words/word_list.txt", "r"))
        with open("datasets/words/vectors.txt", "w") as vec_file, open("datasets/words/strings.txt", "w") as str_file:
            for i, word in enumerate(words):
                v = df_word_embeds.iloc[i]
                word = ''.join(c for c in word if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                word = word.lower()
                str_file.write(word + "\n")
                vec_file.write(" ".join(map(str, v)) + "\n")
        log.info("Words dataset downloaded and processed!")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("Words dataset already exists. Skipped.")

    # MTG
    if not os.path.exists("datasets/mtg"):
        os.makedirs("datasets/mtg")
        log.info("Downloading and generating MTG dataset...")
        start = time.perf_counter()
        # Load dataset
        dataset = load_dataset("TrevorJS/mtg-scryfall-cropped-art-embeddings-open-clip-ViT-SO400M-14-SigLIP-384")
        log.info(f"Dataset loaded. Size = {dataset['train'].num_rows} entries.")
        with open("datasets/mtg/vectors.txt", "w") as vec_file, open("datasets/mtg/strings.txt", "w") as str_file:
            for item in dataset['train']:
                s = item['flavor_text']
                if s is not None:
                    # Remove all non-alphabet symbols
                    s = ''.join(c for c in s if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                    # Convert to lowercase
                    s = s.lower()
                    v = item['open_clip_image_embeddings']
                    if s != "":
                        str_file.write(s + "\n")
                        vec_file.write(" ".join(map(str, v)) + "\n")
        log.info("MTG dataset downloaded and processed!")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("MTG dataset already exists. Skipped.")

    # CodeSearchNet
    if not os.path.exists("datasets/code_search_net"):
        os.makedirs("datasets/code_search_net")
        log.info("Downloading and generating CodeSearchNet dataset...")
        start = time.perf_counter()
        # Load dataset
        dataset = load_dataset("code_search_net", trust_remote_code=True)
        log.info(f"Dataset loaded. Size = {dataset['train'].num_rows} entries.")
        # Convert dataset to required format
        with open("datasets/code_search_net/vectors.txt", "w") as vec_file, open("datasets/code_search_net/strings.txt", "w") as str_file:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
            for item in dataset['train']:
                s = item['func_name']
                # Remove all non-alphabet symbols
                s = ''.join(c for c in s if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                # Convert to lowercase
                s = s.lower()
                code = item['func_code_string']
                inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    v = model(**inputs).last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
                if s != "":
                    str_file.write(s + "\n")
                    vec_file.write(" ".join(map(str, v)) + "\n")
        log.info("CodeSearchNet dataset downloaded and processed!")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("CodeSearchNet dataset already exists. Skipped.")
    
    # SwissProt
    if not os.path.exists("datasets/swissprot"):
        os.makedirs("datasets/swissprot")
        log.info("Downloading and generating SwissProt dataset...")
        start = time.perf_counter()
        # Load dataset
        dataset = load_dataset("khairi/uniprot-swissprot", trust_remote_code=True)
        log.info(f"Dataset loaded. Size = {dataset['train'].num_rows} entries.")
        with open("datasets/swissprot/vectors.txt", "w") as vec_file, open("datasets/swissprot/strings.txt", "w") as str_file:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            model = AutoModel.from_pretrained("Rostlab/prot_bert").to(device)
            for item in dataset['train']:
                s = item['Sequence']
                # Remove all non-alphabet symbols
                s = ''.join(c for c in s if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                # Convert to lowercase
                s = s.lower()
                seq = " ".join(list(item['Sequence']))
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    v = model(**inputs).last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
                if s != "":
                    vec_file.write(" ".join(map(str, v)) + "\n")
                    str_file.write(s + "\n")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("SwissProt dataset already exists. Skipped.")

    # ArXiv
    if not os.path.exists("datasets/arxiv"):
        os.makedirs("datasets/arxiv")
        log.info("Downloading and generating ArXiv dataset...")
        start = time.perf_counter()
        # Load dataset
        dataset = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=False)
        log.info(f"Dataset loaded. Size = {dataset.num_rows} entries.")
        with open("datasets/arxiv/vectors.txt", "w") as vec_file, open("datasets/arxiv/strings.txt", "w") as str_file:
            for item in dataset:
                s = item['title'].replace("\n", " ")
                # Remove all non-alphabet symbols
                s = ''.join(c for c in s if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                # Convert to lowercase
                s = s.lower()
                v = item['vector']
                if s != "":
                    str_file.write(s + "\n")
                    vec_file.write(" ".join(map(str, v)) + "\n")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("ArXiv dataset already exists. Skipped.")
    
    # Arxiv-small
    if not os.path.exists("datasets/arxiv-small"):
        os.makedirs("datasets/arxiv-small")
        log.info("Downloading and generating ArXiv-small dataset...")
        start = time.perf_counter()
        # Load dataset
        dataset = load_dataset("malteos/aspect-paper-embeddings", split="train", streaming=False)
        log.info(f"Dataset loaded. Size = {dataset.num_rows} entries.")
        with open("datasets/arxiv-small/vectors.txt", "w") as vec_file, open("datasets/arxiv-small/strings.txt", "w") as str_file:
            for item in dataset:
                s = item['title'].replace("\n", " ")
                # Remove all non-alphabet symbols
                s = ''.join(c for c in s if (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z'))
                # Convert to lowercase
                s = s.lower()
                v = item['dataset_embeddings']
                if s != "":
                    str_file.write(s + "\n")
                    vec_file.write(" ".join(map(str, v)) + "\n")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("ArXiv-small dataset already exists. Skipped.")

    # SIFT (TEXMEX)
    sift_vectors_path = "datasets/sift/vectors.txt"
    sift_strings_path = "datasets/sift/strings.txt"
    if not (os.path.exists(sift_vectors_path) and os.path.exists(sift_strings_path)):
        os.makedirs("datasets/sift", exist_ok=True)
        log.info("Downloading and generating SIFT dataset...")
        start = time.perf_counter()
        import tarfile

        sift_archive = "datasets/sift/sift.tar.gz"
        if not os.path.exists(sift_archive):
            try:
                download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", sift_archive)
            except Exception:
                log.warning("FTP download failed, retrying with HTTP mirror.")
                download("http://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", sift_archive)

        with tarfile.open(sift_archive, "r:gz") as archive:
            archive.extractall("datasets/sift")

        sift_base_fvecs = _find_existing_file([
            "datasets/sift/sift_base.fvecs",
            "datasets/sift/sift/sift_base.fvecs",
        ])
        vectors, dim, nvecs = _load_fvecs_memmap(sift_base_fvecs)
        log.info(f"SIFT base vectors loaded. Size = {nvecs} entries, dim = {dim}.")

        with open(sift_vectors_path, "w") as vec_file, open(sift_strings_path, "w") as str_file:
            for i in range(nvecs):
                str_file.write(_synthetic_alpha_string(i) + "\n")
                vec_file.write(" ".join(map(str, vectors[i])) + "\n")

        log.info("SIFT dataset downloaded and processed!")
        end = time.perf_counter()
        elapsed = end - start
        log.info(f"Time consumption: {format_time(elapsed)}")
    else:
        log.info("SIFT dataset already exists. Skipped.")
