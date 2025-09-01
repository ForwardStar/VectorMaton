import sys
import datetime
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time

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

log = Logger(level="INFO")  # set threshold level
if not os.path.exists("datasets"):
    os.makedirs("datasets")
if not os.path.exists("datasets/code_search_net"):
    os.makedirs("datasets/code_search_net")
    log.info("Downloading and generating CodeSearchNet dataset...")
    start = time.perf_counter()
    # Load dataset
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    log.info(f"Dataset loaded. Size = {dataset['train'].num_rows} entries.")
    # Convert dataset to required format
    with open("datasets/code_search_net/vectors.txt", "w") as vec_file, open("datasets/code_search_net/strings.txt", "w") as str_file:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        for item in dataset['train']:
            s = item['func_name']
            code = item['func_code_string']
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                v = model(**inputs).last_hidden_state[:,0,:].squeeze(0).numpy()
            str_file.write(s + "\n")
            vec_file.write(" ".join(map(str, v)) + "\n")
    log.info("CodeSearchNet dataset downloaded and processed!")
    end = time.perf_counter()
    elapsed = end - start
    log.info(f"Time consumption: {format_time(elapsed)}")
else:
    log.info("CodeSearchNet dataset already exists. Skipped.")