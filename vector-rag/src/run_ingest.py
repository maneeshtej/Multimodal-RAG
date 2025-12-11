from ingest.pipeline import run_pipeline
import json

def load_config():
    with open("config.json") as f:
        return json.load(f)

if __name__ == "__main__":
    config = load_config()
    pdf_file = config["pdf_input_dir"]   # << changed here
    run_pipeline(pdf_file)
