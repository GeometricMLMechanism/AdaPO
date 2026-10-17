import argparse
import os

import datasets


def make_map_fn(split, source=None):
        def process_fn(example, idx):
            if source is None:
                data_source = example.pop("source")
            else:
                data_source = source
            question = example.pop("prompt")
            solution = example.pop("answer")
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                },
            }
            return data

        return process_fn

def convert(data_source):
    """Convert <data_source>/{train,test}.json into the parquet files read by verl."""
    for split in ("train", "test"):
        dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, f"{split}.json"), split="train")
        dataset = dataset.map(function=make_map_fn(split, data_source), with_indices=True)
        dataset.to_parquet(os.path.join(data_source, f"{split}.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", nargs="+", default=["AIME-TTT", "AMC-TTT", "MATH-TTT", "GPQA-TTT"])
    args = parser.parse_args()
    for data_source in args.data_source:
        convert(data_source)
