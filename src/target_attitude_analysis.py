import os
import pandas as pd
import json
import json_repair
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoliticalSentimentInference:
    def __init__(self, model_path, gpu_device="0", batch_size=8):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

        logger.info(f"Loading model from {model_path}")
        self.llm = LLM(model=model_path,max_model_len=8096, tensor_parallel_size=2, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

        self.batch_size = batch_size

    def prepare_prompt(self, text):
        prompt = (
    "For the following post, identify all political figures or entities discussed and the attitude expressed toward each. "
    "Only consider three possible targets: 'trump', 'biden', or 'other'. For each target, determine whether the post expresses a "
    "'positive', 'negative', or 'neutral' attitude.\n\n"

    "Instructions:\n"
    "- Resolve indirect or implicit references, including pronouns (e.g., 'he', 'our president'), nicknames, emojis, or known context.\n"
    "- Sarcasm, irony, or emotionally charged language should be interpreted for actual sentiment, not literal phrasing.\n"
    "- If a post mentions a 'president's birthday' or celebrates 'our favorite president' in June, it should be interpreted as referring to Donald Trump, whose birthday is June 14.\n"
    "- Do not assume sentiment is positive just because a figure is mentioned in a favorable context (e.g., 'trying to protect Biden' in a sentence full of anger is likely still anti-Biden).\n"
    "- A single post may express opinions about multiple figures. Each should be included with its own sentiment.\n\n"

    "Output format: Return a JSON list of objects like: "
    "[{\"target\": \"biden\", \"attitude\": \"negative\"}, {\"target\": \"trump\", \"attitude\": \"positive\"}]\n\n"

    f"Post:\n{text}"
)


        messages = [{"role": "user", "content": prompt}]
        return messages

    def predict_batch(self, texts):
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                self.prepare_prompt(text),
                add_generation_prompt=True,
                tokenize=False
            )
            for text in texts
        ]

        outputs = self.llm.generate(formatted_prompts, sampling_params=self.sampling_params)

        predictions = []
        for output in outputs:
            try:
                text = output.outputs[0].text.strip()
                prediction = json_repair.loads(text)
            except Exception as e:
                logger.warning(f"Failed to parse output: {output.outputs[0].text.strip()}")
                prediction = text
            predictions.append(prediction)

        return predictions

    def run(self, input_file, output_file):
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return

        df = pd.read_csv(input_file)
        if 'post_full_content' not in df.columns:
            logger.error(f"'post_full_content' column not found in {input_file}")
            return

        texts = df['post_full_content'].dropna().astype(str).tolist()
        results = []

        logger.info(f"Processing {len(texts)} posts...")

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Batches"):
            batch_texts = texts[i:i+self.batch_size]
            batch_predictions = self.predict_batch(batch_texts)
            results.extend(batch_predictions)

        df['prediction'] = results
        df.to_json(output_file, orient='records', indent=2, force_ascii=False)

        readable_output = output_file.replace('.json', '_readable.txt')
        with open(readable_output, 'w', encoding='utf-8') as f:
            for i, row in df.iterrows():
                f.write(f"Post: {row['post_full_content']}\nPrediction: {row['prediction']}\n\n")

        logger.info(f"Saved predictions to {output_file}")
        logger.info(f"Readable output saved to {readable_output}")

def main():
    INPUT_FILE = "../data/data/narrative_analysis_id_root_post.csv"
    OUTPUT_FILE = "../data/data/narrative_analysis_id_root_post_annotated.csv"
    MODEL_PATH = "google/gemma-3-27b-it"
    GPU_DEVICE = "0,1"
    BATCH_SIZE = 64

    model = PoliticalSentimentInference(
        model_path=MODEL_PATH,
        gpu_device=GPU_DEVICE,
        batch_size=BATCH_SIZE
    )

    model.run(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
