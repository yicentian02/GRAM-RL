import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import FlagAutoModel
from trl import GRPOTrainer, GRPOConfig, ModelConfig, TrlParser
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel
import ast


# ==============================
# 1. Load pre-computed corpus embeddings
# ==============================
def load_corpus_embeddings(corpus_file):
    """
    Load precomputed document embeddings from a JSON file.

    Args:
        corpus_file: Path to the JSON file containing document embeddings.

    Returns:
        A dictionary mapping doc_id -> numpy embedding vector.
    """
    with open(corpus_file, "r") as f:
        doc_id_to_embedding = json.load(f)

        # Convert stored lists back to numpy arrays
        return {k: np.array(v) for k, v in doc_id_to_embedding.items()}


# ==============================
# 2. Load the BGE retrieval model
# ==============================
retrieval_model = FlagAutoModel.from_finetuned(
    # "BAAI/bge-base-en-v1.5",
    "bge-base-en-v1.5",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True
)


# ==============================
# 3. Preload all possible corpus embedding files
# ==============================
corpus_embeddings_dict = {
    "query": load_corpus_embeddings("query_corpus_embeddings.json"),
    "summary": load_corpus_embeddings("summary_corpus_embeddings.json"),
    "keyword": load_corpus_embeddings("keyword_corpus_embeddings.json"),
}


def get_completion_content(completion: dict) -> str:
    """
    Extract the text content from the model completion.

    Args:
        completion: Model output dictionary.

    Returns:
        Generated text content.
    """
    return completion[0]["content"]


def parse_responses(completions: list[dict]) -> list[dict]:
    """
    Parse multiple model outputs.

    Args:
        completions: List of model-generated responses.

    Returns:
        Parsed responses containing the generated text.
    """
    return [parse_reasoning_response(get_completion_content(c)) for c in completions]


def parse_reasoning_response(text: str) -> dict:
    """
    Parse a single generated response.

    Args:
        text: Generated text.

    Returns:
        A dictionary containing the parsed response.
    """
    return {"response": text}


# ==============================
# 4. Pairwise retrieval reward function
# ==============================
def pairwise_reward(prompts, question, completions, positive, negative, key, **kwargs):
    """
    Compute reward using pairwise similarity between generated identifiers
    and corpus document embeddings.

    Reward is calculated based on:
        avg_sim(positive_docs) - avg_sim(negative_docs)

    Args:
        prompts: Input prompts
        question: Original query
        completions: Model generated outputs
        positive: Positive document IDs
        negative: Negative document IDs
        key: Identifier type (query / summary / keyword)

    Returns:
        Reward scores for each completion
    """

    parsed_responses = parse_responses(completions)

    # Combine the original question with the generated response
    responses = [question[0] + p["response"] for p in parsed_responses]

    # Encode responses using the BGE model
    with torch.no_grad():
        response_embeddings = retrieval_model.encode(responses)

    # Parse positive and negative document lists
    parsed_positive = [ast.literal_eval(item) for item in positive]
    parsed_negative = [ast.literal_eval(item) for item in negative]

    positive = parsed_positive[0]
    negative = parsed_negative[0]

    corpus_embeddings = corpus_embeddings_dict[key[0]]

    pos_vecs = [corpus_embeddings[pid] for pid in positive if pid in corpus_embeddings]
    neg_vecs = [corpus_embeddings[nid] for nid in negative if nid in corpus_embeddings]

    rewards = []

    margin = 0.2          # Margin threshold for reward
    scale_factor = 100.0  # Scaling factor for reward magnitude

    for emb in response_embeddings:

        if not pos_vecs or not neg_vecs:
            rewards.append(0.0)
            continue

        # Average similarity with positive documents
        avg_pos_sim = np.mean([emb @ vec for vec in pos_vecs])

        # Average similarity with negative documents
        avg_neg_sim = np.mean([emb @ vec for vec in neg_vecs])

        # Compute similarity difference
        raw_diff = avg_pos_sim - avg_neg_sim

        # Reward scaling
        reward = scale_factor * (raw_diff)

        rewards.append(float(reward))

    return rewards


# ==============================
# 5. Load training dataset
# ==============================
def load_data(split="train") -> Dataset:
    """
    Load GRPO training data.

    The dataset is loaded from a CSV file and converted into the
    format required by TRL.

    Returns:
        HuggingFace Dataset object
    """

    data = load_dataset(
        "csv",
        data_files="nq_train.csv"
    )

    data = data[split]

    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": ""},
                {"role": "user", "content": x["prompt"]},
            ],
            "question": x["question"],
            "positive": x["positive"],
            "negative": x["negative"],
            "key": x["key"],
        }
    )

    return data


# ==============================
# 6. GRPO training pipeline
# ==============================
def main(training_args, model_args):

    # Load training dataset
    data = load_data()

    # Load base language model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        base_model_name_or_path=model_args.model_name_or_path,
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    # Apply LoRA to the base model
    lora_model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=lora_model,
        processing_class=tokenizer,
        reward_funcs=pairwise_reward,
        args=training_args,
        train_dataset=data,
    )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model(training_args.output_dir)


# ==============================
# 7. Entry point
# ==============================
if __name__ == "__main__":

    parser = TrlParser((GRPOConfig, ModelConfig))

    training_args, model_args = parser.parse_args_and_config()

    main(training_args, model_args)