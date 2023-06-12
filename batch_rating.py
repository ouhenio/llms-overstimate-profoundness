import os
import guidance
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time
import traceback


from prompts import EVALUATION_PROMPTS_DICT

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

OPENAI_MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TEMPERATURES = {0: 0.1, 1: 0.7}

def load_batch(path: str) -> list:
    df = pd.read_csv(path)
    batch = df.to_dict(orient='records')

    return batch

def evaluate_sentence(
    evaluation_prompt: str,
    sentence: str,
    temperature: int,
    model: str = "gpt-3.5",
) -> int:
    gpt = guidance.llms.OpenAI(OPENAI_MODELS[model], caching=False)
    evaluate_sentence_ = guidance(
        """
    {{#user~}}
    {{evaluation_prompt}}

    Provide your rating as a simple numerical value, without any accompanying text.

    Sentence:
    ----
    {{sentence}}


    {{~/user}}

    {{#assistant~}}
    {{gen 'score' temperature=temperature max_tokens=4}}
    {{~/assistant}}
    """,
        llm=gpt,
    )

    score = int(
        evaluate_sentence_(
            evaluation_prompt=evaluation_prompt,
            sentence=sentence,
            temperature=temperature,
        )["score"]
    )

    return score

if __name__ == "__main__":
    delay_seconds = 5

    batch = load_batch("second_batch_failed_calls.csv")
    batch_results = []
    failed_calls = []
    for element in tqdm(batch):
        try:
            score = evaluate_sentence(
                evaluation_prompt=EVALUATION_PROMPTS_DICT[element["evaluation_prompt"]],
                sentence=element["sentence"],
                temperature=TEMPERATURES[element["temperature"]],
                model="gpt-4"
            )
            rated_element = element.copy()
            rated_element["likert_score"] = int(score)
            batch_results.append(rated_element)
        except Exception as e:
            print(f"Exception encountered: {e}")
            print(traceback.format_exc())
            failed_calls.append(element)
            print("saving progress...")
            rated_df = pd.DataFrame(batch_results)
            rated_df.to_csv("second_batch_results_2.csv", index=False)
            not_rated_df = pd.DataFrame(failed_calls)
            not_rated_df.to_csv("second_batch_failed_calls_2.csv", index=False)
            print(f"Returning in {delay_seconds} seconds...")
            time.sleep(delay_seconds)
    rated_df = pd.DataFrame(batch_results)
    rated_df.to_csv("second_batch_results_2.csv", index=False)
    not_rated_df = pd.DataFrame(failed_calls)
    not_rated_df.to_csv("second_batch_failed_calls_2.csv", index=False)
