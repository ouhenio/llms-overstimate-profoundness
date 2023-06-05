import json
import os
import random

import fire
import guidance
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from prompts import EVALUATION_PROMPTS_DICT

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

OPENAI_MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TEMPERATURES = {"0": 0.1, "1": 0.7}
NUM_SUBJECTS = 198


def load_sentences(path: str, use_header: bool = False) -> list:
    if use_header:
        df = pd.read_excel(path)
    else:
        df = pd.read_excel(path, header=None)
    list_of_sentences = df.iloc[
        :, 0
    ].tolist()  # we asume the sentences are in the first column

    return list_of_sentences


def assign_dataset_to_sentences(sentences: list, dataset: str) -> list:
    return [{"dataset": dataset, "sentence": sentence} for sentence in sentences]


def evaluate_sentences_together(
    evaluation_prompt: str,
    sentences: list,
    temperature: int,
    model: str = "gpt-3.5",
) -> list:
    gpt = guidance.llms.OpenAI(OPENAI_MODELS[model], caching=False)
    evaluate_sentence = guidance(
        """
    {{#user~}}
    {{evaluation_prompt}}

    Give short answers with only the scores as numbers.

    List of sentences:
    ----
    {{#each sentences}}- {{this.sentence}}
    {{/each~}}
    {{~/user}}

    {{#assistant~}}
    {{gen 'scores' max_tokens=1500 temperature=temperature}}
    {{~/assistant}}
    """,
        llm=gpt,
    )

    scores = evaluate_sentence(
        evaluation_prompt=evaluation_prompt,
        sentences=sentences,
        temperature=temperature,
    )["scores"]

    return scores


def evaluate_sentences_separated(
    evaluation_prompt: str,
    sentences: list,
    temperature: int,
    model: str = "gpt-3.5",
) -> list:
    gpt = guidance.llms.OpenAI(OPENAI_MODELS[model], caching=False)
    evaluate_sentence = guidance(
        """
    {{#user~}}
    {{evaluation_prompt}}

    {{sentence.sentence}}

    Give a short answer with only the score as a number.

    Sentence:
    ----
    {{~/user}}

    {{#assistant~}}
    {{gen 'score' temperature=temperature}}
    {{~/assistant}}
    """,
        llm=gpt,
    )

    scores = list(
        map(
            lambda sentence: int(
                evaluate_sentence(
                    evaluation_prompt=evaluation_prompt,
                    sentence=sentence,
                    temperature=temperature,
                )["score"]
            ),
            tqdm(sentences),
        )
    )

    return scores


def assign_experiments_values(
    score,
    trial,
    temperature,
    sentence,
    evaluation_prompt,
    dataset,
    subject,
) -> dict:
    experiment = {
        "subject": subject,
        "trial": trial,
        "temperature": temperature,
        "likert score": score,
        "evaluation_prompt": evaluation_prompt,
        "sentence": sentence,
        "dataset": dataset,
    }
    return experiment


def load_last_state():
    try:
        with open("last_successful_state.json", "r") as f:
            last_state = json.load(f)
        last_subject = last_state.get("subject", 0)
        last_temperature_type = last_state.get("temperature_type", None)
        last_evaluation_type = last_state.get("evaluation_type", None)
    except (FileNotFoundError, json.JSONDecodeError):
        last_subject = 0
        last_temperature_type = None
        last_evaluation_type = None
    return last_subject, last_temperature_type, last_evaluation_type


def rate_sentences(
    model: str = "gpt-3.5",
    output_file: str = "results.csv",
) -> None:
    random.seed(49)

    bs_sentences = assign_dataset_to_sentences(load_sentences("./data/BS.xlsx"), "0")
    bs_sentences_new = assign_dataset_to_sentences(
        load_sentences("./data/New_BS.xlsx"), "1"
    )
    bs_sentences_generated = assign_dataset_to_sentences(
        load_sentences("./data/BS_generated.xlsx", use_header=True), "2"
    )
    motivational_sentences = assign_dataset_to_sentences(
        load_sentences("./data/Motivational.xlsx"), "3"
    )
    mundane_sentences = assign_dataset_to_sentences(
        load_sentences("./data/Mundane.xlsx"), "4"
    )

    bs_sentences_all = bs_sentences + bs_sentences_generated + bs_sentences_new
    all_sentences = bs_sentences_all + mundane_sentences + motivational_sentences

    NUM_SENTENCES = len(all_sentences)

    df = pd.DataFrame()

    last_subject, last_temperature_type, last_evaluation_type = load_last_state()

    # current state variable handlers
    current_subject = last_subject
    current_temperature_type = last_temperature_type
    current_evaluation_type = last_evaluation_type

    skip_temperature = last_temperature_type is not None
    skip_evaluation = last_evaluation_type is not None

    for subject in tqdm(range(last_subject, NUM_SUBJECTS), desc="Subjects"):
        trial_counter = 0
        for temperature_type, temperature_value in tqdm(
            TEMPERATURES.items(), desc="Temperatures"
        ):  # 2 temperature types
            if skip_temperature and temperature_type != last_temperature_type:
                continue
            skip_temperature = False
            for evaluation_type, evaluation_prompt in tqdm(
                EVALUATION_PROMPTS_DICT.items(), desc="Evaluations"
            ):  # 10 evaluation types
                if skip_evaluation and evaluation_type != last_evaluation_type:
                    continue
                skip_evaluation = False
                try:
                    evaluation_scores = evaluate_sentences_separated(
                        evaluation_prompt=evaluation_prompt,
                        sentences=all_sentences,
                        temperature=temperature_value,
                        model=model,
                    )
                except Exception as e:
                    # Save the current state before terminating
                    with open("last_successful_state.json", "w") as f:
                        json.dump(
                            {
                                "subject": current_subject,
                                "temperature_type": current_temperature_type,
                                "evaluation_type": current_evaluation_type,
                            },
                            f,
                        )
                    raise e

                separated_sentences_results = list(
                    map(
                        lambda score_sentence: assign_experiments_values(
                            score=score_sentence[0],
                            temperature=temperature_type,
                            sentence=score_sentence[1]["sentence"],
                            evaluation_prompt=evaluation_type,
                            dataset=score_sentence[1]["dataset"],
                            subject=subject,
                            trial=trial_counter + score_sentence[2],
                        ),
                        zip(evaluation_scores, all_sentences, range(NUM_SENTENCES)),
                    )
                )
                trial_counter += NUM_SENTENCES

                # create new df concatenating experiment results
                df = pd.concat(
                    [df, pd.DataFrame(separated_sentences_results)], ignore_index=True
                )

                # save df preemptively (in case OpenAI fails us)
                df.to_csv(output_file, index=False)

                # Update the current state
                current_subject = subject
                current_temperature_type = temperature_type
                current_evaluation_type = evaluation_type

    # delete the state file when run completes
    if "last_successful_state.json" in os.listdir():
        os.remove("last_successful_state.json")


if __name__ == "__main__":
    fire.Fire(rate_sentences)
