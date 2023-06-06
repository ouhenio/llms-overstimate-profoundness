import json
import os
import random
import time
import traceback

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
    {{gen 'score' temperature=temperature}}
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


def assign_experiment_values(
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
        last_trial_counter = last_state.get("trial_counter", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        last_subject = 0
        last_temperature_type = None
        last_evaluation_type = None
        last_trial_counter = 0
    return last_subject, last_temperature_type, last_evaluation_type, last_trial_counter


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

    all_sentences = (
        bs_sentences
        + bs_sentences_generated
        + bs_sentences_new
        + mundane_sentences
        + motivational_sentences
    )

    NUM_SENTENCES = len(all_sentences)
    NUM_TEMPERATURES = len(TEMPERATURES)
    NUM_EVALUATION_PROMPTS = len(EVALUATION_PROMPTS_DICT)
    TOTAL_TRIALS = (
        NUM_SUBJECTS * NUM_SENTENCES * NUM_TEMPERATURES * NUM_EVALUATION_PROMPTS
    )

    # get last iteration states
    (
        last_subject,
        last_temperature_type,
        last_evaluation_type,
        last_trial_counter,
    ) = load_last_state()

    # current state variable handlers
    current_subject = last_subject
    current_temperature_type = last_temperature_type
    current_evaluation_type = last_evaluation_type
    current_trial_counter = last_trial_counter
    current_i = last_trial_counter % NUM_SENTENCES

    skip_temperature = last_temperature_type is not None
    skip_evaluation = last_evaluation_type is not None

    df = (
        pd.read_csv(output_file)
        if last_temperature_type is not None
        else pd.DataFrame()
    )

    # create progress bar and update it
    progress_bar = tqdm(range(TOTAL_TRIALS))
    progress_bar.update(len(df))

    for subject in range(last_subject, NUM_SUBJECTS):
        trial_counter = last_trial_counter if subject == last_subject else 0
        for (
            temperature_type,
            temperature_value,
        ) in TEMPERATURES.items():
            if skip_temperature and temperature_type != last_temperature_type:
                continue
            skip_temperature = False
            for (
                evaluation_type,
                evaluation_prompt,
            ) in EVALUATION_PROMPTS_DICT.items():
                if skip_evaluation and evaluation_type != last_evaluation_type:
                    continue
                skip_evaluation = False

                succesful_trials = []
                for i in range(current_i, NUM_SENTENCES):
                    sentence = all_sentences[i]["sentence"]
                    dataset = all_sentences[i]["dataset"]

                    try:
                        evaluation_score = evaluate_sentence(
                            evaluation_prompt=evaluation_prompt,
                            sentence=sentence,
                            temperature=temperature_value,
                            model=model,
                        )
                    except Exception as e:
                        # save the current state before terminating
                        with open("last_successful_state.json", "w") as f:
                            json.dump(
                                {
                                    "subject": current_subject,
                                    "temperature_type": current_temperature_type,
                                    "evaluation_type": current_evaluation_type,
                                    "trial_counter": current_trial_counter,
                                },
                                f,
                            )

                        # save advances before terminating
                        print("Encountered and error, saving progress...")
                        df = pd.concat(
                            [df, pd.DataFrame(succesful_trials)], ignore_index=True
                        )
                        df.to_csv(output_file, index=False)
                        raise e

                    # store progress
                    trial_results = assign_experiment_values(
                        score=evaluation_score,
                        temperature=temperature_type,
                        sentence=sentence,
                        evaluation_prompt=evaluation_type,
                        dataset=dataset,
                        subject=subject,
                        trial=trial_counter,
                    )
                    succesful_trials.append(trial_results)

                    # update the current state
                    trial_counter += 1
                    current_subject = subject
                    current_temperature_type = temperature_type
                    current_evaluation_type = evaluation_type
                    current_trial_counter = trial_counter
                    progress_bar.update(1)

                # save advances after succesful iteration
                df = pd.concat([df, pd.DataFrame(succesful_trials)], ignore_index=True)
                df.to_csv(output_file, index=False)

    # delete the state file when run completes
    if "last_successful_state.json" in os.listdir():
        os.remove("last_successful_state.json")


def run_with_retries(
    model: str = "gpt-3.5",
    output_file: str = "results.csv",
    delay_seconds: int = 5,
    max_retries: int = 5,
) -> None:
    original_delay_seconds = delay_seconds
    consecutive_failures = 0
    last_exception_message = None

    while True:
        try:
            rate_sentences(model, output_file)
            break
        except Exception as e:
            if str(e) == last_exception_message:
                consecutive_failures += 1
                delay_seconds += delay_seconds
            else:
                last_exception_message = str(e)
                consecutive_failures = 1
                delay_seconds = original_delay_seconds

            print(f"Exception encountered: {e}")
            print(traceback.format_exc())

            if consecutive_failures >= max_retries:
                raise RuntimeError(
                    """
                    Same exception occurred for the maximum number of retries, \
                    terminating process.
                    """
                )

            if consecutive_failures > 0:
                print("Encountered same exception. Extending delay time...")

            print(f"Retrying in {delay_seconds} seconds...")
            time.sleep(delay_seconds)


if __name__ == "__main__":
    fire.Fire(run_with_retries)
