import os
import pandas as pd
import openai
import guidance
import random
import fire

from dotenv import load_dotenv
from prompts import EVALUATION_PROMPTS_DICT
from tqdm import tqdm


load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')

OPENAI_MODELS = {'gpt-4': 'gpt-4', 'gpt-3.5': 'gpt-3.5-turbo'}
TEMPERATURES = {'0': 0.1, '1': 0.7}

def load_sentences(path: str) -> list:
    df = pd.read_excel(path)
    list_of_sentences = df.iloc[:, 0].tolist() # we asume the sentences are in the first column

    return list_of_sentences

def assign_dataset_to_sentences(sentences: list, dataset: str) -> list:
    return [{"dataset": dataset, "sentence": sentence} for sentence in sentences]


def evaluate_sentences_together(
    evaluation_prompt: str,
    sentences: list,
    temperature: int,
    model: str = 'gpt-3.5',
) -> list:
    gpt = guidance.llms.OpenAI(OPENAI_MODELS[model], caching=False)
    evaluate_sentence = guidance('''
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
    ''', llm=gpt)

    scores = evaluate_sentence(
        evaluation_prompt=evaluation_prompt,
        sentences=sentences,
        temperature=temperature
    )['scores']

    return scores

def evaluate_sentences_separated(
    evaluation_prompt: str,
    sentences: list,
    temperature: int,
    model: str = 'gpt-3.5',
) -> list:
    gpt = guidance.llms.OpenAI(OPENAI_MODELS[model], caching=False)
    evaluate_sentence = guidance('''
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
    ''', llm=gpt)

    scores = list(map(
        lambda sentence: int(evaluate_sentence(
            evaluation_prompt=evaluation_prompt,
            sentence=sentence,
            temperature=temperature
        )["score"]), tqdm(sentences))
    )

    return scores

def assign_experiments_values(
    score,
    trial,
    temperature,
    sentence,
    evaluation_prompt,
    dataset,
    block,
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
        "block": block,
    }
    return experiment

def rate_sentences(
    model: str = 'gpt-3.5',
    output_file: str = 'results.csv',
) -> None:
    random.seed(49)

    bs_sentences = assign_dataset_to_sentences(load_sentences("./data/BS.xlsx"), "0")
    bs_sentences_new = assign_dataset_to_sentences(
        load_sentences("./data/New_BS.xlsx"), "1"
    )
    bs_sentences_generated = assign_dataset_to_sentences(
        load_sentences("./data/BS_generated.xlsx"), "2"
    )
    motivational_sentences = assign_dataset_to_sentences(
        load_sentences("./data/Motivational.xlsx"), "3"
    )
    mundane_sentences = assign_dataset_to_sentences(
        load_sentences("./data/Mundane.xlsx"), "4"
    )

    bs_sentences_all = bs_sentences + bs_sentences_generated + bs_sentences_new
    all_sentences = bs_sentences_all + mundane_sentences + motivational_sentences # 0
    all_sentences_shuffled = all_sentences.copy() # 1
    random.shuffle(all_sentences_shuffled)

    sentences_to_evaluate = {"0": all_sentences, "1": all_sentences_shuffled}

    df = pd.DataFrame()

    subject_counter = 0
    for trial in tqdm(range(200), desc="Trials"): 
        for temperature_type, temperature_value in tqdm(
            TEMPERATURES.items(), desc="Temperatures"
        ):  # 2 temperature types
            for evaluation_type, evaluation_prompt in tqdm(
                EVALUATION_PROMPTS_DICT.items(), desc="Evaluations"
            ): # 10 evaluation types
                for sentences_type, sentences in tqdm(
                    sentences_to_evaluate.items(), desc="Sentences"
                ): # 2 sentence types
                    evaluation_scores = evaluate_sentences_separated(
                        evaluation_prompt=evaluation_prompt,
                        sentences=sentences,
                        temperature=temperature_value,
                        model=model,
                    )
                    separated_sentences_results = list(
                        map(
                            lambda score_sentence: assign_experiments_values(
                                score=score_sentence[0],
                                trial=trial,
                                temperature=temperature_type,
                                sentence=score_sentence[1]["sentence"],
                                evaluation_prompt=evaluation_type,
                                dataset=score_sentence[1]["dataset"],
                                block=sentences_type,
                                subject=subject_counter,
                            ),
                            zip(evaluation_scores, sentences),
                        )
                    )

                    # create new df concatenating experiment results
                    df = pd.concat([df, pd.DataFrame(separated_sentences_results)], ignore_index=True)

                    evaluation_scores = evaluate_sentences_together(
                        evaluation_prompt=evaluation_prompt,
                        sentences=sentences,
                        temperature=temperature_value,
                        model=model,
                    )
                    evaluation_scores = [
                        int(string) for string in list(evaluation_scores) if string.isdigit()
                    ]
                    together_sentences_results = list(
                        map(
                            lambda score_sentence: assign_experiments_values(
                                score=score_sentence[0],
                                trial=trial,
                                temperature=temperature_type,
                                sentence=score_sentence[1]["sentence"],
                                evaluation_prompt=evaluation_type,
                                dataset=score_sentence[1]["dataset"],
                                block=sentences_type,
                                subject=subject_counter,
                            ),
                            list(zip(evaluation_scores, sentences)),
                        )
                    )

                    # create new df concatenating experiment results
                    df = pd.concat([df, pd.DataFrame(together_sentences_results)], ignore_index=True)

                    # save df preemtively (in case OpenAI fails us)
                    df.to_csv(output_file, index=False)
            subject_counter += 1


if __name__ == '__main__':
    fire.Fire(rate_sentences)
