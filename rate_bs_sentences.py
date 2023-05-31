import pandas as pd
import openai
import guidance
import types
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
OPENAI_MODEL = 'gpt-3.5-turbo'

evaluation_prompts_dict = {
    "vanilla": """
        We are interested in how people experience the profound. Below are a series of statements taken from relevant websites. Please read each statement and take a moment to think about what it might mean. Then please rate how “profound” you think it is. Profound means “of deep meaning; of great and broadly inclusive significance.”

        Rate the profoundness of the following sentences on the following 5-point scale: 1= Not at all profound, 2 = somewhat profound, 3 = fairly profound, 4 = definitely profound, 5 = very profound 
    """,
    "n_shot": """
        We are interested in how people experience the profound. Below are a series of statements taken from relevant websites. Please read each statement and take a moment to think about what it might mean. Then please rate how “profound” you think it is. Profound means “of deep meaning; of great and broadly inclusive significance.”

        Rate the profoundness of the following sentences on the following 5-point scale: 1= Not at all profound, 2 = somewhat profound, 3 = fairly profound, 4 = definitely profound, 5 = very profound

        For instance, the sentence: 

        "This life is nothing short of a summoning rekindling of karmic complexity." 

        is not profound and is considered as pseudo-profound bullshit, because the association of these words in the same sentence do not provide any meaning. 
    """,
    "cot": """
        We are interested in how people experience the profound. Below are a series of statements taken from relevant websites. Please read each statement and take a moment to think about what it might mean. Then please rate how “profound” you think it is. Profound means “of deep meaning; of great and broadly inclusive significance.”

        Rate the profoundness of the following sentences on the following 5-point scale: 1= Not at all profound, 2 = somewhat profound, 3 = fairly profound, 4 = definitely profound, 5 = very profound

        To give your answer, first compare the statements that were given with a normal mundane sentence, such as:

        "The girl on the bicycle has blond hair"  (mundane case)
        "The creative adult is the child who survived" (profound case)


        Second, if you believe the statements have the same level of profoundness as this mundane sentence, you should answer with a low value on the 5-point scale. 
    """
}

EVALUATION_PROMPTS = types.SimpleNamespace(**evaluation_prompts_dict)

def load_sentences(path):
    df = pd.read_excel(path)
    list_of_sentences = df.iloc[:, 0].tolist() # we asume the sentences are in te first column

    return list_of_sentences

def list_to_string(sentences):
    return '\n'.join(['{}) {}'.format(i + 1, string) for i, string in enumerate(sentences)])

def evaluate_sentences(evaluation_prompt, sentences):
    gpt = guidance.llms.OpenAI(OPENAI_MODEL)
    evaluate_sentence = guidance('''
    {{#user~}}
    {{evaluation_prompt}}

    {{sentences}}
    {{~/user}}

    {{#assistant~}}
    {{gen 'scores'}}
    {{~/assistant}}
    ''', llm=gpt)

    scores = evaluate_sentence(evaluation_prompt=evaluation_prompt, sentences=sentences)['scores']

    return scores

if __name__ == '__main__':
    bs_sentences = load_sentences("data/BS.xlsx")
    bs_sentences_generated = load_sentences("data/BS_generated.xlsx")

    string_bs_sentences_generated = list_to_string(bs_sentences_generated)

    vanilla_scores = evaluate_sentences(
        evaluation_prompt=EVALUATION_PROMPTS.vanilla, sentences=string_bs_sentences_generated
    )