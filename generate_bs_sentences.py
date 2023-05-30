import fire
import pandas as pd
import numpy as np
import spacy
import random
import openai
import guidance
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
OPENAI_MODEL = 'gpt-3.5-turbo'
nlp = spacy.load("en_core_web_lg")
random.seed(49)

def generate_bs_sentences(
    input_file='data/BS.xlsx',
    cleanup_sentences=False,
    output_file='data/BS_generated.xlsx',
    similar_word_replacement_number=10,
):
    bs_sentences = pd.read_excel(input_file, header=None)[0]
    sentences = list(map(
        lambda sentence: create_sentence_variation(
            sentence,
            similar_word_number=similar_word_replacement_number
        ), tqdm(bs_sentences))
    )

    if cleanup_sentences:
        sentences = cleanup_sentences(sentences)
    else:
        print('Skipping sentence cleanup')

    save_sentences(sentences, output_file)


def create_sentence_variation(sentence, similar_word_number=10):
  tokens = nlp(sentence)

  new_words = []
  for token in tokens:
    if token.is_alpha and not token.is_stop and token.pos_ == "NOUN" and random.random() < 0.9:
      new_word = random.choice(most_similar_words(token.text, n=similar_word_number))
      new_words.append(new_word)
    else:
      new_words.append(token.text)

  return ' '.join([word for word in new_words])


def most_similar_words(word, n=10):
    vector_words = nlp.vocab.vectors.most_similar(
        np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]),
        n=n
    )
    words = [nlp.vocab.strings[string] for string in vector_words[0][0]]
    return words


def cleanup_sentences(sentences):
    gpt = guidance.llms.OpenAI(OPENAI_MODEL)
    rewrite_sentence = guidance('''
    {{#user~}}
    Fix any inconsistent uppercase, spaces or punctuation of the following sentence:

    {{sentence}}
    {{~/user}}

    {{#assistant~}}
    {{gen 'answer'}}
    {{~/assistant}}
    ''', llm=gpt)

    new_sentences = list(map(
        lambda sentence: rewrite_sentence(sentence=sentence)['answer'], sentences
    ))
    return new_sentences

def save_sentences(sentences, output_file):
    df = pd.DataFrame(sentences)
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='sentences', index=False)
    writer.close()

if __name__ == '__main__':
    fire.Fire(generate_bs_sentences)
