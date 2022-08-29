import re
import argparse
import json

import pandas as pd
import numpy as np

import nltk
import pymorphy2

from yargy import Parser, rule
from yargy.predicates import gram,  dictionary
from yargy.interpretation import fact


REGULAR_EXPRESSIONS_FOR_PARSING = {
    'greeting': 'здравствуйте|добрый день|добрый вечер|доброе утро|привет|приветствую',
    'farewell': 'до свидания|хорошего вечера|хорошего дня|всего хорошего|доброй ночи|всего доброго',
    'introduce': 'меня .*зовут|мо[её] имя|имя мо[её]',
    'probably_contain_company': 'компани[июя]|организаци[июя]|представляю',
}


Name = fact(
    'Name',
    ['first', 'last']
)


NAME_INTR = rule(
    dictionary({
        'зовут',
        'меня',
        'я',
        'это',
    }),
    gram('Name')\
    .interpretation(
        Name.first.inflected()
    )
).interpretation(Name)


COMPANY = rule(
    dictionary({
        'компания',
        'компанию',
        'организация',
        'организацию'
    }),
    gram('NOUN'),
    gram('NOUN').optional().repeatable(),
)


class YargyParser:
    """Class, containing methods for using yargy parser to extract text entities."""
    
    @staticmethod
    def extract_introduce_names(text: str, lower: bool = True):
        parser = Parser(NAME_INTR)
        output = []
        if lower:
            _text = text.lower()
            
        for match in parser.findall(_text):
            name = match.fact.first
            output.append(name)
        
        return list(set(output))
    
    @staticmethod
    def extract_company(text: str, lower: bool = True):
        parser = Parser(COMPANY)
        output = []
        if lower:
            _text = text.lower()
        
        for match in parser.findall(_text):
            company_name = ' '.join([_.value for _ in match.tokens][1:])
            if len(company_name) > 2:
                output.append(company_name)
        
        return list(set(output))

        
class NerExtractor:
    """Class containing methods for NER.
    Methods:
        extract_names(text, prob_thresh): proccesses text and finds all names, which confidence >= prob_thresh
        
    """
    NAME_TAG = 'Name'
    morph_analyzer = pymorphy2.MorphAnalyzer()
    
    @classmethod
    def extract_names(cls, text: str, prob_thresh: float = 0.4) -> list[str]:
        names = []

        for word in nltk.word_tokenize(text):
            for p in cls.morph_analyzer.parse(word):
                if cls.NAME_TAG in p.tag and p.score >= prob_thresh:
                    names.append(p.normal_form)
                    break
        return list(set(names))


def build_arg_parser():
    """Builds a parser of arguments.
    Args:
        None
    Returns:
        parser object
    
    """
    parser = argparse.ArgumentParser(description='Parses .csv file with dialogues.')
    parser.add_argument('--path', type=str)
    return parser


def make_output_path(path: str) -> str:
    """Format data path to output path.
    Args:
        path (str): Origin path
    Returns:
        str: formatted path

    """
    splitted_path = path.split('.')
    splitted_path[-2] = splitted_path[-2] + '_parsed'
    parsed_file_path = '.'.join(splitted_path)
    return parsed_file_path


def find_regexp(reg_exp: str, lower: bool=True):
    """Returns function, checking regexp.
    Args:
        reg_exp (str): regular expression, which will be checked
        lower (bool): flag, defining whether to lower the input string
    Returns:
        function: function, checking reg exp in a string

    """
    def find(string: str) -> bool:
        if lower:
            string = string.lower()
        if re.search(reg_exp, string):
            return True
        return False
        
    return find


def get_insights(df: pd.DataFrame, dlg_id: int) -> dict[str, any]:
    """Returns dict with insights from dataframe.
    Args:
        df (DataFrame): dataframe, containing supportive columns
        dlg_id (int): id of a dialogue, which insights will be received
    Returns:
        dict[str, bool | ...]: dict with insights
        
    """
    insights = {}
    name_extractor = NerExtractor()
    yp = YargyParser()

    dlg_df = df[df['dlg_id'] == dlg_id]
    insights['manager_greeting'] = bool(np.any(dlg_df[(dlg_df['role'] == 'manager')].greeting))
    insights['manager_farewell'] = bool(np.any(dlg_df[(dlg_df['role'] == 'manager')].farewell))
    insights['manager_introduced'] = bool(np.any(dlg_df[(dlg_df['role'] == 'manager')].introduce))
    insights['all_names'] = name_extractor.extract_names(' '.join(dlg_df.text.to_list()))
    insights['manager_name'] = yp.extract_introduce_names(' '.join(dlg_df[(dlg_df['role'] == 'manager')].text.to_list()))
    insights['client_name'] = yp.extract_introduce_names(' '.join(dlg_df[(dlg_df['role'] == 'client')].text.to_list()))
    insights['company'] = yp.extract_company(' '.join(dlg_df.text.to_list()))

    return insights


if __name__ == '__main__':

    arg_parser = build_arg_parser() 
    args = arg_parser.parse_args()
    
    if not args.path.endswith('.csv'):
        raise ValueError('Input file must have .csv format')

    df = pd.read_csv(args.path)
    print(f'\nOpened file "{args.path}".')

    # adding supportive columns to dataframe
    for reg_exp_name, reg_exp in REGULAR_EXPRESSIONS_FOR_PARSING.items():
        regexp_finder = find_regexp(reg_exp)
        df.loc[:, reg_exp_name] = df['text'].apply(regexp_finder)
    
    dialogues_insights = dict()
    for dlg_id in df.dlg_id.unique():
        dialogues_insights[int(dlg_id)] = get_insights(df, dlg_id)

    parsed_file_path = make_output_path(args.path)
    df.to_csv(parsed_file_path)
    
    with open('insights.json', mode='w', encoding='utf-8') as fp:
        json.dump(dialogues_insights ,fp)
    
    for dlg_id, insights in dialogues_insights.items():
        print("Dialogue id:", dlg_id)
        print("Manager_greeting:", insights['manager_greeting'])
        print("Manager_farewell:", insights['manager_farewell'])
        print("greeting + farewell:"    , insights['manager_greeting'] and insights['manager_farewell'])
        print("Manager indtroduced:", insights['manager_introduced'])
        print("Names in dialogue:", insights['all_names'])
        print("Manager introduced name:", insights['manager_name'])
        print("Client introduced name:", insights['client_name'])
        print("Company:", insights['company'] )
        print()
    

    print(f'\nSaved parsed file to "{parsed_file_path}"!\n')
