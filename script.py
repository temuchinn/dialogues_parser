import re
import argparse
import pandas as pd
import numpy as np


REGULAR_EXPRESSIONS_FOR_PARSING = {
    'greeting': 'здравствуйте|добрый день|добрый вечер|доброе утро|привет|приветствую',
    'farewell': 'до свидания|хорошего вечера|хорошего дня|всего хорошего|доброй ночи|всего доброго',
    'introduce': 'меня .*зовут|мо[её] имя|имя мо[её]',
    'ask_name': '(как к вам можно|как (я ){0,1}могу к вам) обра(щаться|титься)|уточнит[ье] (пожалуйста ){0,1}(сво[её] ){0,1}имя|как вас (зовут|можно звать)',
    'probably_contain_company': 'компани[июя]|организаци[июя]|представляю',
}


def build_parser():
    parser = argparse.ArgumentParser(description='Parses .csv file with dialogues.')
    parser.add_argument('--path', type=str)
    return parser


def make_parsed_path(path):
    splitted_path = path.split('.')
    splitted_path[-2] = splitted_path[-2] + '_parsed'
    parsed_file_path = '.'.join(splitted_path)
    return parsed_file_path


def find_regexp(reg_exp, lower=True):
    def find(string):
        if lower:
            string = string.lower()
        if re.search(reg_exp, string):
            return True
        return False
        
    return find


def get_insignts(df: pd.DataFrame, dlg_id: int) -> dict:
    insights = {}
    
    dlg_df = df[df['dlg_id'] == dlg_id]
    insights['manager_greeting'] = np.any(dlg_df[(dlg_df['role'] == 'manager')].greeting)
    insights['manager_farewell'] = np.any(dlg_df[(dlg_df['role'] == 'manager')].farewell)

    return insights

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.path.endswith('.csv'):
        raise ValueError('Input file must have .csv format')
    df = pd.read_csv(args.path)
    print(f'\nOpened file "{args.path}".')

    for reg_exp_name, reg_exp in REGULAR_EXPRESSIONS_FOR_PARSING.items():
        regexp_finder = find_regexp(reg_exp)
        df.loc[:, reg_exp_name] = df['text'].apply(regexp_finder)
    
    dialogues_insights = dict()


    parsed_file_path = make_parsed_path(args.path)
    df.to_csv(parsed_file_path)
    print(f'\nSaved parsed file to "{parsed_file_path}"!\n')
