import re
import argparse
import pandas as pd


REGULAR_EXPRESSIONS_FOR_PARSING = {
    'greeting': 'здравствуйте|добрый день|добрый вечер|доброе утро|привет|приветствую',
    'farewell': 'до свидания|хорошего вечера|хорошего дня|всего хорошего|доброй ночи|всего доброго',
    'introduce': 'меня .*зовут|мо[её] имя|имя мо[её]',
}


def build_parser():
    parser = argparse.ArgumentParser(description='Parses .csv file with dialogues.')
    parser.add_argument('--path', type=str)
    return parser


def find_regexp(reg_exp, lower=True):
    def find(string):
        if lower:
            string = string.lower()
        if re.search(reg_exp, string):
            return True
        return False
        
    return find


def make_parsed_path(path):
    splitted_path = path.split('.')
    splitted_path[-2] = splitted_path[-2] + '_parsed'
    parsed_file_path = '.'.join(splitted_path)
    return parsed_file_path

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.path.endswith('.csv'):
        raise ValueError('Input file must have .csv format')
    df = pd.read_csv(args.path)
    print(f'\nOpened file {args.path}.')

    parsed_file_path = make_parsed_path(args.path)
    df.to_csv(parsed_file_path)
    print(f'\nSaved parsed file to {parsed_file_path}!\n')
