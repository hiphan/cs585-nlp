from github import Github
import requests
import json
import os
from utils import *


class DataGetter:

    def __init__(self, github_token, mode='all'):
        self.token = github_token
        self.mode = mode
        self.github = Github(self.token)
        self.repo = self.github.search_repositories('OneStopEnglishCorpus')[0]
        self.path = 'https://raw.githubusercontent.com/nishkalavallabhi/OneStopEnglishCorpus/master/'

    def create_json(self, save=True):
        ele = self.create_json_level(level='Ele', save=False)
        inter = self.create_json_level(level='Int', save=False)
        adv = self.create_json_level(level='Adv', save=False)
        corpus = {
            'elementary': ele,
            'intermediate': inter,
            'advanced': adv
        }
        if save:
            path = './data/all.json'
            with open(path, 'w') as fp:
                json.dump(corpus, fp, sort_keys=True, indent=4, separators=(',', ': '))
        return corpus

    def create_json_level(self, level, save=True):
        if level == 'Ele':
            print('Loading elementary level documents...')
        elif level == 'Int':
            print('Loading intermediate level documents...')
        elif level == 'Adv':
            print('Loading advanced level documents...')
        else:
            raise ValueError('Invalid difficulty')
        level_corpus = {}
        for tf in self.repo.get_contents('Texts-SeparatedByReadingLevel/%s-Txt' % level):
            title = tf.path.split("/")[-1]
            if title == '.DS_Store' or tf.type == 'dir':
                continue
            url = self.path + tf.path
            req = requests.get(url)
            if req.status_code == requests.codes.ok:
                content = req.content.decode('utf-8-sig')
            else:
                print('Error. Cannot load file at ' + url)
            if level == 'Int':
                content = content[len('Intermediate '):]
            level_corpus[title] = content
        if save:
            path = './data/%s.json' % level
            with open(path, 'w') as fp:
                json.dump(level_corpus, fp, sort_keys=True, indent=4, separators=(',', ': '))
        print('Finished loading %d documents.' % len(level_corpus))
        return level_corpus


if __name__ == '__main__':
    token = 'a5d4405a989f78633bb4f29355117ce4a8da5135'
    dg = DataGetter(token)

    if not os.path.isdir('./data/'):
        os.mkdir('./data/')


    # dg.create_json_level(level='Ele')
    # dg.create_json_level(level='Int')
    # dg.create_json_level(level='Adv')
    dg.create_json()
