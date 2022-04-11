from dataset import Dataset
import spacy
import pandas as pd
from spacy.tokens import DocBin
import numpy as np


class ExtractorModel:

    def __init__(self, dataset: Dataset):
        self.train = dataset.train
        self.test = dataset.test
        self.validation = dataset.validation
        self.nlp = spacy.load('en_core_web_md')

    def evaluate_model(self):
        nlp = spacy.load("../output/model-best")
        predictions = []
        columns = ['Index', 'Text', 'Original', 'Prediction']
        data = {'Index': [], 'Text': [], 'Original': [], 'Prediction': []}

        for i, (description, number) in enumerate(zip(self.test['transaction_descriptor'].values, self.test['store_number'].values), 1):
            data['Index'].append(i)
            data['Text'].append(description)
            data['Original'].append(number)
            prediction = nlp(description).ents
            if len(prediction) > 0:
                predictions.append(self.clean_number(str(prediction[0])))
            else:
                predictions.append("")
            data['Prediction'].append(predictions[i-1])

        pd.DataFrame(data, columns=columns).to_csv('../info/result.csv', index=False)

        predictions = np.array(predictions)
        numbers = np.array(self.test['store_number'].values)

        data_analysis = dict([('Total', [len(predictions)]),
                             ('Well Predictions', [sum(predictions == numbers)]),
                             ('Bad Predictions', [sum(predictions != numbers) - sum(predictions == '')]),
                             ('Null Predictions', [sum(predictions == '')])])

        pd.DataFrame(data_analysis, columns=['Total', 'Well Predictions', 'Bad Predictions', 'Null Predictions'])\
            .to_csv('../info/result_analysis.csv', index=False)

    def model(self):
        transaction_descriptor = [self.train['transaction_descriptor'].values,
                                  self.validation['transaction_descriptor'].values]
        store_number = [self.train['store_number'].values,
                        self.validation['store_number'].values]
        dbs = []

        for descriptions, numbers in zip(transaction_descriptor, store_number):
            db = DocBin()
            for description, number in zip(descriptions, numbers):
                doc = self.nlp(description)
                for token in doc:
                    if token.text[-len(number):] == number:
                        start, end, label = [(token.idx, token.idx + len(number), token.ent_type_)][0]
                        span = doc.char_span(start, end, label='store_number')
                        if span is not None:
                            doc.ents = [span]
                            db.add(doc)
                        break
            dbs.append(db)

        dbs[0].to_disk("../data/train.spacy")
        dbs[1].to_disk("../data/dev.spacy")

    def clean_number(self, number):
        c = number[0]
        while c == '0':
            number = number[1:]
            c = number[0]
        return number

    def save_info(self, nlp=None, option=''):
        if nlp is None:
            nlp = self.nlp
        else:
            option = '_trained'

        descriptions = self.test['transaction_descriptor'].values
        columns_tokenized = ['Index', 'Text', 'Lemma', 'Pos', 'Tag', 'Dep', 'Shape', 'Is_Alpha', 'Is_Stop']
        data_tokenized = {'Index': [], 'Text': [], 'Lemma': [], 'Pos': [], 'Tag': [], 'Dep': [],
                          'Shape': [], 'Is_Alpha': [], 'Is_Stop': []}
        columns_entities = ['Index', 'Entity', 'Entity_Type']
        data_entities = {'Index': [], 'Entity': [], 'Entity_Type': []}

        for i, row in enumerate(descriptions, 1):
            for token in nlp(row):
                data_tokenized['Index'].append(i)
                data_tokenized['Text'].append(token.text)
                data_tokenized['Lemma'].append(token.lemma_)
                data_tokenized['Pos'].append(token.pos_)
                data_tokenized['Tag'].append(token.tag_)
                data_tokenized['Dep'].append(token.dep_)
                data_tokenized['Shape'].append(token.shape_)
                data_tokenized['Is_Alpha'].append(token.is_alpha)
                data_tokenized['Is_Stop'].append(token.is_stop)
            for ent in nlp(row).ents:
                data_entities['Index'].append(i)
                data_entities['Entity'].append(ent)
                data_entities['Entity_Type'].append(ent.label_)

        pd.DataFrame(data_tokenized, columns=columns_tokenized).to_csv('../info/test_tokenized' + option + '.csv', index=False)
        pd.DataFrame(data_entities, columns=columns_entities).to_csv('../info/test_entities' + option + '.csv', index=False)
