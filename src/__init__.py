from dataset import Dataset
from extractor_model import ExtractorModel
import spacy

PATH = "../db/"
DB = "Summer Internship - Homework Exercise.csv"
SEPARATOR = ","


if __name__ == '__main__':

    dataset = Dataset(PATH+DB, SEPARATOR)
    extractor_model = ExtractorModel(dataset)

    # extractor_model.save_info()
    # extractor_model.model()
    # extractor_model.save_info(nlp=spacy.load("../output/model-best"))
    extractor_model.evaluate_model()
