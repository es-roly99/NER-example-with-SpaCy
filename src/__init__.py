from dataset import Dataset
from extractor_model import ExtractorModel
import spacy

DB = "work-shops.csv"
MODEL_PIPELINE = "en_core_web_md"
COLUMN_DESCRIPTION = "transaction_descriptor"
COLUMN_ENTITY = "store_number"


if __name__ == '__main__':

    dataset = Dataset(DB)
    extractor_model = ExtractorModel(dataset, MODEL_PIPELINE, COLUMN_DESCRIPTION, COLUMN_ENTITY)

    # extractor_model.save_info()
    # extractor_model.model()
    # python3 -m spacy train ./data/config.cfg --output ./output
    # extractor_model.save_info(nlp=spacy.load("../output/model-best"))
    # extractor_model.evaluate_model()
