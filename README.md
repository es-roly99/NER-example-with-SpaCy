# About
Named entity recognition (NER) is a sub-task of information extraction (IE) that seeks out and categorizes specified entities in a body or bodies of texts. NER is also known simply as entity identification, entity chunking and entity extraction. NER is used in many fields in artificial intelligence (AI) including natural language processing (NLP) and machine learning.

In this task the objective is to recognize the store number within the store description. For this, a semi-automatic model is created using Python and SpaCy (free and open- source library for NLP) for the collection of entities.


#Structure
1. In the data folder will be the necessary files to create the NER.
2. Inside db are the databases.
3. In the info folder all the analyses carried out will be saved.
4. Output corresponds to the created models.
5. In src are the Python classes.

#Steps to Run
1. Download SpaCy library “pip install -U spacy”
2. Download the pipeline for English medium size “python -m spacy download
en_core_web_md”
3. In __init__.py run extractor_model.save_info() to obtain the test .csv with token and
entities.
4. In __init__.py run extractor_model.model() to create the train and dev.spacy files
5. With the terminal open in the project folder run “python -m spacy train ./data/
config.cfg --output ./output” for train the data and create the models.
6. In __init__.py run extractor_model.save_info(nlp=spacy.load(“../output/model-
best”)) to obtain the test .csv with token and entities with the new model.
7. In __init__.py run extractor_model.evaluate_model() for evaluate the test and create
metrics with the best model.
