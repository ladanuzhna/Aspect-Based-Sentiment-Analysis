import pytesseract
from pdf2image import convert_from_path
import os
import en_core_web_sm
import re
from rake_nltk import Metric,Rake


TESSERACT_EXE = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

def get_text(filename):
    """
    The function takes a path to the pdf file and converts it to the generator of text pages
    :param filename: name of the file to read
    :return: generator of pages in the file scanned
    """
    basePath = os.path.dirname(os.path.realpath(__file__))
    PDF_file = filename
    pdf_file_path = os.path.join(basePath,PDF_file)
    images = convert_from_path(pdf_file_path)

    for pg, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        yield re.sub(r'[^\w\s]','',text)

def get_entities(text):
    """

    :param text: string, text on which to perform entity recognition
    :return: prints out entities
    """
    nlp = en_core_web_sm.load() # Loading pre-trained space NLP model
    doc = nlp(text)
    #nlp = spacy.load("en_core_web_sm")
    #spacy.displacy.serve(doc, style="ent")
    print([(X.text, X.label_) for X in doc.ents])

def RAKE_words(text):
    """

    :param text:
    :return:
    """
    vacuum_stopwords = ['vacuum','operating','product','instructions','step','cleaner','1','2','3','4','5','6','7','8','9','0']
    r = Rake(stopwords=vacuum_stopwords,min_length=1,max_length=4,ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)  # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(text)
    r.get_ranked_phrases()  # To get keyword phrases ranked highest to lowest.
    return r.ranked_phrases


def extract_measurements(text):
    """

    :param text: string, page on which to search for specifications
    :return: list of specifications found (wattage, voltage, measurements)
    """
    expression = r"""\b
            (?:
              Length\ +(?P<Length>\d+(?:\.\d+)?|\d+-\d+\/\d+\ +(?:in|ft|cm|m)|
              Width\ +(?P<Width>\d+(?:\.\d+)?\ +(?:in|ft|cm|m)|
              Weight\ +(?P<Weight>\d+(?:\.\d+)?|\d+-\d\ +(?:oz|lb|g|kg)|
              Electrical\ +(?P<Electrical>[^ ]+\ +(?:VAC|Hz|[aA]mps)
            )
            \b
    """

    return re.findall(expression,text,flags=re.X|re.MULTILINE)

if __name__ == "__main__":
    t = ""
    for text in get_text("manual1.pdf"):

        t = t+'\n'+text
        #print(text)
        #print(extract_measurements(text))
    s = RAKE_words(t)
    get_entities(t)
    with open('manual1.csv', 'w') as f:
        for line in s:
            f.write(line)
            f.write('\n')
