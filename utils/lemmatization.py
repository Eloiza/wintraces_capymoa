# import nltk
# from nltk.stem import WordNetLemmatizer
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # Initialize wordnet lemmatizer
# wnl = WordNetLemmatizer()

# # Example inflections to reduce
# example_words = ["program12412", "123145", "123", "program","programming","programer","programs","programmed"]
# # example_words = [""]
# # Perform lemmatization
# print("{0:20}{1:20}".format("--Word--","--Lemma--"))
# for word in example_words:
#     print ("{0:20}{1:20}".format(word, wnl.lemmatize(word, pos="v")))

# """
# --Word--            --Lemma--           
# program             program             
# programming         program             
# programer           programer           
# programs            program             
# programmed          program
# """   

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure you have the necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Example words
words = ["running", "ran", "runs", "better", "studies"]

# POS tagging
tagged_words = nltk.pos_tag(words)
print(tagged_words)
# Lemmatize each word with its POS tag
lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words]

print(lemmatized_words)
