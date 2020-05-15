'''
title, url, source, country, date, keywords
'''

import json

import nltk as nltk
from pandas import np
from sklearn.manifold import TSNE
import re

dataset = ""
with open("data/dataset.jl", "r") as file:
    dataset = file.read().replace('\n', '')
instances = dataset.split('}{')

# Forgive me Father for I have sinned
instances[0] = instances[0][1:]
instances[-1] = instances[-1][:-1]
#

#basic nlp cleanup process to clean and pick up only the relevant terms
misinfo_instances = []
sentences = []
for instance in instances:
    current_json = json.loads("{" + instance + "}")
    misinfo_instances.append(current_json)
    tagged_sentence = nltk.tag.pos_tag(current_json["title"]
                                       .upper().replace("CORONA VIRUS", "CORONA-VIRUS")
                                       .replace("WHITE HOUSE", "WHITE-HOUSE")
                                       .replace("UNITED STATES", "UNITED-STATES")
                                       .replace("”", '"').split())
    edited_sentence = [word.replace(".", "")
                           .replace(",", "").replace('"', "") for word, tag in tagged_sentence if
                       tag == 'NN' or tag == 'NNP' or tag == 'NNS' or
                       tag == 'NNPS'
                       # or tag == 'JJ' or tag == 'JJS' or tag == 'JJS'
                       # or tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or
                       #tag == 'VBZ' or tag == 'VBP'
                       ]
    new_edited_sentence = []
    for word in edited_sentence:
        if re.match('[^a-zA-Z\d:-]', word) is None:
            new_edited_sentence.append(word)
    sentences.append(new_edited_sentence)
    # print(instance["title"])

print(misinfo_instances[0]["title"])
print(misinfo_instances[0]["keywords"])
#


# t-sne plot
#import nltk
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

black_list_words = ["DUE", "USE", "DAYS", "SAID", "MADE", "SENT", "LAST",
                    "TWO", "THE", "HE", "SHE", "TAKE", "HER", "SINCE", "DAY", "ONE", "SOME", "AWAY",
                    "OVER", "AS", "GIVES", "USING", "THE", "OFF", "THESE", "PROVES", "THREE",
                    "WHILE", "CALLED", "GETTING", "THROUGH", "TAKEN", "BETWEEN", "FOUND",
                    "SAYING", "YEARS", "ACCORDING", "HOW", "DR", "ALSO"]

model = Word2Vec(sentences, min_count=30)
model.save('model/w2vmodel.bin')
# display_closestwords_tsnescatterplot(model, "fake")
words = list(model.wv.vocab)
words = list((set().union([x[0] for x in
                          model.wv.similar_by_word('UNITED-STATES', topn=20)
                          #model.wv.similar_by_word('fake', topn=20)#+
                          #model.wv.similar_by_word('news', topn=20)
                         ])).difference(black_list_words)
             )

    #model.similar_by_word('corona')).extend(model.similar_by_word('fake')).\
    #extend(model.similar_by_word('fake-news'))
X = model[words]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
#
