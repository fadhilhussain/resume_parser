import spacy

nlp = spacy.load("en_core_web_lg")

text = nlp('''Fadhil Hussain
ibnhussainkv@gmail.com
+91 9645343919
Kannur, Kerala
Summary
Enthusiastic and detail-oriented fresher in Machine Learning with hands-on experience in regression, classification, and
clustering. Proficient in Python and scikit-learn, with a strong foundation in building end-to-end machine learning and deep
learning ANN projects. Passionate about applying data-driven solutions to real-world problems and continuously learning
advanced techniques.''')

for ent in  text.ents:
    print(ent.text,"--->",ent.label_)
