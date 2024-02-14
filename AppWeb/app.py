from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import re
import string
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack# Importez vos fonctions personnalisées
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
import joblib
from scipy.sparse import hstack

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
st = ISRIStemmer()
nltk.download('stopwords')
stop=stopwords.words('arabic')
import nltk
nltk.download('punkt')

def count_sentences_arabic(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)
def analyze_arabic_text(text):
    # Nombre de phrases dans le texte
    sentences = count_sentences_arabic(text)

    # Nombre de mots dans le texte
    words = len(text.split())

    # Nombre de caractères dans le texte
    characters = len(text)

    return sentences, words, characters

def analyze_arabic_file(file_path):
    # Lecture du contenu du fichier
    print(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read() # lire tout le contenu du fichier et stocker le dans la variable file_content

        # Analyse du texte du fichier en utilisant la fonction analyze_arabic_text
        sentences_count, words_count, characters_count = analyze_arabic_text(file_content)

        # Affichage des résultats
        #print("Le nombre de phrases dans le fichier est : ", sentences_count)
        #print("Le nombre de mots dans le fichier est : ", words_count)
        #print("Le nombre de caractères dans le fichier est : ", characters_count)
        return sentences_count, words_count, characters_count
        
    except FileNotFoundError:
        print(f"Fichier introuvable : {file_path}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
from jpype import startJVM, shutdownJVM, java
import os
import sys
from pathlib import Path
import jpype
from jpype import *

def extraire_colonne_secondaire(texte):
    mots = texte.split('{{')
    colonne_secondaire = [mot.split(':')[1][:-2] if '}}' in mot else '' for mot in mots]
    colonne_secondaire = [mot.strip('}').strip() for mot in colonne_secondaire if mot]
    return colonne_secondaire


def lsp(chemin_fichier):
    #print(jpype.getDefaultJVMPath())
    path1='c:\\tmp'
    
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath() ,'-ea', classpath=[path1],convertStrings=True )
    
    path2= 'C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\PosTaggerAlKhalil.jar'
    jpype.addClassPath(path2)
    
    path3= 'C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\ADAT-Analyzer.jar'
    jpype.addClassPath(path3)
    
    #Import the Java class
    LemmaStemma = jpype.JClass("net.oujda_nlp_team.ADATAnalyzer")
    POStag = jpype.JClass("net.nlp.parser.TestParserClient")
    
    # Chemin vers le fichier
    #chemin_fichier = r"C:\Users\mimif\OneDrive\Bureau\S3\maz\Ressources lisibilité\niveaux\1\استئجار شقة.txt"
    #chemin_fichier = r"C:\Users\mimif\OneDrive\Bureau\S3\maz\Ressources lisibilité\niveaux\test.txt"
    
    with open(chemin_fichier, encoding="utf8") as file:
        file_content = file.read()
        
    #Lemmatizer, type de retour String
    lemmes = LemmaStemma.getInstance().processADATLemmatizerString(file_content)
    
    #Stemmer, type de retour String
    stems = LemmaStemma.getInstance().processADATStemmerString(file_content)

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    lemmes = extraire_colonne_secondaire(lemmes)
    stems = extraire_colonne_secondaire(stems)

     # Retirer les marqueurs "{{" et "}}" de chaque élément avant de les fusionner
    lemmes = [lemme.replace('{{', '').replace('}}', '') for lemme in lemmes]
    stems = [stem.replace('{{', '').replace('}}', '') for stem in stems]
    # Concaténer les lemmes et les stems en une seule chaîne de caractères
    #lemme_string = ' '.join(lemmes)
    #stem_string = ' '.join(stems)
    
    return  lemmes, stems#lemme_string, stem_string


from jpype import startJVM, shutdownJVM, java
import os
import sys
from pathlib import Path
import jpype
from jpype import *



def write_lemmas_to_file(lemmas_list):
    # Chemin vers le fichier temporaire
    file_path = r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\tmp.txt"  # Adapter le chemin si nécessaire
    
    # Écriture des lemmes dans le fichier temporaire
    with open(file_path, 'w', encoding='utf-8') as file:
        for lemma in lemmas_list:
            file.write(lemma + '\n')
    
    return file_path

def extraire_tags_POS(fichier_POS):
    # Lire le contenu du fichier de sortie POS
    with open(fichier_POS, 'r', encoding='utf-8') as file:
        contenu = file.read()
    
    # Extraire les tags PoS de chaque {{ }} dans le fichier de sortie POS
    tags_POS = []
    elements = contenu.split('{{')
    for element in elements:
        if '|' in element and '+' in element:
            tag = element.split('+')[1].split('|')[0]
            tags_POS.append(tag)

    # Construire une chaîne avec des espaces entre chaque tag POS
    tags_string = ' '.join(tags_POS)
    
    return tags_string

def POS(chemin_fichier):

    #chemin_fichier = write_lemmas_to_file(lems)
    #print(jpype.getDefaultJVMPath())
    path1='c:\\tmp'
    
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath() ,'-ea', classpath=[path1],convertStrings=True )

    path3= 'C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\ADAT-Analyzer.jar'
    jpype.addClassPath(path3)
    
    #Import the Java class
    POSclass = jpype.JClass("net.oujda_nlp_team.ADATAnalyzer")
    
    with open(chemin_fichier, encoding="utf8") as file:
        file_content = file.read()

    #POSTagger, type de retour .txt file
    POSclass.getInstance().processADATStemmerWithPOSFile(chemin_fichier, "utf-8", "output_POS.txt", "utf-8")
    
    #liste de pos tags
    chemin_fichier = r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\output_POS.txt"
    pos = extraire_tags_POS(chemin_fichier)
    
    return pos

def lemmes_ambigus(lems, pos):
    lemme_amb = []
    for lem in lems: #set(lems):  # Parcours des lemmes uniques
        indices = [i for i, x in enumerate(lems) if x == lem]  # Récupération des indices du lem dans lems
        pos_tmp = pos[indices[0]]  # Premier tag POS associé au lem
        
        for index in indices[1:]:  # Parcours des indices restants pour le même lem
            pos_actuel = pos[index]  # Tag POS actuel associé au lem
            if pos_actuel != pos_tmp:  # Comparaison des tags POS
                lemme_amb.append(lem)  # Si les tags POS sont différents, ajout à la liste des lemmes ambigus
                break  # Sortie de la boucle pour passer au lem suivant

    return lemme_amb
def enlever_ponctuation_arabe(fichier):
    # Liste des symboles de ponctuation arabes
    ponctuation_arabe = '،؍؛؟٪٫٬؞؟؛؛،۔؟٪٫٬٭٠١٢٣٤٥٦٧٨٩٪٪۔ۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩‘’“”•…:'
    
    # Lecture du fichier
    with open(fichier, 'r', encoding='utf-8') as file:
        contenu = file.read()

    # Suppression de la ponctuation arabe
    contenu_sans_ponctuation = ''.join(caractere for caractere in contenu if caractere not in ponctuation_arabe)

    # Écriture du contenu sans ponctuation dans le même fichier
    with open(fichier, 'w', encoding='utf-8') as file:
        file.write(contenu_sans_ponctuation)

import string

def enlever_ponctuation(fichier):
    # Liste des symboles de ponctuation
    ponctuation = string.punctuation

    # Lecture du fichier
    with open(fichier, 'r', encoding='utf-8') as file:
        contenu = file.read()

    # Suppression de la ponctuation
    '''When using join() in Python, the string specified before the join() method acts as a separator between the elements being joined.
     indicates that all the characters that pass the condition (i.e., 
     those not in ponctuation) will be concatenated directly without any spaces or separators between them, forming a single string.    
    '''
    contenu_sans_ponctuation = ''.join(caractere for caractere in contenu if caractere not in ponctuation) # '' emptycaract not space caract

    # Écriture du contenu sans ponctuation dans le même fichier
    with open(fichier, 'w', encoding='utf-8') as file:
        file.write(contenu_sans_ponctuation)

def count_element(lemmes, stems):
  
    # Calcul du nombre de lemmes générés
    nombre_lemmes = len(lemmes)

    # Calcul du nombre de stems
    nombre_stems = len(stems)
    
    # Création d'un ensemble (set) pour obtenir les lemmes distincts
    #Lorsque nous convertissons une liste (ou toute autre structure itérable) en un ensemble en utilisant set(), 
        #Python élimine automatiquement les doublons, ne laissant que les éléments uniques.
    lemmes_distincts = set(lemmes)
    
    # Calcul du nombre de lemmes distincts
    nombre_lemmes_distincts = len(lemmes_distincts)
    
    # Calcul du nombre de lemmes fréquents (apparaissant plus d'une fois)
    lemmes_freq = {}
    for lemme in lemmes:
        if lemme in lemmes_freq:
            lemmes_freq[lemme] += 1
        else:
            lemmes_freq[lemme] = 1

    nombre_lemmes_freq = sum(1 for lemme, freq in lemmes_freq.items() if freq > 1)
    
    return nombre_lemmes, nombre_lemmes_distincts, nombre_lemmes_freq, nombre_stems 


def process_and_save_file():
    chemin_fichier= r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\fichier.txt"
    # Calcul nbr de phrases, mots et caracts
    sentences_count, words_count, characters_count = analyze_arabic_file(chemin_fichier)
    # Enlever la ponctuation
    enlever_ponctuation_arabe(chemin_fichier)
    enlever_ponctuation(chemin_fichier)
    # Extraction de lemmes et de stems
    lemmes, stems = lsp(chemin_fichier)
    #liste de posTagger
    write_lemmas_to_file(lemmes)
    file_path = r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\tmp.txt"
    pos = POS(file_path)
    # calcul nombre de lemme, lemmeDinstinct, lemmeFreq, et nombre de stem
    nombre_lemmes, nombre_lemmes_distincts, nombre_lemmes_freq, nombre_stems = count_element(lemmes, stems)
    #les lemmes ambigus
    lem_amb = lemmes_ambigus(lemmes, pos)
    #le nombre de lemmes ambigus
    nombre_lemmes_ambigus = len(lem_amb)
    # Construire une chaîne avec des espaces entre chaque tag POS
    lemmes = ' '.join(lemmes)
    lem_amb = ' '.join(lem_amb)
    stems = ' '.join(stems)

    with open(chemin_fichier, encoding="utf8") as file:
        contenu = file.read()

    # Création d'un DataFrame avec les données
    df = pd.DataFrame({
        'Nom du fichier': [os.path.basename(chemin_fichier)],
        'Contenu': [contenu],
        'SL1': [sentences_count],
        'WL1': [words_count],
        'WL5': [characters_count],
        'lemmes': [lemmes],
        'lemmes_ambigus': [lem_amb],
        'nombre_lemmes': [nombre_lemmes],
        'VL1': [nombre_lemmes_distincts],
        'AMb1': [nombre_lemmes_ambigus],
        'VL2': [nombre_lemmes_freq],
        'stems': [stems],
        'WL4': [nombre_stems],
        'POS': [pos],
        'classe': 'inconnu'  # Spécifiez la classe pour le fichier
    })

    # Enregistrement du DataFrame dans un fichier CSV
    chemin_csv = r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\fichier.csv"
    df.to_csv(chemin_csv, index=False)  # Ne pas inclure l'index dans le fichier CSV

def clean(text):
  #remove all English chars 
  text = re.sub(r'\s*[A-Za-z]\s*', ' ' , text)
  #remove hashtags
  text = re.sub("#", " ", text)
  #remove all numbers 
  text = re.sub(r'\[0-9]*\]',' ',text)
  #remove duplicated chars
  text = re.sub(r'(.)\1+', r'\1', text)
  #remove :) or :(
  text = text.replace(':)', "")
  text = text.replace(':(', "")
  #remove multiple exclamation
  text = re.sub(r"(\!)\1+", ' ', text)
  #remove multiple question marks
  text = re.sub(r"(\?)\1+", ' ', text)
  #remove multistop
  text = re.sub(r"(\.)\1+", ' ', text)
  #remove additional spaces
  text = re.sub(r"[\s]+", " ", text)
  text = re.sub(r"[\n]+", " ", text)
  
  return text

def remStopWords(Text):
  return " ".join(word for word in Text.split() if word not in stop)

def stemWords(Text):
  return " ".join(st.stem(word) for word in Text.split())

def predict_new_data():
    # Charger le modèle SVM sauvegardé
    svm_model = joblib.load('modele_svm.joblib')
    chemincsv=r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\fichier.csv"
    # Charger les données à partir du fichier CSV
    dataFrame_new = pd.read_csv(chemincsv)

    # Prétraitement du contenu des nouvelles données
    dataFrame_new['Contenu'] = dataFrame_new['Contenu'].apply(lambda Text: remStopWords(str(Text)))
    dataFrame_new['Contenu'] = dataFrame_new['Contenu'].apply(lambda Text: clean(Text))
    dataFrame_new['Contenu'] = dataFrame_new['Contenu'].apply(lambda Text: stemWords(Text))
    
    # Charger le TF-IDF vectorizer
    tfidf_vect = joblib.load('tfidf_vect.joblib')

    # Transformer les nouvelles données avec le TF-IDF vectorizer
    xvalid_tfidf_new = tfidf_vect.transform(dataFrame_new['Contenu'])

    # Extraction des caractéristiques TF-IDF pour les nouvelles données
    # tfidf_vect = TfidfVectorizer()
    # xvalid_tfidf_new = tfidf_vect.transform(dataFrame_new['Contenu'])

    # Concaténation des nouvelles caractéristiques avec les caractéristiques TF-IDF
    nouvelles_caracteristiques_new = dataFrame_new[['SL1', 'WL1', 'WL4', 'WL5', 'AMb1', 'VL1', 'VL2']].values
    xvalid_combined_new = hstack((xvalid_tfidf_new, nouvelles_caracteristiques_new))

    # Faire des prédictions sur les nouvelles données
    predictions_new = svm_model.predict(xvalid_combined_new)

    # Afficher les prédictions
    print(predictions_new)
    return predictions_new
def save_text_to_file(text):
    chemin_fichier= r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\fichier.txt"
    with open(chemin_fichier, 'w', encoding='utf-8') as file:
        file.write(text)
import pandas as pd

def extract_values_from_csv():
    # Charger les données à partir du fichier CSV
    
    chemincsv=r"C:\\Users\\XPS\\Desktop\\FerhatOumaimaQassimiIkhlas\\fichier.csv"
    dataFrame = pd.read_csv(chemincsv)

    # Accéder aux valeurs de chaque colonne
    sl1 = dataFrame['SL1'].iloc[0]
    wl1 = dataFrame['WL1'].iloc[0]
    wl5 = dataFrame['WL5'].iloc[0]
    nombre_lemmes = dataFrame['nombre_lemmes'].iloc[0]
    vl1 = dataFrame['VL1'].iloc[0]
    amb1 = dataFrame['AMb1'].iloc[0]
    vl2 = dataFrame['VL2'].iloc[0]
    wl4 = dataFrame['WL4'].iloc[0]

    return  sl1, wl1, wl5, nombre_lemmes, vl1, amb1, vl2, wl4

app = Flask(__name__)
# Route pour la page d'accueil avec le formulaire HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer la saisie utilisateur et retourner la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['texte']
        save_text_to_file(input_text)
        process_and_save_file()
        prediction = predict_new_data()
        #print("predictiion est ",prediction)
        sl1, wl1, wl5,nombre_lemmes, vl1, amb1, vl2,wl4 = extract_values_from_csv()
        #print("SL1:", sl1)
        # print("WL1:", wl1)
        # print("WL5:", wl5)
        # print("Nombre de lemmes:", nombre_lemmes)
        # print("VL1:", vl1)
        # print("AMb1:", amb1)
        # print("VL2:", vl2)
        # print("WL4:", wl4)
        # print("Classe:", prediction[0])
        resultat = {
            'texte':input_text,
            'prediction': int(prediction),
            'sl1': int(sl1),
            'wl1': int(wl1),
            'wl5': int(wl5),
            'nombre_lemmes': int(nombre_lemmes),
            'vl1': int(vl1),
            'amb1': int(amb1),
            'vl2': int(vl2),
            'wl4': int(wl4)  
            
        }
        print("reslat" ,resultat)
        # Votre logique pour extraire les valeurs à partir du CSV
        return jsonify(resultat)
        #return render_template('result.html', prediction=prediction[0], input_text=input_text,sl1=sl1,wl1=wl1,wl5=wl5,nombre_lemmes=nombre_lemmes,vl1=vl1,amb1=amb1,vl2=vl2,wl4=wl4)

if __name__ == '__main__':
    app.run(debug=True)
