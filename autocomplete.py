import pandas as pd

df = pd.read_csv('words.csv')
# Convertir le texte en minuscules
text = df['words'].str.lower().str.cat(sep=' ')
# Convertir le texte en une liste de mots
words = text.split()


def autocomplete(prefix):
    list_auto = [word for word in words if word.startswith(prefix)]
    if len(list_auto) == 2 :
        list_auto += "."
    elif len(list_auto) == 1 :
        list_auto.append(".")
        list_auto.append("?")
    else  :
        list_auto.append("tion")
        list_auto.append("ble")
        list_auto.append("ing")

    
        
    return list_auto[:3]

