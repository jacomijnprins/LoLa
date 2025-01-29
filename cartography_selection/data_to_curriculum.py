import pandas as pd
import os
import random
from assigntools.LoLa.read_nli import snli_jsonl2dict
# from assigntools.LoLa.sen_analysis import spacy_process_sen2tok, display_doc_dep
from data_utils_glue import convert_string_to_unique_number

def str_to_int(row):
    """
    Function to change from string label to integer label
    """ 
    if row == 'entailment':
        label = 0
    elif row == 'neutral':
        label = 1
    elif row == 'contradiction':
        label = 2
    return label

def load_data():
    """
    This function loads the cartography and SNLI data and gets it both to the same length.
    Then adds guid's to both datasets to link instances.
    Then converts SNLI to dataframe with right columns.
    """
    cartography_data = pd.read_json('data/snli_roberta_0_6_data_map_coordinates.jsonl', lines=True)

    if os.path.exists('data/SNLI.csv'):
        SNLI_dataframe = pd.read_csv('data/SNLI.csv', index_col=False)
        cartography_data = pd.merge(cartography_data, SNLI_dataframe[['guid', 'label']], how='right', left_on='guid', right_on='guid')
        return SNLI_dataframe, cartography_data
    
    SNLI, S2A = snli_jsonl2dict('data/snli_1.0') 
    SNLI_with_wrong, S2A = snli_jsonl2dict('data/snli_1.0', clean_labels=False)

    wrong_keys = list(set(list(SNLI_with_wrong['train'].keys())).difference(list(SNLI['train'].keys())))

    # the guid values for the data instances that have a bad gold label
    wrong_ids = []
    for key in wrong_keys:
        wrong_ids.append(convert_string_to_unique_number(key))

    # deleting the wrong indeces from the cartography set
    for guid in wrong_ids:
        cartography_data = cartography_data[cartography_data['guid'] != guid]

    # create different SNLI dictionary with the correct keys
    SNLI_dictionary = {}
    for key in SNLI['train'].keys():
        new_key = convert_string_to_unique_number(key)
        SNLI_dictionary[new_key] = SNLI['train'][key]

    # convert SNLI dictionary to dataframe
    SNLI_dataframe = pd.DataFrame(SNLI_dictionary.items(), columns=['guid', 'dictionaries'])
    SNLI_dataframe = SNLI_dataframe.join(pd.json_normalize(SNLI_dataframe.dictionaries))[["guid", "p", "h", "g"]]
    SNLI_dataframe = SNLI_dataframe.rename(columns={"p": "premise", "h": "hypothesis", "g": "label"})
    SNLI_dataframe['label']= SNLI_dataframe.label.apply(lambda row: str_to_int(row))

    # save the correct SNLI dataframe and rewrite cartography data
    SNLI_dataframe.to_csv(f'data/SNLI.csv', index=False)
    with open('data/snli_roberta_0_6_data_map_coordinates.jsonl', "w") as f:
        f.write(cartography_data.to_json(orient='records', lines=True, force_ascii=False))

    cartography_data = pd.merge(cartography_data, SNLI_data[['guid', 'label']], how='right', left_on='guid', right_on='guid')

    return SNLI_dataframe, cartography_data


def create_categories():
    """
    Takes the SNLI dataset and splits into the three catagories:
    easy-to-learn, ambiguous, hard-to-learn
        - Columns: guid - index - confidence - variability - correctness - label
    """
    variability_split = 0.2
    correctness_split = 4

    ambiguous = cartography_data[cartography_data['variability']>=variability_split]
    easy_to_learn = cartography_data[(cartography_data['variability']<variability_split) & (cartography_data['correctness']>= correctness_split)] 
    hard_to_learn = cartography_data[(cartography_data['variability']<variability_split) & (cartography_data['correctness']< correctness_split)] 

    return ambiguous, easy_to_learn, hard_to_learn


def make_curriculum(name):
    """
    - Calls on function to create the cartography categories
    - Calls on function to find the guids for the curriculum
    - Saves the curriculum as a .csv file
        - Columns: guid - premise - hypothesis - label
    """
    ambiguous, easy_to_learn, hard_to_learn = create_categories()
    
    guids = mix_categories_balanced(ambiguous, easy_to_learn, hard_to_learn) # change the guid function

    curriculum = SNLI_data[(SNLI_data['guid'].isin(guids))]
    print(curriculum['label'].value_counts())

    curriculum.to_csv(f'curricula/{name}.csv')


"""
Functions that return the guids for a specific curriculum
    - Name of function is the name of the file under which it gets saved
"""

def most_ambiguous(ambiguous):
    return list(ambiguous.sort_values('variability')['guid'][-2000:])


def most_ambiguous_balanced(ambiguous):
    # sorted = ambiguous.sort_values('variability')

    entailment = ambiguous[ambiguous['label']==0].sort_values('variability')['guid'][-667:]
    neutral = ambiguous[ambiguous['label']==1].sort_values('variability')['guid'][-667:]
    contradiction = ambiguous[ambiguous['label']==2].sort_values('variability')['guid'][-667:]

    guids = pd.concat([entailment, neutral, contradiction])

    return ambiguous[ambiguous['guid'].isin(list(guids))].sort_values('variability')['guid']


def most_hard_to_learn_balanced(hard_to_learn):
    entailment = hard_to_learn[hard_to_learn['label']==0].sort_values('variability')['guid'][-667:]
    neutral = hard_to_learn[hard_to_learn['label']==1].sort_values('variability')['guid'][-667:]
    contradiction = hard_to_learn[hard_to_learn['label']==2].sort_values('variability')['guid'][-667:]

    guids = pd.concat([entailment, neutral, contradiction])

    return hard_to_learn[hard_to_learn['guid'].isin(list(guids))].sort_values('variability')['guid']


def mix_categories(ambiguous, easy_to_learn, hard_to_learn):
    # print(len(sorted(list(ambiguous.index))))
    ambiguous_guids = ambiguous['guid'][random.sample(sorted(list(ambiguous.index)), 667)]
    easy_to_learn_guids = easy_to_learn['guid'][random.sample(sorted(list(easy_to_learn.index)), 667)]
    hard_to_learn_guids = hard_to_learn['guid'][random.sample(sorted(list(hard_to_learn.index)), 667)]

    return pd.concat([ambiguous_guids, easy_to_learn_guids, hard_to_learn_guids])

def mix_categories_balanced(ambiguous, easy_to_learn, hard_to_learn):
    entailment = ambiguous['guid'][random.sample(sorted(list(ambiguous[ambiguous['label']==0].index)), 222)]
    neutral = ambiguous['guid'][random.sample(sorted(list(ambiguous[ambiguous['label']==1].index)), 222)]
    contradiction = ambiguous['guid'][random.sample(sorted(list(ambiguous[ambiguous['label']==2].index)), 222)]
    ambiguous_guids = pd.concat([entailment, neutral, contradiction])

    entailment = easy_to_learn['guid'][random.sample(sorted(list(easy_to_learn[easy_to_learn['label']==0].index)), 222)]
    neutral = easy_to_learn['guid'][random.sample(sorted(list(easy_to_learn[easy_to_learn['label']==1].index)), 222)]
    contradiction = easy_to_learn['guid'][random.sample(sorted(list(easy_to_learn[easy_to_learn['label']==2].index)), 222)]
    easy_to_learn_guids = pd.concat([entailment, neutral, contradiction])

    entailment = hard_to_learn['guid'][random.sample(sorted(list(hard_to_learn[hard_to_learn['label']==0].index)), 222)]
    neutral = hard_to_learn['guid'][random.sample(sorted(list(hard_to_learn[hard_to_learn['label']==1].index)), 222)]
    contradiction = hard_to_learn['guid'][random.sample(sorted(list(hard_to_learn[hard_to_learn['label']==2].index)), 222)]
    hard_to_learn_guids = pd.concat([entailment, neutral, contradiction])

    return pd.concat([ambiguous_guids, easy_to_learn_guids, hard_to_learn_guids])


if __name__ == "__main__":
    global SNLI_data, cartography_data
    SNLI_data, cartography_data = load_data()

    # ambiguous, easy_to_learn, hard_to_learn = create_categories()
    # print(len(mix_categories(ambiguous, easy_to_learn, hard_to_learn)))

    make_curriculum('mix_categories_balanced') # change this name and don't forget to change the guid function 
