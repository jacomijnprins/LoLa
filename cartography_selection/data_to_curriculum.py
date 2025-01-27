import pandas as pd
from assigntools.LoLa.read_nli import snli_jsonl2dict
from assigntools.LoLa.sen_analysis import spacy_process_sen2tok, display_doc_dep
from data_utils_glue import convert_string_to_unique_number

def load_data():
    cartography_data = pd.read_json('data/snli_roberta_0_6_data_map_coordinates.jsonl', lines=True)

    SNLI, S2A = snli_jsonl2dict('data/snli_1.0') 
    print(f"Length of the SNLI dataset with the wrong labels: {len(SNLI['train'])}")

    SNLI_with_wrong, S2A = snli_jsonl2dict('data/snli_1.0', clean_labels=False)
    print(f"Length of the clean SNLI dataset: {len(SNLI_with_wrong['train'])}")

    wrong_keys = list(set(list(SNLI_with_wrong['train'].keys())).difference(list(SNLI['train'].keys())))

    # the guid values for the data instances that have a bad gold label
    wrong_indeces = []
    for key in wrong_keys:
        wrong_indeces.append(convert_string_to_unique_number(key))

    # deleting the wrong indeces from the cartography set
    for ID in wrong_indeces:
        cartography_data = cartography_data[cartography_data['guid'] != ID]

    # create different SNLI dictionary with the correct keys
    SNLI_dictionary = {}
    for key in SNLI['train'].keys():
        new_key = convert_string_to_unique_number(key)
        SNLI_dictionary[new_key] = SNLI['train'][key]

    return SNLI_dictionary, cartography_data

def create_categories():
    variability_split = 0.2
    correctness_split = 4

    ambiguous = cartography_data[cartography_data['variability']>=variability_split]
    easy_to_learn = cartography_data[(cartography_data['variability']<variability_split) & (cartography_data['correctness']>= correctness_split)] 
    hard_to_learn = cartography_data[(cartography_data['variability']<variability_split) & (cartography_data['correctness']< correctness_split)] 

    return ambiguous, easy_to_learn, hard_to_learn

def make_curriculum(name):
    guids = choose_guids_guids()

    curriculum = []

    for guid in guids:
        instance = SNLI_dictionary[guid]
        if instance['g'] == 'entailment':
            label = 0
        elif instance['g'] == 'neutral':
            label = 1
        elif instance['g'] == 'contradiction':
            label = 2
        curriculum.append([instance['guid'], instance['p'], instance['h'], label])

    curriculum = pd.DataFrame(curriculum, columns=['guid', 'premise', 'hypothesis', 'label'])

    curriculum.to_csv(f'curricula/{name}.csv')

def choose_guids():
    """
    This is the function that can be changed to create different 
    """
    ambiguous, easy_to_learn, hard_to_learn = create_categories()
    
    guids = list(ambiguous.sort_values('variability')['guid'][-2000:])

    return guids

if name == "__main__":
    SNLI_dictionary, cartography_data = load_data()

    make_curriculum('most_ambiguous')

