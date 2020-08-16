# Created by Hansi at 2/28/2020

import spacy

from project_config import USER_FILLER, URL_FILLER

nlp = spacy.load('en_core_web_lg')

ignored_entities = ['WORK_OF_ART', 'DATE', 'TIME', 'QUANTITY', 'PERCENT', 'MONEY', 'ORDINAL', 'CARDINAL']
ignored_tokens = [USER_FILLER, URL_FILLER]

# spacy documentation - https://spacy.io/api/annotation#named-entities
entity_names = dict()
entity_names['PERSON'] = 'person'
entity_names['NORP'] = 'nationality'
entity_names['FAC'] = 'building'
entity_names['ORG'] = 'organisation'
entity_names['GPE'] = 'location'  # 'country'
entity_names['LOC'] = 'location'
entity_names['PRODUCT'] = 'product'  # 'object'
entity_names['EVENT'] = 'event'
entity_names['WORK_OF_ART'] = 'title'
entity_names['LAW'] = 'law'
entity_names['LANGUAGE'] = 'language'
entity_names['DATE'] = 'date'
entity_names['TIME'] = 'time'
entity_names['PERCENT'] = 'percentage'
entity_names['MONEY'] = 'money'
entity_names['QUANTITY'] = 'quantity'
entity_names['ORDINAL'] = 'ordinal'  # not in vocab
entity_names['CARDINAL'] = 'cardinal'


# Read vocabulary of trained model
def read_vocab(filepath):
    f = open(filepath, 'r', encoding='utf-8')
    vocab = f.readlines()
    f.close()
    vocab = [str.replace(word, '\n', '') for word in vocab]
    return vocab


def tokenize_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokens


# Filter unknown words to the vocabulary
# vocab - vocabulary as a list of words
# words - word list to compare
def get_unknown_tokens(vocab, words, do_lower_case):
    unknowns = set()
    for word in words:
        if do_lower_case:
            if word.lower() not in vocab:
                unknowns.add(word)
        else:
            if word not in vocab:
                unknowns.add(word)
    return unknowns


# Identify named entities in the given string
# return dictionary of entity text: label
def get_entities(str):
    dict_entities = dict()
    doc = nlp(str)
    for ent in doc.ents:
        dict_entities[ent.text] = ent.label_
    # print([(X.text, X.label_) for X in doc.ents])
    return dict_entities


# Replace vocabulary unknown words in the text by known entities
def replace_with_entities(text, vocab, tokenizer, do_lower_case):
    replaced_words = []
    new_text = text
    dict_entities = get_entities(text)
    tokens = tokenize_text(text, tokenizer)
    unknown_words = get_unknown_tokens(vocab, tokens, do_lower_case)
    # remove ignored tokens from unknown words
    unknown_words = [x for x in unknown_words if x not in ignored_tokens]
    # print("unkown words: ", unknown_words)

    for entity_word in dict_entities.keys():
        entity_words = entity_word.split()
        if dict_entities[entity_word] not in ignored_entities:
            # replace whole entity text with label if all words in entity text are unknown
            # check unknown_words contains all elements in entity_words
            if all(elem in unknown_words for elem in entity_words):
                new_text = str.replace(new_text, entity_word, entity_names[dict_entities[entity_word]])
                replaced_words.append(entity_word + '-' + entity_names[dict_entities[entity_word]])
            # if only few of words in entity text are unknown, replace them only with entity label
            else:
                for word in entity_words:
                    if word in unknown_words:
                        new_text = str.replace(new_text, word, entity_names[dict_entities[entity_word]])
                        replaced_words.append(word + '-' + entity_names[dict_entities[entity_word]])
    return new_text, replaced_words

# if __name__ == "__main__":
# # #     # text = "IMPOANT STORY DEVELOPING: twitteruser reports 5 long-term facilities have COVID-19 outbreaks in Ozaukee and Washington counties. 6 more suspected. Workers may have been transferring virus between facilities. Story leads twitteruser at 9!"
# # #     # text = "IMPORTANT STORY DEVELOPING: @USER reports 5 long-term facilities have COVID-19 outbreaks in Ozaukee and Washington Counties. 6 more suspected. Workers may have been transferring virus between facilities. Story leads @USER at 9!"
#     text = "Second case DR ðŸ‡©ðŸ‡´ The Canadian woman has not been identified, however they indicated that she is 70 years old and that she was staying with her husband in a Bayahibe hotel, according to the Minister of Public Health. #CoronaVirusRD #CoronaVirus #COVID19"
#     # model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, use_cuda=torch.cuda.is_available())
#     # tokenizer = model.tokenizer
#     # tokens = tokenise_text(text, tokenizer)
#     # print(tokens)
#
#     tokenizer = TweetTokenizer(reduce_len=True, strip_handles=False)
#     tokens = tokenize_text(text, tokenizer)
#     print(tokens)
#     vocab = read_vocab(VOCAB_PATH)
#
#     print(get_entities(text))
#
#     new_text, replaced_words = replace_with_entities(text, vocab, tokenizer, True)
#     print(new_text)
#     # print(replaced_words)
