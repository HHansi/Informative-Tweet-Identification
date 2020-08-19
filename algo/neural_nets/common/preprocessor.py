import re
from html.parser import HTMLParser

import demoji
import emoji
import unicodedata
import unidecode

from algo.neural_nets.common.ne_processor import preprocess_with_ne
from algo.neural_nets.models.transformers.args.args import PREPROCESS_WITH_NE
from project_config import USER_FILLER, URL_FILLER

demoji.download_codes()

# HTML parser
html_parser = HTMLParser()
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
control_char_regex = re.compile(r'[\r\n\t]+')

USER = "@USER"
URL = "HTTPURL"


def remove_words(x):
    x = x.replace(USER, '')
    x = x.replace(URL, '')
    return x


def add_emoji_text(x):
    emoji_text = demoji.findall(x)
    for em in emoji_text.keys():
        x = x.replace(em, " " + emoji_text[em] + " ")
    return x


def transformer_pipeline(x, preprocess_type):
    if "ct-bert" == preprocess_type:
        return preprocess_ct_bert(x)
    if "bert-tweet" == preprocess_type:
        return preprocess_bert_tweet()
    else:
        return preprocess(x)


def preprocess(x):
    x = remove_words(x)
    x = add_emoji_text(x)
    x = standardize_text(x)
    x = standardize_punctuation(x)
    if PREPROCESS_WITH_NE:
        x = preprocess_with_ne(x)
    return x


def preprocess_ct_bert(x):
    # clean RT tags
    text = clean_retweet_tags(x)
    # standardize
    text = standardize_text(text)

    text = replace_usernames(text, USER_FILLER)
    text = replace_urls(text, URL_FILLER)
    text = asciify_emojis(text)
    text = standardize_punctuation(text)
    text = replace_multi_occurrences(text, USER_FILLER)
    text = replace_multi_occurrences(text, URL_FILLER)
    text = remove_unicode_symbols(text)

    if PREPROCESS_WITH_NE:
        text = preprocess_with_ne(text)
    return text


def preprocess_bert_tweet(text):
    text = replace_usernames(text, "@USER")
    text = replace_urls(text, "HTTPURL")
    text = asciify_emojis(text)

    if PREPROCESS_WITH_NE:
        text = preprocess_with_ne(text)
    return text


def clean_retweet_tags(x):
    x = x.replace("RT", '')
    return x


# preprovessing for covid-twitter-bert model
def standardize_text(text):
    """
    1) Escape HTML
    2) Replaces some non-standard punctuation with standard versions.
    3) Replace \r, \n and \t with white spaces
    4) Removes all other control characters and the NULL byte
    5) Removes duplicate white spaces
    """
    # escape HTML symbols
    text = html_parser.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace('…', '...')
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    # remove all remaining control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text.strip()


def replace_usernames(text, filler):
    # USER tag is a marker used internally. use filler instead
    text = text.replace(USER, f'{filler}')
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text


def replace_urls(text, filler):
    # URL is a marker used internally. use filler instead
    text = text.replace(URL, filler)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text


def asciify_emojis(text):
    """
    Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
    For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text


def standardize_punctuation(text):
    return ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in text])


def replace_multi_occurrences(text, filler):
    """Replaces multiple occurrences of filler with n filler"""
    # only run if we have multiple occurrences of filler
    if text.count(filler) <= 1:
        return text
    # pad fillers with whitespace
    text = text.replace(f'{filler}', f' {filler} ')
    # remove introduced duplicate whitespaces
    text = ' '.join(text.split())
    # find indices of occurrences
    indices = []
    for m in re.finditer(r'{}'.format(filler), text):
        index = m.start()
        indices.append(index)
    # collect merge list
    merge_list = []
    for i, index in enumerate(indices):
        if i > 0 and index - old_index == len(filler) + 1:
            # found two consecutive fillers
            if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                # extend previous item
                merge_list[-1][1] = index
                merge_list[-1][2] += 1
            else:
                # create new item
                merge_list.append([old_index, index, 2])
        old_index = index
    # merge occurrences
    if len(merge_list) > 0:
        new_text = ''
        pos = 0
        for (start, end, count) in merge_list:
            new_text += text[pos:start]
            new_text += f'{count} {filler}'
            pos = end + len(filler)
        new_text += text[pos:]
        text = new_text
    return text


def remove_unicode_symbols(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
    return text
