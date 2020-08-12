import demoji
demoji.download_codes()


def remove_words(x):
    x = x.replace('@USER', '')
    x = x.replace('HTTPURL', '')
    return x


def add_emoji_text(x):
    emoji_text = demoji.findall(x)
    for em in emoji_text.keys():
        x = x.replace(em, " " + emoji_text[em] + " ")
    return x


def transformer_pipeline(x):
    x = remove_words(x)
    x = add_emoji_text(x)
    return x
