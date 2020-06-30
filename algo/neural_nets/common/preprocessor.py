def remove_words(x):
    x = x.replace('@USER', '')
    x = x.replace('HTTPURL', '')
    return x


def transformer_pipeline(x):
    x = remove_words(x)
    return x
