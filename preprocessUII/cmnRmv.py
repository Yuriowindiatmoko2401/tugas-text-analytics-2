import re, unicodedata

def rmNon_Ascii(text): # membuang character ascii atau emoticon 
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def rmURLs(text): # membuang url
    return re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)

def rmPunc(text): # membuang tanda baca
    return re.sub(r'[^\w]|_',' ',text)

def rmDigit_string(text): # membuang digit didalam string 
    return re.sub("\S*\d\S*", "", text).strip()

def rmDigitnumbers(text): # membuang digit dan angka
    return re.sub(r"\b\d+\b", " ", text)

def rmHashtag(text): # membuang hashtag
    return re.sub(r"#(\w+)", ' ', text, flags=re.MULTILINE)

def rmMention(text): # membuang mention
    return re.sub(r"@(\w+)", ' ', text, flags=re.MULTILINE)

def rmXML(text): # membuang xml character
    return re.sub("&(?:#([0-9]+)|#x([0-9a-fA-F]+)|([0-9a-zA-Z]+));"," ",text)

def rmRT(text): # membuang retweet atau RT
    return re.sub("\s*RT\s*@[^:]*"," ",text)

def lowercase(text): # casefolding menjadi huruf kecil
    return text.lower()

def rmAdditionalWs(text): # membuang spasi2 tambahan
    return re.sub('[\s]+', ' ', text)

