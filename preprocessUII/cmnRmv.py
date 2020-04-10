import re, unicodedata

def rmNon_Ascii(text):
    """membuang character ascii atau emoticon 
    
    :return: clean string from ascii character
    :rtype: string
    """
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def rmURLs(text):
    """membuang url
    
    :return: clean string from url
    :rtype: string
    """
    return re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)


def rmPunc(text): 
    """membuang tanda baca
    
    :return: clean string from punctuation
    :rtype: string
    """
    return re.sub(r'[^\w]|_',' ',text)

def rmDigit_string(text): 
    """membuang digit didalam string 
    
    :return: clean string from digit
    :rtype: string
    """
    return re.sub("\S*\d\S*", "", text).strip()

def rmDigitnumbers(text): 
    """membuang digit dan angka
    
    :return: clean string from digit n numbers
    :rtype: string
    """
    return re.sub(r"\b\d+\b", " ", text)

def rmHashtag(text):
    """membuang hashtag
    
    :return: clean string from hashtag
    :rtype: string
    """
    return re.sub(r"#(\w+)", ' ', text, flags=re.MULTILINE)

def rmMention(text):
    """membuang mention
    
    :return: clean string from mention @
    :rtype: string
    """
    return re.sub(r"@(\w+)", ' ', text, flags=re.MULTILINE)

def rmXML(text):
    """membuang xml character
    
    :return: clean string from xml character
    :rtype: string
    """
    return re.sub("&(?:#([0-9]+)|#x([0-9a-fA-F]+)|([0-9a-zA-Z]+));"," ",text)

def rmRT(text):
    """membuang retweet atau RT
    
    :return: clean string from RT
    :rtype: string
    """
    return re.sub("\s*RT\s*@[^:]*"," ",text)

def lowercase(text): 
    """casefolding menjadi huruf kecil
    
    :return: lower case string
    :rtype: string
    """
    return text.lower()

def rmAdditionalWs(text):
    """membuang spasi2 tambahan
    
    :return: clean string whitespace unnecessary
    :rtype: string
    """
    return re.sub('[\s]+', ' ', text)

