
def get_dates_in_month(year, month, time_zone):
    num_days = monthrange(year, month)[1]
    first_date_of_month = datetime.datetime(year,month,1, tzinfo=time_zone)
    last_date_of_month =  datetime.datetime(year,month,num_days, tzinfo=time_zone)
    return get_dates_inrange(first_date_of_month, last_date_of_month)

def get_dates_inrange(date1, date2):
    if (not isinstance(date1, datetime.datetime)) | (not isinstance(date2, datetime.datetime)):
        return "date1 and date2 should be of type datetime.date"
    num_days = (date2 - date1).days + 1
    date_list = [date1 + datetime.timedelta(days=x) for x in range(0, num_days)]
    return date_list

def remove_nonAlphaNumeric(s: str):
    return re.sub(r'\W+', '', s)

def tokenize(raw_text: str):
    lower_tokens = [w.lower() for w in word_tokenize(raw_text)]
    refined_tokens = [remove_nonAlphaNumeric(w) for w in lower_tokens]
    return list(filter(None, refined_tokens))

def w2v_vectorize(raw_text: str, model):
    words = tokenize(raw_text)
    word_vectors = model.wv
    vector = [0] * model.wv.vector_size
    for word in words:
        if word in word_vectors.vocab:
            vector += word_vectors.get_vector(word)
    vector = [np.round(d, 4) for d in vector]
    return vector