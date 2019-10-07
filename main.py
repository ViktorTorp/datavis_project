import numpy as np
import math as math
from tqdm import tqdm
from TF_idf import TFIDF as tf_idf
from collections import defaultdict
import operator


def create_bows(file_name):
    """ Reads bog of words from file

    Read and initialises BOWs 

    Parameters
    ----------
    file_name : string
        path to file containing bows

    Returnes
    --------
    bow : dict
        dict with bag of word representation
    """
    bow = defaultdict(int)
    with open(file_name) as f:
        for line in tqdm(f):
            line = line.split()
            for word_count in line[:-1] :
                word_count = word_count.split(":")
                word, count = word_count[0],int(word_count[1])
                if word == "#label#":  # Does not include sentence labels
                    continue
                if "_" not in word:  # Only include unigrams
                    bow[word] += count
    return bow


def load_embeddings(file_name):
    """ Load embeddings

    Loads embeddings and creates a vocabulary.
    All vectors are normalized, which make the cosine_similarity more efficient.

    Parameters
    ----------
    file_name : string
        path to file containing bows

    Returnes
    --------
    vocab : list
        list of all words in vocabulary
    word_embedding : dict
        Dict with all words as keys and their vector as value
    """
    vocab = []
    word_embedding = {}
    with open(file_name) as f:
        for line in f:
            _line = line.split()
            word = _line[0]
            vector = [float(x) for x in _line[1:]]
            if len(vector) == 300:
                l2 = math.sqrt(sum([x*x for x in vector]))
                _vector = [x / l2 for x in vector]
                vocab.append(word)
                word_embedding[word] = _vector
            else:
                print("Could not create embedding for {}".format(word))
    return vocab, word_embedding


def cosine_similarity(source_vector, target_vector):
    """ Calculates cosine similarity between two vectors

    These calculates are made on the assumtion, 
    that the vector are normalized

    Parameters
    ----------
    source_vector : list
        The embdedding vector for the source word
    target_vector : list
        The embdedding vector for the target word

    Returnes
    --------
    dot_product : float
        The final cosine similaroty
    """
    assert len(source_vector) == len(target_vector)
    dot_product = np.dot(source_vector,target_vector)
    return dot_product 


def find_distances(source_vector, word_embedding):
    """ Find distances between all words and a vector
    
    Finds all the distinces from a source vector to all
    words in the word_embedding. These results are sorted descendingly
    and returned as a tuple (distance,word)

    Parameters
    ----------
    source_vector : list
        The embdedding vector for the source word
    word_embedding : dict
        Dict with all words as keys and their vector as value

    Returnes
    --------
     : list
        returens a list of tuples (distance,word) ordered
        descendingly with respect to distance
    """
    distances = [(cosine_similarity(source_vector, vector), word) for word, vector in word_embedding.items()]
    return sorted(distances, key=lambda t: t[0], reverse=True)


def find_closet(source_vector, word_embedding, target_vocab = None, number_closet = 1):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    distances = [(cosine_similarity(source_vector, vector),word) for word, vector in word_embedding.items()]
    result = sorted(distances, key=lambda t: t[0], reverse=True)

    if target_vocab:
        result = [x for x in result if x[1] in target_vocab]
    return result[:number_closet]


def create_word_pair(source_word, word_embedding):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    return (source_word, find_closet(word_embedding[source_word],word_embedding)[1])


def create_non_pivot_pair(source_domain, word, target_domain, target_vocab, word_embedding):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    try:
        source_vector = np.array(word_embedding[source_domain])
        word_vector = np.array(word_embedding[word])
        target_vector = np.array(word_embedding[target_domain])
        subtraction_vector =  word_vector - source_vector
        new_vector = subtraction_vector + target_vector
        closet_word = find_closet(new_vector,word_embedding, target_vocab, 10)

        # Remove redundant words
        stop_words = []
        words = [source_domain, word, target_domain]
        for w in words:
            stop_words.append(w)
            stop_words.append(w+"s")
            stop_words.append(w+"'s")
        cos_sim, closet_word = [(cos_sim, w) for cos_sim, w in closet_word if w.lower() not in stop_words][0]
        return (word, closet_word, cos_sim)
    except:
        return ("##NAN##", "##NAN##", 0)

def create_lexicon(source_domain, source_words, target_domains, target_domains_vocab, word_embedding):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    result = []
    for word in source_words:
        for idx, target_domain in enumerate(target_domains):
            _, pair_word, cos_sim = create_non_pivot_pair(source_domain, word, target_domain,target_domains_vocab[idx] , word_embedding)
            result.append((source_domain,word,target_domain, pair_word, cos_sim))
    return result






# Find Non-pivot words for each domain
def non_pivot_words(tfidfs):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    result = {}
    for domain, scores in tfidfs.items():
        result[domain] =  list(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
    return result



def find_words_for_presentation(non_pivots_res, tfidfs, file_name, num_words = 5):
    """ Calculates tf

    Parameters
    ----------
    bow : dict
        dict with bag of word representationf

    Returnes
    --------
    tf : dict
        Tf dict
    """
    domains = non_pivots_res.keys()
    domain_words = {}
    f = open(file_name, "w")
    f.write("Domain, word, score")
    f.write("\n")
    for domain in domains:
        other_domains = [d for d in domains if d != domain]
        temp_res = []
        temp_dict = {}
        for word, score in non_pivots_res[domain]:
            other_domain_scores = [tfidfs[d][word] for d in other_domains]

            word_dict = {d: tfidfs[d][word] for d in other_domains}
            word_dict[domain] = score
            temp_dict[word] = word_dict

            if other_domain_scores.count(0) == 4: # adjusts the number of allowed 0's in other domains
                continue
            temp_res.append((word,score, np.mean(other_domain_scores)))

        temp_res.sort(key=lambda row: row[2])
        temp_res.sort(key=lambda row: row[1], reverse = True)
        temp_res = temp_res[:num_words]
        final_words = []
        for temp_word, _, _ in temp_res:
            final_words.append(temp_word)
            for temp_domain, temp_score in temp_dict[temp_word].items():
                f.write("{}, {}, {}".format(temp_domain,temp_word,temp_score))
                f.write("\n")
        domain_words[domain] = final_words
    f.close() 
    return domain_words





if __name__ == '__main__':
    from TF_idf import TFIDF as tf_idf
    from time import time as time


    print("loading embeddings....")
    vocab, word_embedding = load_embeddings("data/vectors.vec")
    print(" - done loading")




    # tf-idf

    dvd_data = create_bows("data/processed_acl/dvd/unlabeled.review")
    book_data = create_bows("data/processed_acl/books/unlabeled.review")
    el_data = create_bows("data/processed_acl/electronics/unlabeled.review")
    kit_data = create_bows("data/processed_acl/kitchen/unlabeled.review")



    bows = [dvd_data,book_data, el_data,kit_data]
    names = ["dvd","books","electronics","kitchen"]
    print("")
    print("Tranforming domain data....")
    tfidf = tf_idf()
    tfidf_scores = tfidf.fit_transform(bows,names)
    non_pivot = non_pivot_words(tfidf_scores)

    # create file for datavis illustration about non-pivot words
    top_non_pivit = find_words_for_presentation(non_pivot,tfidf_scores, "Data_vis_tfidf.txt")
    print(" - done loading")
    print("")

    # top_non_pivit = {}
    # for domain in names:
    #     top_non_pivit[domain] = [word for word,_ in non_pivot[domain][:5]]
    all_target_domains_vocab = [bow.keys() for bow in bows]
    with open("word_maps.txt","w") as f:
        f.write("s")
        f.write(" ,")
        f.write("w")
        f.write(" ,")
        f.write("td")
        f.write(" ,")
        f.write("tw")
        f.write(" ,")
        f.write("co")
        f.write("\n")

        for idx, domain in enumerate(names):
            other_domains = [dom for dom in names if dom != domain]
            source_words = top_non_pivit[domain]
            target_domains_vocab = [all_target_domains_vocab[i] for i in range(len(all_target_domains_vocab)) if i != idx]
            start = time()
            word_pair = create_lexicon(domain, source_words , other_domains, target_domains_vocab, word_embedding)
            stop = time()
            for do, wo, ta, pw, co in word_pair:
                if co > 1:
                    co = 1.
                f.write(str(do))
                f.write(" ,")
                f.write(str(wo))
                f.write(" ,")
                f.write(str(ta))
                f.write(" ,")
                f.write(str(pw))
                f.write(" ,")
                f.write("{}".format(co))
                f.write("\n")
                print(do,",",wo,",",ta,",",pw,",", 179*co)

