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


def find_closet(source_vector, word_embedding, target_vocab=None, number_closet=1):
    """ Find number of closet words

    Uses find_distances to find the closest word vectors

    Parameters
    ----------
    source_vector : list
        The embdedding vector for the source word
    word_embedding : dict
        Dict with all words as keys and their vector as value
    target_vocab : list (optional)
        list of strings, with a targets vocabulary
        (default is None)
    number_closet : int (optional)
        number of closest words
        (default is 1)

    Returnes
    --------
    closest : list
        returens a list with number_closest tuples (distance,word) ordered
        descendingly with respect to distance
    """
    result = find_distances(source_vector, word_embedding)
    closest = list()
    idx = 0
    while len(closest) < number_closet:
        if result[idx][1] in target_vocab:
            closest.append(result[idx])
        idx += 1
    return closest



def create_word_pair(source_word, word_embedding, target_vocab=None):
    """ Creates a word pair, with source_word and the closest word

    Uses find_closet to find the single closest words vector

    Parameters
    ----------
    source_vector : list
        The embdedding vector for the source word
    word_embedding : dict
        Dict with all words as keys and their vector as value
    target_vocab : list (optional)
        list of strings, with a targets vocabulary
        (default is None)

    Returnes
    --------
    closest : tuple
        A tuple containing the source_word and the word which is closest
        (source_word, closest_word)
    """
    return (source_word, find_closet(word_embedding[source_word],word_embedding, target_vocab)[1])


def create_non_pivot_pair(source_domain, word, target_domain, target_vocab, word_embedding):
    """ Creates a non pivot word pair between two domains. 

    Uses the find_closest to find the closest words, and then removes words
    which are either related to the source_domain, word or the target_domain

    Parameters
    ----------
    source_domain : string
        The name of the source domain
    word : string
        the word from the source domain, one want to find a
        non pivot pair to
    target_domain : string
        The name of the target domain
    target_vocab : list
        list of strings, with a targets vocabulary
    word_embedding : dict
        Dict with all words as keys and their vector as value

    Returnes
    --------
     : tuple
        a tuple containing the (word, closest_word, cosine_similarity)
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
    """ Create a lexicon between source words and to several target_vocabs

    Create a lexicon between a source domain and one or several
    target domains. 

    Parameters
    ----------
    source_domain : string
        The name of the source domain
    source_words : list
        list of words from the source domain, one want to find a
        non pivot pair to
    target_domains : list
        List of the name of the target domains
    target_domains_vocab : list
        List of lists with strings, i.e. a list of target vocabularies
    word_embedding : dict
        Dict with all words as keys and their vector as value

    Returnes
    --------
    result : list
        A list containing the different lexicons,
        i.e. a list of create_non_pivot_pair outputs
    """

    result = []
    for word in source_words:
        for idx, target_domain in enumerate(target_domains):
            _, pair_word, cos_sim = create_non_pivot_pair(source_domain, word, target_domain,target_domains_vocab[idx] , word_embedding)
            result.append((source_domain,word,target_domain, pair_word, cos_sim))
    return result


def non_pivot_words(tfidfs):
    """ Find Non-pivot words for each domain
    
    Finds non-pivot words based on tf-idf scores

    Parameters
    ----------
    tfidfs : dict
        dict with domains as keys and they tf-idf dicts as value

    Returnes
    --------
    result: dict
        A dict with domains as keys and a list tuples(word, tf-idf scores)
        sorted descendingly with respect to tf-idf scores
    """

    result = {}
    for domain, scores in tfidfs.items():
        result[domain] =  list(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
    return result



def find_words_for_presentation(non_pivots_res, tfidfs, file_name, num_words = 5):
    """ Find words for datavis presentation

    Parameters
    ----------
    non_pivots_res : dict
        A dict with domains as keys and a list tuples(word, tf-idf scores)
        sorted descendingly with respect to tf-idf scores
        i.e. result from non_pivot_words
    tfidfs : dict
        dict with domains as keys and they tf-idf dicts as value
    file_name : string
        path to file the new file with the results
    num_words : int (optional)
        number of words form each lexicon
        (default is 5)

    Returnes
    --------
    domain_words : dict
        A dict with domains as keys and a list of domain words.
    """

    domains = non_pivots_res.keys()
    domain_words = {}
    f = open(file_name, "w")
    f.write("Domain, word, score")
    f.write("\n")
    for domain in domains:
        # other_domains = [d for d in domains if d != domain]
        temp_res = []
        idx = 0
        # temp_dict = {}
        while len(temp_res) < num_words:
        # for word, score in non_pivots_res[domain]:
            #other_domain_scores = [tfidfs[d][word] for d in other_domains]
            word, score = non_pivots_res[domain][idx]
            word_dict = {d: tfidfs[d][word] for d in domains}
            idx += 1
            #word_dict[domain] = score
            #temp_dict[word] = word_dict

            if list(word_dict.values()).count(0) < 4: # adjusts the number of allowed 0's in other domains
                temp_res.append(word)
                for temp_domain, temp_score in word_dict.items():
                    f.write("{}, {}, {}".format(temp_domain,word,temp_score))
                    f.write("\n")
        domain_words[domain] = temp_res

        # temp_res.sort(key=lambda row: row[2])
        # temp_res.sort(key=lambda row: row[1], reverse = True)
        # temp_res = temp_res[:num_words]
        # final_words = []
        # for temp_word, _, _ in temp_res:
        #     final_words.append(temp_word)
        #     for temp_domain, temp_score in temp_dict[temp_word].items():
        #         f.write("{}, {}, {}".format(temp_domain,temp_word,temp_score))
        #         f.write("\n")
        # domain_words[domain] = final_words
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
    top_non_pivit = find_words_for_presentation(non_pivot,tfidf_scores, "results/Data_vis_tfidf.txt")
    print(" - done transforming")
    print("")

    # top_non_pivit = {}
    # for domain in names:
    #     top_non_pivit[domain] = [word for word,_ in non_pivot[domain][:5]]
    all_target_domains_vocab = [bow.keys() for bow in bows]
    with open("word_maps.txt","w") as f:
        f.write("source_domain")
        f.write(" ,")
        f.write("source_word")
        f.write(" ,")
        f.write("target_domain")
        f.write(" ,")
        f.write("target_word")
        f.write(" ,")
        f.write("cosine_similarity")
        f.write("\n")

        for idx, domain in enumerate(names):
            print("Creating lexicon with {} as domain".format(domain))
            other_domains = [dom for dom in names if dom != domain]
            source_words = top_non_pivit[domain]
            target_domains_vocab = [all_target_domains_vocab[i] for i in range(len(all_target_domains_vocab)) if i != idx]
            start = time()
            word_pair = create_lexicon(domain, source_words , other_domains, target_domains_vocab, word_embedding)
            stop = time()
            for sd, sw, td, tw, cs in word_pair:
                if cs > 1:
                    cs = 1.
                f.write(str(sd))
                f.write(" ,")
                f.write(str(sw))
                f.write(" ,")
                f.write(str(td))
                f.write(" ,")
                f.write(str(tw))
                f.write(" ,")
                f.write("{}".format(cs))
                f.write("\n")
                print(sd,",",sw,",",td,",",tw,",", cs)
            print("- Done creating lexicon with {} as domain".format(domain))
    print("Done writing results to {}".format("results/Data_vis_tfidf.txt"))
