#!/usr/bin/env python
# coding: utf-8

# ## Programming Assignment 2: Query Processor
# **Student**: Sarah Oliveira Elias
# 
# **Registration**: 2018048478

import sys
import argparse
import nltk
import multiprocessing
import mmap
import math
import os
import itertools

# returns the number of tokens for each document stored in 'document_index' file
def get_all_docs_sizes():
    
    global index_dir
    doc_sizes = []
    doc_sizes.append(0)
    
    with open(index_dir + '/document_index.bin', 'rb') as f:
        data = f.read()
        
    data = data.decode('utf-8')
    i = 0
    for i in range(len(data)):
        
        value = ''
        if data[i] == ',':
            i+=1
            while(data[i] != '|'):
                value += data[i]
                i+=1
            doc_sizes.append(int(value))
        i+=1
    return doc_sizes

# returns number of documents in corpus
def get_corpus_size():
    global index_dir
    
    with open(index_dir + '/document_index.bin', 'rb') as f:
        data = b''
        byte = f.read(1)
        while (byte != b'|'):
            data += byte
            byte = f.read(1)
    
    return int(data.decode('utf-8')) # 4641784 

# reads all queries from file passed in command line
def read_queries():
    global queries_file
    
    with open(queries_file, 'r') as f:
        lines = f.readlines()
    
    for i in range(len(lines)-1):
        lines[i] = lines[i][:-1]

    return lines

# tokenizes query, removes stopwords and does stemming
def process_query(query):
    
    # for removing punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r"[^\W\d_']+")

    # convert text to array of alphaanumeric tokens
    tokens = tokenizer.tokenize(query)
    
    # stemming algorithm
    ps = nltk.stem.PorterStemmer()
    
    # selects only english stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # removing the stopwords and stemming them
    filtered_tokens = []
    for w in tokens:
        
        # filter words from other alphabets
        if w.isascii():
            if w not in stopwords:
                stemmedWord = ps.stem(w)
                filtered_tokens.append(stemmedWord)
    
    return filtered_tokens

# reads the 'term_lexicon' n returns postions of each token's posting in 'index' file
def get_posting_positions(query):
    global index_dir
    offsets = []
    
    with open (index_dir + '/term_lexicon.bin', 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            
            for q in query:
                token = '|' + q
                token = token.encode('utf-8')
        
                # find token in lexicon
                idx = m.find(token)
                if idx != -1:

                    # get offsets
                    m.seek(idx+1)
                    while(1):
                        if (m.read(1) == b'|'):
                            while(1):
                                if (m.read(1) == b'|'):
                                    t = m.tell()
                                    m.seek(idx + len(token))
                                    offsets.append(m.read(t - idx - len(token) - 1).decode('utf-8'))
                                    break
                            break
    
    offsets_matrix = []
    for o in offsets:
        offsets_matrix.append([int(o.split('|')[0]), int(o.split('|')[1])])
        
    return offsets_matrix

# finds each posting in 'index' file and returns them in an array
def get_postings(postings_pos):
    
    postings = []
    tuples = []
    
    with open(index_dir + '/inverted_index.bin', 'rb') as f:
        
        for pos in postings_pos:
       
            f.seek(pos[0])
            posting = f.read(pos[1] - pos[0]).decode('utf-8')

            posting = posting[1:-1]
            docs = posting.split('][')
            postings.append(docs)
    
    return postings   

# creates a hash map for docid and its respective values related to a list of posting
def create_hash_map(postings_array):
    
    hash_map = dict()
    max_len = 1
    for i,posting in enumerate(postings_array):
        if(i == 0):
            for p in posting:
                docid, value = p.split(', ')
                hash_map[docid] = [(int(value),i)]
        else:
            for p in posting:
                docid, value = p.split(', ')
                if (docid in hash_map.keys()):
                    hash_map[docid].append((int(value),i)) 
           
    return hash_map


# TF-IDF - returns the TF-IDF for a given doc and a given query
# doc_value contains the number of appearances, in the given doc, of each term in the given query
# uses the sum of all results as the final result 
def TFIDF(docid, doc_values, postings_len):
    
    global doc_sizes, corpus_size
    
    doc_size = doc_sizes[int(docid)]
    
    score = 0
    for i, v in enumerate(doc_values):
        
        doc_v = v[0]
        p_idx = v[1]
        
        # TF = (Number of times term t appears in a document) / (Total number of terms in the document)
        tf = doc_v / doc_size

        # IDF = log_2(Total number of documents / Number of documents with term t in it)
        idf = corpus_size / postings_len[p_idx]
        idf_log = math.log2(idf)
        
        # TF-IDF = TF(t) * IDF(t)
        score += tf*idf_log
        
    return score

# performs DAAT and ranks each doc using 'TF-IDF'
def DAAT_TFIDF(postings_array):
 
    scores = dict()
    
    hash_map = create_hash_map(postings_array)
    
    postings_len = []
    for p in postings_array:
        postings_len.append(len(p))
    
    for docid in hash_map:
        # makes a conjunctive operation, meaning that a doc needs to have all terms to be selected
        if (len(hash_map[docid]) == len(postings_array) ):
            scores[docid] = TFIDF(docid, hash_map[docid], postings_len)
            
    return scores

# BM25 - returns the BM25 for a given doc and a given query
# doc_value contains the number of appearances, in the given doc, of each term in the given query
# uses the sum of all results as the final result 
def BM25(docid, doc_values, idfs, avg_sizes):
    
    global doc_sizes
    
    # standard values in literature
    k1 = 1.2
    b = 0.75
    
    score = 0
    for i, v in enumerate(doc_values):
        
        doc_v = int(v[0])
        p_idx = v[1]
        
        # apply BM25 equation
        doc_denominator = doc_v + k1*( 1 -b + b*doc_sizes[int(docid)]/avg_sizes )
        score += idfs[p_idx] * ( (doc_v * (k1 + 1) ) / doc_denominator )
        
    return score 

# performs DAAT and ranks each doc using 'BM25'
def DAAT_BM25(postings_array, avg_sizes):
    
    global doc_sizes, corpus_size, ranker
    
    hash_map = create_hash_map(postings_array)
    
    idfs = []
    scores = dict()
    
    # get terms IDF's
    for posting in postings_array:
        idfs.append(math.log(1 + (corpus_size - len(posting) + 0.5)/(len(posting) + 0.5)))                
    
    # calculate score for each document - only if doc is present in at least one of the postings
    for docid in hash_map:   
        # makes a conjunctive operation, meaning that a doc needs to have all terms to be selected
        if (len(hash_map[docid]) == len(postings_array) ):
            scores[docid] = BM25(docid, hash_map[docid], idfs, avg_sizes)
    
    return scores

# THREADS - each thread receives and processes a query 
def task(args):
    
    avg_sizes = args[0]
    q = args[1]
    
    # process query
    query = process_query(q)
    query.sort()
    postings_pos = get_posting_positions(query)
    postings_array = get_postings(postings_pos)
    
    # calculate scores through DAAT
    if (ranker == 'TFIDF'): scores = DAAT_TFIDF(postings_array)
    else: scores = DAAT_BM25(postings_array, avg_sizes)

    # order scores and pic top 10
    sorted_scores = dict(sorted(scores.items(), reverse=True, key=lambda x: x[1]))
    first_10_items = dict(itertools.islice(sorted_scores.items(), 10))

    return (first_10_items, q)


# creates a process pool
def create_threads(queries):
    
    global corpus_size, ranker, doc_sizes

    task_args = []
    
    # if BM25 is passed then avg_sizes is needed
    if ranker == 'BM25':
        avg_sizes = 0
        for i in range(1, corpus_size+1):
            avg_sizes += doc_sizes[i]
        avg_sizes /= corpus_size

    else: avg_sizes = 0
    
    # create args for each thread
    for q in queries:
        task_args.append([(avg_sizes, q)])

    # create pool and map the tasks to the process pool asynchronously
    with multiprocessing.Pool() as pool:
        async_results = [pool.map_async(task, arg) for arg in task_args]
        # get result
        results = [r.get() for r in async_results]

    return results

# prints the result to standard output
def print_results(results):
    for r in results:
        print('{ "Query": "' + r[0][1] + '",\n  "Results: [')
        for key, value in sorted(r[0][0].items(), reverse=True, key=lambda x: x[1]):
            print('    { "ID": "' + str(key) + '",\n      "Score": '+ str(round(value,3)) + ' },')
        print('   ] },')
        
        
# -i < INDEX >: the path to an index directory.
# -q < QUERIES >: the path to a file with the list of queries to process
# -r < RANKER >: a string informing the ranking function (either “TFIDF” or “BM25”) to be used to score documents for each query.
def main():
    
    # GET INITIAL VARIABLES
    global index_dir, queries_file
    
    queries = read_queries()
    
    # uses 5 threads at a time - each one processes one query
    for i in range(0, len(queries), 5):
        results = create_threads(queries[i:i+5])
        print_results(results)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-i',
        dest='index',
        action='store',
        required=True,
        type=str,
        help='the path to an index directory'
    )
    parser.add_argument(
        '-q',
        dest='queries_file',
        action='store',
        required=True,
        type=str,
        help='the path to a file with the list of queries to process'
    )
    parser.add_argument(
        '-r',
        dest='ranker',
        action='store',
        required=True,
        type=str,
        help='a string informing the ranking function (either “TFIDF” or “BM25”) to be used to score documents for each query'
    )

    # GET INITIAL VARIABLES
    args = parser.parse_args()
    index_dir = args.index
    queries_file = args.queries_file
    ranker = args.ranker
    
    doc_sizes = get_all_docs_sizes()
    corpus_size = get_corpus_size()
    
    main()
