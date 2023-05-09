#!/usr/bin/env python
# coding: utf-8
import sys
import resource 
import argparse
import pandas as pd
import nltk 
import psutil
import multiprocessing
import time
import math
import os
import itertools
import json

nltk.download('stopwords')
nltk.download('punkt')

# # Part 1 - Partial Indexes

# ### Pre Processing
# 
# Receives a text, tokenizes it, removes stopwords from it and does stemming
def process_text(text):
    
    # for removing punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r"[^\W\d_']+")

    # convert text to array of alphaanumeric tokens
    tokens = tokenizer.tokenize(text)
    
    # stemming algorithm
    ps = nltk.stem.PorterStemmer()
    
    # selects only english stopwords - not working
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


# ### Insert token in index
# 
# If token does not exists in index yet, creates a new posting for that token. Also keeps the number of times the token appears in the document.
def insert_to_index(docid, token, index):
    if token not in index:
        # create new posting for the new token
        index[token] = [[docid,1]]
    
    elif index[token][-1][0] == docid:
        # doc already exists in token's list - increase token's frequency in doc
        index[token][-1][1] += 1
    
    else:
        index[token].append([docid, 1])


# ### Indexer
# This pre-processes every component of the document calling proccess_text function for each one of them. Then, combines the processed tokens into one array and inserts them into the index using insert_to_index function.
def indexer(chunk, index, docs_idx):
    
    for c in chunk:
        d = json.loads(c)
        
        tokens = process_text(d["text"])

        docs_idx.append((d["id"], len(tokens)))
            
        for t in tokens:
            
            insert_to_index(int(d["id"]), t, index)


            
# ### Read chunks of the file
def read_chunk(start, chunk_size):
   
    global corpus_path

    with open(corpus_path, 'rb') as f:
        f.seek(start)

        # Read the file in chunks
        chunk = list(itertools.islice(f, chunk_size))
        
        ptr = f.tell()
           
    return chunk, ptr



# ### Write partial indexes to disk
def write_index(index, index_name):
    with open(index_name, "w", encoding="utf-8") as f:
        for key, value in index.items():
            f.write(f"'{key}': {value},\n")


# ### Memory 
# Checks if current memory usage has reach the limit
def checkMemoryFull():
     
    global memory_limit
    process = psutil.Process()
    limit = (memory_limit * 0.9) / 4
    return process.memory_info().rss >= limit


                    
def task(idx):
    
    global file_values
    
    a = file_values
    s = a[idx][0][0]
    f_size = a[idx][1] # number of lines in file / 4
    chunksize = a[idx][2]
    
    docs_idx = []
    index = dict()
    
    r = math.floor((f_size)/chunksize)
    m = (f_size) % chunksize + 1
    counter = 0
    
    for i in range(r+1): 
        
        if (i == r): chunk, ptr = read_chunk(s, m)
        else: chunk, ptr = read_chunk(s, chunksize)
        s = ptr
        
        indexer(chunk, index, docs_idx)

        if(checkMemoryFull()):
            write_index(index, out_dir + '/i' + str(idx) + '_' + str(counter) + '.json')
            counter += 1

            # write to document index
            with open(out_dir + '/document_index_' + str(idx) + '.bin', 'w') as w:
                w.write(str(docs_idx))

            # clean variables
            docs_idx = []
            index = dict()
            
    write_index(index, out_dir + '/i' + str(idx) + '_' + str(counter) + '.json')
    counter += 1

    # write to document index
    with open(out_dir + '/document_index_' + str(idx) + '.bin', 'w') as w:
        w.write(str(docs_idx))

    return counter
    


# ### Read corpus
# Reads the corpus in parts, 'chunk by chunk'. In each step, divides a chunk between various threads, that will each return a parcial result. Then, merges these parcial results into one partial index and writes it on disk. Repeats this process until all corpus has been consumed.
def get_file_values(chunksize):

    # auxiliar variables
    global out_dir
    
    with open('corpus.jsonl', 'r') as f:
        num_lines = sum(1 for line in f)

    size = math.floor(num_lines / 4)

    end = size+1
    counter = 1
    pos = []
    pos.append(0)

    with open('corpus.jsonl', 'rb') as f:
        for i, line in enumerate(f):
            if (i == end):
                p = f.tell()
                pos.append(p)
                end = end + size + 1
                counter += 1
                if (counter == 4): break

    
    return [(p, size, chunksize) for p in zip(pos)]
    
    
def create_threads():
    
    pool = multiprocessing.Pool()
    
    args = range(4)
    async_results = [pool.map_async(task, (arg,)) for arg in args]
    results = [r.get() for r in async_results]

    return results



# order files by token
def order_files(idx, num_files):
    
    global out_dir
    for i in range(num_files):
        count = 0
        d = dict()
        with open(out_dir + '/i' + str(idx) + '_' + str(i) + '.json', 'r' ) as f:
            for line in f:
                line = line[:-2]
                line = '{' + str(line) + '}'
                aux = dict(eval(line))
                k = list(aux.keys())[0]
                d[k] = aux[k]
             
        # order keys
        keys = list(d.keys())
        keys = sorted(set(keys))
        
        d_ordered = dict()
        for k in keys:
            d_ordered[k] = d[k]
            
        # write to disk
        write_index(d_ordered, out_dir + '/i' + str(idx) + '_' + str(i) + '.json')

        
        
# # Part 2 - External merge sort
# 
# For merging all the partial indexes that were written to disk
# 
# The merging process follows these steps:
# 
# 1) Read first chunk of each file and save into dictionaries
# 
# 2) Group dictionaries keys and order them
# 
# 3) Merge dictionaries
# 
# 4) When a dict is completly consumed, pause the process, read new chunk from respective file and reload the dict
# 
# 5) Restart the process until all files have been completly consumed
# 
# The result will be written into a binary file named **'inverted_index.bin'**, a hash map and frequency information will be calculated and written into files **'term_lexicon.bin'** and **'document_index.bin'**




# ### Write files for inverted index, document index and term lexicon
# **Inverted index**: keeps only the documents ids and the number of term apperances in that each document
# 
# **Document index**: keeps the frequency of each term in the intire index (number of files where the term appears)
# 
# **Term lexicon**: keeps the offsets of terms in the Inverted Index file
# 
# These files will be written in multiple executions of this function.

def write_final_files(index):
    global out_dir
    with open(out_dir + '/inverted_index.bin', 'ab') as inv_i, open(out_dir + '/term_lexicon.bin', 'ab') as lex:
        
        num_lists = 0
        for term in index.keys():
            
            num_lists += 1
            
            offset_init = inv_i.tell()

            # write inverted index
            docs = str(index[term])[1:-1]
            docs = docs.replace('], [', '][')
            docs_in_bytes = docs.encode('utf-8')
            inv_i.write(docs_in_bytes)

            offset_end = inv_i.tell()

            # write term lexicon
            term = term.replace("'", "")
            s = term + str(offset_init) + '|' + str(offset_end) + '|'
            lex.write(s.encode('utf-8'))


    return num_lists

def write_doc_index():
    
    with open(out_dir + '/document_index_0.bin', 'r') as f:
        file = ''
        file += f.read()
        
    with open(out_dir + '/document_index.bin', 'w') as o:
        o.write(file)
        
        
    for i in range(1,4):
        with open(out_dir + '/document_index_' + str(i) + '.bin', 'r') as f:
            file = ''
            file += f.read()
        
        with open(out_dir + '/document_index.bin', 'a') as o:
            o.write(file)
        
        
        
def read_file_chunk(filename, start, chunk_size):
    
    if (start == -1): return {}, -1
    
    with open(filename, 'rb') as f:
        f.seek(start)

        # Read the file in chunks
        chunk = list(itertools.islice(f, chunk_size))

        if not chunk: return {}, -1

        d = {}
        for line in chunk:
            line = line.decode('utf-8')
            line = '{' + str(line) + '}'
            aux = dict(eval(line))
            k = list(aux.keys())[0]
            d[k] = aux[k]

        ptr = f.tell()
           
    return d, ptr




def process_keys(dicts, keys = []):
    
    # save all keys into a unique array
    for d in dicts:
        keys = keys + list(d.keys())

    # order keys and remove duplicates
    keys = sorted(set(keys))
    
    return keys



# ### Merge
# Executes the process of reading chunks from each partial index inside the files, merging them together and writing them to disk, respecting the memory limit.
def k_merge(dicts, keys, limit_size, result, filename, p_flag, num_lists):
    
    global memory_limit
    aux = []
    empty = -1
    
    
    for k_i, k in enumerate(keys):

        if (empty != -1):
            keys = keys[k_i:]
            return empty, keys
       
        for i, d in enumerate(dicts):
            if (k in d):
                aux = aux + d[k]
                d.pop(k)
                
                # check if an index was completly consumed
                if (not d): empty = i
        
        result[k] = aux
        aux = []
       

    
    # write final values
    if (p_flag): write_index(result, filename)
    else: 
        n = write_final_files(result) 
        num_lists.append(n)
            
    
    return None, None     
  
    
def external_merge(idx, num_files, p_flag):
    # variables
    global out_dir, memory_limit
    limit = memory_limit * 0.9
    dicts = []
    f_ptr = []
    result = dict()
    chunk_size = math.floor((memory_limit/10) / (num_files + 1))
    limit_size = chunk_size
    
    num_lists = []
    
    ## START READING FILES
    for i in range(num_files):
        
        if (p_flag): filename = out_dir + '/i' + str(idx) + '_' + str(i) + '.json'
        else: filename = out_dir + '/i' + str(i) + '.json'
        
        d, p = read_file_chunk(filename, 0, chunk_size)
        f_ptr.append(p)  # files pointers keep position where to start reading file
        dicts.append(d)

  
    ## ORDER KEYS
    keys = process_keys(dicts)
    
    files_left = num_files
    while (files_left > 0):
    
        ## MERGE DICTIONARIES
        if (p_flag): filename = out_dir + '/i' + str(idx) + '_' + str(i) + '.json'
        else: filename = out_dir + '/i' + str(i) + '.json'
        
        empty_d, keys = k_merge(dicts, keys, limit_size, result, filename, p_flag, num_lists)
        
    
        if (not empty_d and not keys): break
        
        # write the current result to disk if result is full
        process = psutil.Process() 
        if (process.memory_info().rss >= limit):
            n = write_final_files(result)
            num_lists.append(n)
            result = dict()
        
        
 
        ## RELOAD EMPTY DICTIONARY with chunk from file
        dicts[empty_d], f_ptr[empty_d] = read_file_chunk(out_dir + '/i' + str(idx) + '_' + str(empty_d) + '.json', f_ptr[empty_d], chunk_size)
        
          
        if (not dicts[empty_d]): files_left -= 1 
        else: keys = process_keys([dicts[empty_d]], keys)
    
    return num_lists


MEGABYTE = 1024 * 1024
def memory_limit(value):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def main():
    """
    Your main calls should be added here
    """
    
    start_time = time.time()
    
    global corpus_path, memory_limit, out_dir
    
    ### INDEXER
    num_files = create_threads()
    
    
    ### MERGE
    
    # order each process file and merge then into 4 bigger files
    for i in range(4):
        order_files(i, num_files[i][0])
        num_lists = external_merge(i, num_files[i][0], True)
    
    for i in range(4):
        for j in range(4):
            os.remove(out_dir + '/i' + str(i) + '_' + str(j) + '.json')


    n = []
    n = external_merge(0, 4, False)
    
    for j in range(4):
        os.remove(out_dir + '/i' + str(j) + '.json')
    
    ### FINAL VALUES CALCULATIONS
    num_lists = 0
    for i in range(4):
        num_lists = num_lists + int(n[i])
        
    write_doc_index()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    index_size = os.path.getsize(out_dir + '/inverted_index.bin')
    
    avg_list_size = 0
    
    print('{ "Index Size":', index_size,',\n"Elapsed Time":', elapsed_time,',\n"Number of Lists":',num_lists,',\n"Average List Size":',avg_list_size ,'}')


    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-m',
        dest='memory_limit',
        action='store',
        required=True,
        type=int,
        help='memory available'
    )
    
    # adding new args
    parser.add_argument(
        '-c',
        dest='corpus_path',
        action='store',
        required=True,
        type=str,
        help='corpus path'
    )
    parser.add_argument(
        '-i',
        dest='out_dir',
        action='store',
        required=True,
        type=str,
        help='path to index directory'
    )
    
    args = parser.parse_args()
    memory_limit(args.memory_limit)
    
    
    file_values = get_file_values(1000)
    
    memory_limit = args.memory_limit*1024*1024
    out_dir = args.out_dir
    corpus_path = args.corpus_path
    
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)

