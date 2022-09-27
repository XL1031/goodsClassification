'''
这里面的函数是向量转化函数，参考README中的网站，里面按照本项目修改了部分内容，主体函数尽量不要变
'''
import spacy
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models, similarities
from gensim.matutils import sparse2full
import numpy as np
import math

# text2vec methods
# 以_开头的函数是类内函数调用，其余的是对象调用
class text2vec():
    def __init__(self, doc_list):
        #Initialize
        self.doc_list = doc_list
        self.nlp, self.docs, self.docs_dict = self._preprocess(self.doc_list)
    
    # Functions to lemmatise docs
    def _keep_token(self, t):
        return (t.is_alpha and 
                not (t.is_space or t.is_punct or 
                     t.is_stop or t.like_num))
    def _lemmatize_doc(self, doc):
        return [t.lemma_ for t in doc if self._keep_token(t)]


    # Gensim to create a dictionary and filter out stop and infrequent words (lemmas).
    def _get_docs_dict(self, docs):
        docs_dict = Dictionary(docs)
        # CAREFUL: For small corpus please carefully modify the parameters for filter_extremes, or simply comment it out.
        # no_lelow控制下限，即至少在no_below个文档中出现的词被保留
        # no_above控制上限，即出现在超过总文档*no_above数量的词被删除
        docs_dict.filter_extremes(no_below=2, no_above=1)
        docs_dict.compactify()
        return docs_dict

    # Preprocess docs
    def _preprocess(self, doc_list):
        #Load spacy model
        nlp  = spacy.load('en_core_web_sm')
        #lemmatise docs
        docs = [self._lemmatize_doc(nlp(doc)) for doc in doc_list] 
        #Get docs dictionary
        docs_dict = self._get_docs_dict(docs)
        return nlp, docs, docs_dict


    # Gensim can again be used to create a bag-of-words representation of each document,
    # build the TF-IDF model, 
    # and compute the TF-IDF vector for each document.
    def _get_tfidf(self, docs, docs_dict):
        docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
        model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
        docs_tfidf  = model_tfidf[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
        return docs_vecs


    #Get avg w2v for one document
    def _document_vector(self, doc, docs_dict, nlp):
        # remove out-of-vocabulary words
        doc_vector = [nlp(word).vector for word in doc if word in docs_dict.token2id]
        return np.mean(doc_vector, axis=0)


    # Get a TF-IDF weighted Glove vector summary for document list
    # Input: a list of documents, Output: Matrix of vector for all the documents
    def tfidf_weighted_wv(self):
        #tf-idf
        docs_vecs   = self._get_tfidf(self.docs, self.docs_dict)

        #Load glove embedding vector for each TF-IDF term
        tfidf_emb_vecs = np.vstack([self.nlp(self.docs_dict[i]).vector for i in range(len(self.docs_dict))])

        #To get a TF-IDF weighted Glove vector summary of each document, 
        #we just need to matrix multiply docs_vecs with tfidf_emb_vecs
        docs_emb = np.dot(docs_vecs, tfidf_emb_vecs)

        return docs_emb

    # Get average vector for document list
    def avg_wv(self):
        docs_vecs = np.vstack([self._document_vector(doc, self.docs_dict, self.nlp) for doc in self.docs])
        return docs_vecs

    # Get TF-IDF vector for document list
    def get_tfidf(self,other_documents=""):
        if other_documents == "":
            docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
            self.model_tfidf = TfidfModel(docs_corpus, id2word=self.docs_dict)
            docs_tfidf = self.model_tfidf[docs_corpus]
            docs_vecs = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_tfidf])
            return docs_vecs
        else:
            nlp, docs, docs_dict = self._preprocess(other_documents)
            docs_corpus = [self.docs_dict.doc2bow(doc) for doc in docs]
            docs_tfidf = self.model_tfidf[docs_corpus]
            docs_vecs = np.vstack([sparse2full(c,len(self.docs_dict)) for c in docs_tfidf])
            return docs_vecs

    # Get Latent Semantic Indexing(LSI) vector for document list
    def get_lsi(self, num_topics=300):
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
        model_lsi = models.LsiModel(docs_corpus, num_topics, id2word=self.docs_dict)
        docs_lsi  = model_lsi[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_lsi])
        return docs_vecs

    # Get Random Projections(RP) vector for document list
    def get_rp(self):
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
        model_rp = models.RpModel(docs_corpus, id2word=self.docs_dict)
        docs_rp  = model_rp[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_rp])
        return docs_vecs

    # Get Latent Dirichlet Allocation(LDA) vector for document list
    def get_lda(self, num_topics=100):
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
        model_lda = models.LdaModel(docs_corpus, num_topics, id2word=self.docs_dict)
        docs_lda  = model_lda[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_lda])
        return docs_vecs

    # Get Hierarchical Dirichlet Process(HDP) vector for document list
    def get_hdp(self):
        docs_corpus = [self.docs_dict.doc2bow(doc) for doc in self.docs]
        model_hdp = models.HdpModel(docs_corpus, id2word=self.docs_dict)
        docs_hdp  = model_hdp[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(self.docs_dict)) for c in docs_hdp])
        return docs_vecs

    
    
# Similarity Calculation methods
class simical():
    def __init__(self, vec1, vec2):
        self.vec1 = vec1
        self.vec2 = vec2

    def _VectorSize(self, vec) :
        return math.sqrt(sum(math.pow(v,2) for v in vec))

    def _InnerProduct(self) :
        return sum(v1*v2 for v1,v2 in zip(self.vec1,self.vec2))

    def _Theta(self) :
        return math.acos(self.Cosine()) + 10
   
    def _Magnitude_Difference(self) :
        return abs(self._VectorSize(self.vec1) - self._VectorSize(self.vec2))

    # Euclidean Distance 
    def Euclidean(self) :
        return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(self.vec1, self.vec2)))

    # Cosine Similarity 
    def Cosine(self) :
        # +1为了防止稀疏矩阵分母为0
        result = self._InnerProduct() / (self._VectorSize(self.vec1) * self._VectorSize(self.vec2) + 1)
        return result

    # Triangle’s Area Similarity (TS)
    def Triangle(self) :
        theta = math.radians(self._Theta())
        return (self._VectorSize(self.vec1) * self._VectorSize(self.vec2) * math.sin(theta)) / 2

    # Sector’s Area Similairity (SS)
    def Sector(self) :
        ED = self.Euclidean()
        MD = self._Magnitude_Difference()
        theta = self._Theta()
        return math.pi * math.pow((ED+MD),2) * theta/360
    
    # TS-SS
    def TS_SS(self) :
        return self.Triangle() * self.Sector()
