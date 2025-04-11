import numpy as np
from six import iteritems
from collections import defaultdict

class Dataset:

    '''
    Attributes:
        ur(:obj:`defaultdict` of :obj:`list`): The users ratings. This is a
            dictionary containing lists of tuples of the form ``(item_inner_id,
            rating)``. The keys are user inner ids.
        ir(:obj:`defaultdict` of :obj:`list`): The items ratings. This is a
            dictionary containing lists of tuples of the form ``(user_inner_id,
            rating)``. The keys are item inner ids.
        n_users: Total number of users.
        n_items: Total number of items.
        rating_scale(tuple): The minimum and maximal rating of the rating
            scale.
        global_mean: The mean of all ratings.
    '''

    def __init__(self, corpus, vocab = None):

        self.vocab = vocab
        self.corpus = corpus
        self.rating_scale = (corpus['rating'].min(), corpus['rating'].max())
        self.global_mean = corpus['rating'].mean()
        self.n_reviews = corpus.shape[0]
        self.n_words = len(vocab) if vocab is not None else 0

       
        self.raw2inner_id_users = {}
        self.raw2inner_id_items = {}
        current_u_index = 0
        current_i_index = 0
        self.uratings = defaultdict(list)
        self.iratings = defaultdict(list)
        self.ureviews = defaultdict(list)
        self.ireviews = defaultdict(list)
        self.reviews = {}
        
        d = 0
        for index, row in self.corpus.iterrows():
            urid = row['user']
            irid = row['item']
            r = row['rating']
            if self.vocab is not None:
                doc = row['review']
            try:
                uid = self.raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                self.raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = self.raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                self.raw2inner_id_items[irid] = current_i_index
                current_i_index += 1
            
            if self.vocab is not None:
                self.ureviews[uid].append((d, iid, r, doc))
                self.ireviews[iid].append((d, uid, r, doc))
                self.reviews[d] = (uid, iid, r, doc)
                
            self.uratings[uid].append((d, iid, r))
            self.iratings[iid].append((d, uid, r))
            
            d += 1

        self.n_users = len(self.uratings)  # number of users
        self.n_items = len(self.iratings)  # number of items
        
        self.inner2raw_id_users = {inner: raw for (raw, inner) in iteritems(self.raw2inner_id_users)}
        self.inner2raw_id_items = {inner: raw for (raw, inner) in iteritems(self.raw2inner_id_items)}


    def all_ratings(self):
        """Generator function to iterate over all ratings.
        Yields:
            A tuple ``(d_index, uid, iid, rating)`` where ids are inner ids
        """
        for u, u_rs in iteritems(self.uratings):
            for d, i, r in u_rs:
                yield d, u, i, r
                
    def all_reviews(self):
        """Generator function to iterate over all reviews.
        Yields:
            A tuple ``(d_index, uid, iid, rating, review)`` where ids are inner ids
        """
        for u, u_rs in iteritems(self.ureviews):
            for d, i, r, doc in u_rs:
                yield d, u, i, r, doc
    
    def batch_reviews(self,current_batch):
        for d, review in iteritems(current_batch):
            u, i, r, doc = review
            yield d, u, i, r, doc 

        
    def generate_batch(self, batch_size):       
        indices = np.arange(self.n_reviews)
        np.random.shuffle(indices)
        idx = 0
        while idx + batch_size <= self.n_reviews:
            batch_indices = indices[idx : idx + batch_size]
            batch = {key : self.reviews[key] for key in batch_indices}
            idx += batch_size
            yield self.batch_reviews(batch)
        if idx < self.n_reviews:
            batch_indices = indices[idx : self.n_reviews]
            batch = {key : self.reviews[key] for key in batch_indices}
            yield self.batch_reviews(batch)


    def knows_raw_user(self, urid):
        '''
        Indicate if the user is part of the trainset.
        A user is part of the trainset if the user has at least one rating.
        '''
        return urid in self.raw2inner_id_users.keys()

    def knows_raw_item(self, irid):
        """Indicate if the item is part of the trainset.
        An item is part of the trainset if the item was rated at least once.
        """
        return irid in self.raw2inner_id_items.keys()


    def to_raw_uid(self, iuid):
        """Convert a **user** inner id to a raw id."""           
        try:
            return self.inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(str(iuid) + ' is not a valid inner id.')


    def to_raw_iid(self, iiid):
        """Convert an **item** inner id to a raw id."""
        try:
            return self.inner2raw_id_items[iiid]
        except KeyError:
            raise ValueError(str(iiid) + ' is not a valid inner id.')


    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)
    
    def match_online_corpus(self,new_corpus):
        """Generator function to iterate over online new corpus.
        Yields:
            A tuple ``(d_index, uid, iid, rating, review)`` where ids are inner ids
        """            
        for index, row in new_corpus.iterrows():
            urid = row['user']
            irid = row['item']
            r = row['rating']
            doc = row['review']
            try:
                uid = self.raw2inner_id_users[urid]
            except KeyError:
                uid = None # new user
            try:
                iid = self.raw2inner_id_items[irid]
            except KeyError:
                iid = None # new item
            yield index, uid, iid, r, doc



