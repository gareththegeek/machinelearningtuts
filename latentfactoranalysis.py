import implicit
import spicy 
import pandas as pd

dataFile = ".\\data\\ml-100k\\u.data"
data = pd.read_csv(
    dataFile,
    sep="\t",
    header=None,
    usecols=[0, 1, 2],
    names=["userId", "itemId", "rating"])

data["userId"] = data["userId"].astype("category")
data["itemId"] = data["itemId"].astype("category")
rating_matrix = spicy.sparse.coo_matrix((data["rating"].astype(float),
                            (data["itemId"].cat.codes.copy(),
                             data["userId"].cat.codes.copy())))

user_factors, item_factors = implicit.alternating_least_squares(rating_matrix, factors=10, regularization=0.01)

print(user_factors[196])

user196 = item_factors.dot(user_factors[196])

import heapq
recommendations = heapq.nlargest(3, range(len(user196)), user196.take)

print(recommendations)
