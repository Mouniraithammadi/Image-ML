
import pinecone
import numpy as np
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pinecone.init(api_key=config["api_key"],environment=config["environment"])
index = pinecone.Index(index_name=config["index_name"])

# def put(ids ,vectors):
#     ids_arr = np.array(ids)
#     arr = np.concatenate([ids_arr[:,np.newaxis],vectors],axis=1)
#     vectors_list = [(str(id_),tuple(values)) for id_,*values in arr]
# def put(ids_arr, vectors):
#     ids_arr = np.array(ids_arr)  # convert ids_arr to a NumPy array
#     ids_arr = ids_arr[:,np.newaxis]  # add a new dimension to ids_arr
#     vectors = np.array(vectors)  # convert vectors to a NumPy array
#     vectors = vectors[:,np.newaxis]  # add a new dimension to vectors
#     vectors_list = np.concatenate([ids_arr,vectors],axis=1)
#     return index.upsert(vectors=vectors_list)
# def put(ids, vectors):
#     ids_arr = np.array(ids)
#     arr = np.concatenate([ids_arr[:, np.newaxis], vectors], axis=1)
#     vectors_list = [(str(id_), tuple(values)) for id_, *values in arr]
#     return index.upsert(vectors=vectors_list)
def put(ids, vectors):
    ids_arr = np.array(ids)
    ids_arr = ids_arr[:,np.newaxis]
    vectors_flat = [np.array(values).flatten() for values in vectors]  # Flatten vectors
    arr = np.concatenate([ids_arr,vectors_flat],axis=1)
    vectors_list = [(str(id_),tuple(values)) for id_,*values in arr]
    return index.upsert(vectors=vectors_list)

def get(Vector):
    ndarray = np.array(Vector)
    # Convert to list
    vector = ndarray.tolist()
    res = index.query(queries=[vector],top_k=1)
    return res



# pinecone.delete_index("index")
# query_vector = [0, 2, 3]
# results = get(query_vector)
# print(results)

