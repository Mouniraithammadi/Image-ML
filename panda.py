import pandas as pd




df = pd.DataFrame(
    data={
        "id":["a","b","c","d"],
        "vector":[
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,22]
        ]
    }
)
print(df.id)