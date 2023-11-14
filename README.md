Implementation of DSA5104 Project

# Data Processing

Data Processing includes:

1. Data Preprocessing.
2. Indexing and queries using elsticsearch.

runing by:

`python ./codes/preprocess/run_preprocess.py --dataset tmall`

# Run some modern CTR models

For feature interaction-based methods DeepFM:

`python run_deepfm_seq.py`

For user modeling methods DIN and DIEN:

`python run_dien_seq.py`

`python run_din_seq.py`

For sample-level retrieval-based methods RIM:

`python run_rim_seq.py`

#  SQL Query

There are 10 MySQL query instances inside `SQL_queries.sql`, run them in the MySQL can get the query result.\

Here's one query example:

```sql
-- 1. Find the number of items sold by each seller
SELECT S.seller_id, COUNT(UA.item_id) AS sold_items_count
FROM Seller S
JOIN UserActions UA ON S.item_id = UA.item_id
WHERE UA.action_type = 1519052
GROUP BY S.seller_id
ORDER BY sold_items_count DESC;
```

The query result is:

| **seller_id** | **sold_items_count** |
| ------------- | -------------------- |
| 1509276       | 93                   |
| 1507865       | 87                   |
| 1507981       | 86                   |
| ...           | ...                  |

# MongoDB Query
There are 10 MongoDB query instances inside `MongoDB Queries.txt`, run them in MongoShell to get the query result.\

Here'e one query example:

```bash
db.seller.aggregate([
  {
    $project: {
      _id: 0,
      user_id: "$_id",
      numberOfItemsSold: { $size: "$items_sold" }
    }
  },
  { $sort: { numberOfItemsSold: -1 } }
]);

db.users.find(
  { user_id: "131" },
  {
    time_stamp: 1,
    "items._id": 1,
    "items.cat_id": 1,
    "items.brand_id": 1
  }
);
```

The query result is:

[
  { user_id: '1509276', numberOfItemsSold: 93 },
  { user_id: '1507865', numberOfItemsSold: 87 },
  { user_id: '1507981', numberOfItemsSold: 86 },
  { user_id: '1508160', numberOfItemsSold: 80 },
  { user_id: '1510433', numberOfItemsSold: 75 }
  ...
]



# Traiditional Recommendation Algorithms

There are 4 traditional recommendation algorithms inside `Traiditional_recommendation_algorithms.sql`, run them in the MySQL can get the query result as the result of recommendation strategy.

Here is one algorithm detail:

```sql
-- User-based Collaborative Filtering Algorithm
-- Item clicked by user 5475
WITH User1ClickedItems AS (SELECT item_id
                           FROM UserActions
                           WHERE user_id = 5475
                             AND action_type = 1519052)

-- Find other users similar to the specified user
   , SimilarUsers AS (SELECT ua.user_id, COUNT(DISTINCT ub.item_id) AS common_items
                      FROM UserActions ua
                               JOIN UserActions ub ON ua.item_id = ub.item_id
                      WHERE ua.user_id <> ub.user_id
                        AND ua.action_type = 1519052
                        AND ub.action_type = 1519052
                      GROUP BY ua.user_id
                      ORDER BY common_items DESC
    LIMIT 5
    )

-- Get the products that these users have clicked but the specified user has not clicked
SELECT su.user_id, ua.item_id
FROM SimilarUsers su
         JOIN UserActions ua ON su.user_id = ua.user_id
         LEFT JOIN User1ClickedItems u1ci ON ua.item_id = u1ci.item_id
WHERE u1ci.item_id IS NULL;
```

The query result is:

| **user_id** | **item_id** |
| ----------- | ----------- |
| 331151      | 1411783     |
| 331151      | 1023499     |
| 399377      | 673444      |
| ...         | ...         |




