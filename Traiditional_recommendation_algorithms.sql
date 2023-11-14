use tmall;

-- User-based Collaborative Filtering Algorithm
-- Item clicked by user 5475
WITH User1ClickedItems AS (
    SELECT item_id
    FROM UserActions
    WHERE user_id = 5475 AND action_type = 1519052
)

-- Find other users similar to the specified user
, SimilarUsers AS (
    SELECT ua.user_id, COUNT(DISTINCT ub.item_id) AS common_items
    FROM UserActions ua
    JOIN UserActions ub ON ua.item_id = ub.item_id
    WHERE ua.user_id <> ub.user_id AND ua.action_type = 1519052 AND ub.action_type = 1519052
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



-- Item-based Collaborative Filtering Algorithm
-- Choose features for collaborative filtering
WITH ItemFeatures AS (
    SELECT item_id, cat_id, brand_id
    FROM Items
)

-- Calculate similarity between items
, ItemSimilarities AS (
    SELECT
        a.item_id AS item1,
        b.item_id AS item2,
        COUNT(DISTINCT a.cat_id) + COUNT(DISTINCT a.brand_id) AS similarity_score
    FROM ItemFeatures a
    JOIN ItemFeatures b ON a.item_id <> b.item_id
    GROUP BY a.item_id, b.item_id
)

-- Suppose we want to recommend similar products for the product with item_id = 871678
, RecommendedItems AS (
    SELECT
        iss.item2 AS recommended_item,
        SUM(iss.similarity_score) AS total_similarity
    FROM ItemSimilarities iss
    WHERE iss.item1 = 871678
    GROUP BY iss.item2
    ORDER BY total_similarity DESC
    LIMIT 5 
)

-- Get recommended item information
SELECT ri.recommended_item, i.cat_id, i.brand_id
FROM RecommendedItems ri
JOIN Items i ON ri.recommended_item = i.item_id;




-- Recommendations Based on Similarity
-- Get the clicked products of user with user_id 226009
WITH UserClickedItems AS (
    SELECT DISTINCT item_id
    FROM UserActions
    WHERE user_id = 226009 AND action_type = 1519052
)

-- Calculate Jaccard Similarity
, JaccardSimilarity AS (
    SELECT
        a.user_id AS user1,
        b.user_id AS user2,
        COUNT(DISTINCT a.item_id) AS common_items,
        COUNT(DISTINCT b.item_id) AS total_items,
        COUNT(DISTINCT a.item_id) / COUNT(DISTINCT b.item_id) AS jaccard_similarity
    FROM UserActions a
    JOIN UserActions b ON a.item_id = b.item_id
    WHERE a.user_id < b.user_id AND a.action_type = 1519052 AND b.action_type = 1519052
    GROUP BY a.user_id, b.user_id
)

-- Obtain 
SELECT
    226009 AS user_to_recommend_for,
    js.user2 AS similar_user,
    ua.item_id AS recommended_item
FROM UserClickedItems uc
JOIN JaccardSimilarity js ON 226009 = js.user1
JOIN UserActions ua ON js.user2 = ua.user_id AND ua.action_type = 1519052
LEFT JOIN UserClickedItems uc_actual ON ua.item_id = uc_actual.item_id
WHERE uc_actual.item_id IS NULL
ORDER BY js.jaccard_similarity DESC
LIMIT 5;



-- Recommendations Based on Graph
-- Set target_user
SET @target_user = 226009;

-- Create a recursive query to find users with similar behavior
WITH RECURSIVE UserRecommendations AS (
    SELECT
        u1.user_id AS source_user,
        u2.user_id AS target_user,
        u2.item_id AS recommended_item,
        1 AS depth
    FROM UserActions u1
    INNER JOIN UserActions u2 ON u1.item_id = u2.item_id
    WHERE u1.user_id = @target_user 
    AND u1.user_id <> u2.user_id
    AND u1.action_type = 1519052
    UNION ALL
    SELECT
        ur.source_user,
        ur.target_user,
        u.item_id AS recommended_item,
        ur.depth + 1
    FROM UserRecommendations ur
    INNER JOIN UserActions u ON ur.target_user = u.user_id
    WHERE ur.depth < 3 -- Set the recursion depth and adjust it as needed
)
-- Find recommended items
SELECT DISTINCT source_user, target_user, recommended_item
FROM UserRecommendations
WHERE depth = 1; -- Set recommendated depth









