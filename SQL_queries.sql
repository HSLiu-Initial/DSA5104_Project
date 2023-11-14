use tmall;

-- 1. Find the number of items sold by each seller
SELECT S.seller_id, COUNT(UA.item_id) AS sold_items_count
FROM Seller S
JOIN UserActions UA ON S.item_id = UA.item_id
WHERE UA.action_type = 1519052
GROUP BY S.seller_id
ORDER BY sold_items_count DESC;

-- 2.Get a user's click history, including product details
SELECT UA.time_stamp, I.item_id, I.cat_id, I.brand_id
FROM UserActions UA
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.user_id = 131 AND UA.action_type = 1519052;

-- 3. Find the brands clicked most by users in each age range:
SELECT U.age_range, I.brand_id, COUNT(UA.item_id) AS click_count
FROM Users U
JOIN UserActions UA ON U.user_id = UA.user_id
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052
GROUP BY U.age_range, I.brand_id
ORDER BY U.age_range, click_count DESC;

-- 4. Find the most clicked age range and category combinations
SELECT U.age_range, I.cat_id, COUNT(UA.item_id) AS click_count
FROM Users U
JOIN UserActions UA ON U.user_id = UA.user_id
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052
GROUP BY U.age_range, I.cat_id
ORDER BY click_count DESC
LIMIT 1;

-- 5. Find the number of people of each gender that clicks product from a specific brand ID 1511035
SELECT U.gender, I.brand_id, COUNT(UA.item_id) AS click_count
FROM Users U
JOIN UserActions UA ON U.user_id = UA.user_id
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052 AND I.brand_id = 1511035
GROUP BY U.gender, I.brand_id
ORDER BY click_count DESC;

-- 6. Find users who have clicked products from a certain brand and the number of times they have clicked them
SELECT UA.user_id, COUNT(UA.action_id) AS click_count
FROM UserActions UA
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052 AND I.brand_id = 1511040
GROUP BY UA.user_id
ORDER BY click_count DESC;

-- 7. Find users who clicked on a specific brand's products and list what they clicked on and when
SELECT UA.user_id, UA.item_id, UA.time_stamp
FROM UserActions UA
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052 AND I.brand_id = 1511076;

-- 8. Find the top 10 users who clicked on the most items and the number of clicked items
SELECT UA.user_id, COUNT(UA.item_id) AS click_count
FROM UserActions UA
WHERE UA.action_type = 1519052
GROUP BY UA.user_id
ORDER BY click_count DESC
LIMIT 10;

-- 9. Find the 10 most clicked items with their item_id
SELECT UA.item_id, COUNT(UA.action_id) AS click_count
FROM UserActions UA
WHERE UA.action_type = 1519052
GROUP BY UA.item_id
ORDER BY click_count DESC
LIMIT 10;

-- 10. Find users who clicked on items in a specific category and show the brands they clicked on the most
SELECT UA.user_id, I.brand_id, COUNT(UA.action_id) AS click_count
FROM UserActions UA
JOIN Items I ON UA.item_id = I.item_id
WHERE UA.action_type = 1519052 AND I.cat_id = 1504652
GROUP BY UA.user_id, I.brand_id
ORDER BY click_count DESC;

