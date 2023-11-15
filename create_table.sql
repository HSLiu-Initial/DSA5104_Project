-- Users table remains unchanged
CREATE TABLE Users (
    user_id INT PRIMARY KEY,
    age_range INT,
    gender INT
);

-- Items table without user_id
CREATE TABLE Items (
    item_id INT PRIMARY KEY,
    cat_id INT,
    brand_id INT
    -- user_id removed, so no foreign key reference to Users table here
);

-- UserActions table with added user_id
CREATE TABLE UserActions (
    action_id INT PRIMARY KEY,
    user_id INT,
    item_id INT,
    action_type INT,
    time_stamp TIMESTAMP,
    FOREIGN KEY (item_id) REFERENCES Items(item_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id) -- New foreign key added here
);

-- Seller table remains unchanged
CREATE TABLE Seller (
    seller_id INT,
    item_id INT,
    PRIMARY KEY (seller_id, item_id),
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);

-- FeatureMapping table remains unchanged
CREATE TABLE FeatureMapping (
    mapping_id INT PRIMARY KEY,
    feature_name VARCHAR(255),
    original_value VARCHAR(255),
    mapped_value INT
);

-- NegativeSamples table remains unchanged
CREATE TABLE NegativeSamples (
    sample_id INT PRIMARY KEY,
    item_id INT,
    label BOOLEAN,
    FOREIGN KEY (item_id) REFERENCES Items(item_id)
);


CREATE TABLE Sequences (
    seq_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action_id INT,
    seq_rank INT,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (action_id) REFERENCES UserActions(action_id)
);



-- SearchingPool table remains unchanged
CREATE TABLE SearchingPool (
    pool_id INT PRIMARY KEY,
    action_id INT,
    FOREIGN KEY (action_id) REFERENCES UserActions(action_id)
);

-- Result table remains unchanged
CREATE TABLE Result (
    search_action_id INT,
    pool_id INT,
    PRIMARY KEY (search_action_id, pool_id),
    FOREIGN KEY (search_action_id) REFERENCES UserActions(action_id),
    FOREIGN KEY (pool_id) REFERENCES SearchingPool(pool_id)
);
