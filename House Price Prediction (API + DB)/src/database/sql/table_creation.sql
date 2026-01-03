-- Create schemas for data organization
CREATE SCHEMA raw;
CREATE SCHEMA processed;
CREATE SCHEMA models;

-- Raw property data table (matching your CSV structure)
CREATE TABLE raw.properties (
    id SERIAL PRIMARY KEY,
    property_id VARCHAR(100) UNIQUE,
    
    -- From CSV columns
    brokered_by NUMERIC,
    status VARCHAR(50),
    price NUMERIC(12,2),
    bed INTEGER,
    bath NUMERIC(3,1),
    acre_lot NUMERIC(10,4),
    street NUMERIC,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    house_size NUMERIC(10,2),
    prev_sold_date DATE,
    
    -- Metadata 
    source VARCHAR(50) DEFAULT 'Kaggle',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Store complete raw record
    raw_json JSONB
);

-- Processed features table
CREATE TABLE processed.features (
    id SERIAL PRIMARY KEY,
    property_id VARCHAR(100) REFERENCES raw.properties(property_id) ON DELETE CASCADE,
    
    -- Numerical features
    total_rooms NUMERIC(5,1),
    price_per_sqft NUMERIC(10,2),
    house_age INTEGER,
    lot_sqft NUMERIC(10,2),
    
    -- Ratios and engineered features
    bed_bath_ratio NUMERIC(5,2),
    size_to_lot_ratio NUMERIC(5,4),
    price_category VARCHAR(20),
    
    -- Encoded categorical features
    state_encoded INTEGER,
    city_encoded INTEGER,
    status_encoded INTEGER,
    
    -- Additional features
    has_prev_sale BOOLEAN,
    is_large_house BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions table
CREATE TABLE models.predictions (
    id SERIAL PRIMARY KEY,
    property_id VARCHAR(100),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    predicted_price NUMERIC(12,2),
    actual_price NUMERIC(12,2),
    prediction_error NUMERIC(12,2),
    absolute_error NUMERIC(12,2),
    percentage_error NUMERIC(5,2),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata table
CREATE TABLE models.model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    algorithm VARCHAR(100),
    hyperparameters JSONB,
    
    -- Performance metrics
    train_rmse NUMERIC(12,4),
    test_rmse NUMERIC(12,4),
    train_r2 NUMERIC(5,4),
    test_r2 NUMERIC(5,4),
    train_mae NUMERIC(12,4),
    test_mae NUMERIC(12,4),
    
    -- Tracking
    features_used TEXT[],
    training_samples INTEGER,
    test_samples INTEGER,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Create indexes for better query performance
CREATE INDEX idx_properties_city ON raw.properties(city);
CREATE INDEX idx_properties_state ON raw.properties(state);
CREATE INDEX idx_properties_status ON raw.properties(status);
CREATE INDEX idx_properties_price ON raw.properties(price);
CREATE INDEX idx_properties_created ON raw.properties(created_at);

CREATE INDEX idx_features_property ON processed.features(property_id);

CREATE INDEX idx_predictions_property ON models.predictions(property_id);
CREATE INDEX idx_predictions_model ON models.predictions(model_name, model_version);

-- Verify tables were created
\dt raw.*
\dt processed.*
\dt models.*