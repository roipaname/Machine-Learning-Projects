import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import DatabaseConnection
logging.basicConfig(level=logging.INFO)

class PricePredictor:
    """Make Predictions using trained models"""
    def __init__(self,model_name='Random Forest'):
        self.db=DatabaseConnection()
        self.db.connect()
        self.model_name=model_name
        self.model=self.load_model()
    def load_model(self):
        """Load trained model from disk"""
        model_filename=f"{self.model_name.lower().replace(' ','_')}_v1.pkl"
        model_path=os.path.join('trained_models',model_filename)

        if not os.path.exists(model_path):
            logging.error(f"model {self.model_name} not found")
            raise FileNotFoundError(f"Model not found : {model_path}")
        with open(model_path,'rb') as f:
            model=pickle.load(f)

        logging.info(f" Loaded model : {self.model_name}")
        return model
    def get_test_data(self,limit=100):
        """Get test data from database"""
        query = f"""
        SELECT 
            f.*,
            p.price as actual_price,
            p.city,
            p.state,
            p.bed,
            p.bath,
            p.house_size
        FROM processed.features f
        JOIN raw.properties p ON f.property_id = p.property_id
        WHERE p.price IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """
        df=self.db.read_sql(query)
        return df
    def prepare_features(self,df):
        """prepare features for prediction"""
        exclude_cols=['property_id','actual_price','created_at','price_category','state','city','bed','bath','house_size','house_age']
        feature_cols=[col for col in df.columns if col not in exclude_cols]
        X=df[feature_cols].fillna(df[feature_cols].median())
        return X, feature_cols
    def make_predictions(self,save_to_db:True):
        """Make predictions on test data"""
        print(f"\n{'='*60}")
        print(f"Making Predictions with {self.model_name}")
        print(f"{'='*60}\n")

        df=self.get_test_data()

        logging.info(f"Loaded {len(df) } test properties")

        X,feature_cols=self.prepare_features(df)

        predictions=self.model.predict(X)

        actual=df['actual_price'].values
        errors=predictions-actual
        abs_errors=np.abs(errors)
        pct_errors=(abs_errors/actual)*100


         # Create results dataframe
        results = pd.DataFrame({
            'property_id': df['property_id'],
            'city': df['city'],
            'state': df['state'],
            'bed': df['bed'],
            'bath': df['bath'],
            'house_size': df['house_size'],
            'actual_price': actual,
            'predicted_price': predictions,
            'error': errors,
            'abs_error': abs_errors,
            'pct_error': pct_errors
        })

        # Display statistics
        print("\nPrediction Statistics:")
        print(f"  Mean Absolute Error: ${abs_errors.mean():,.2f}")
        print(f"  Median Absolute Error: ${np.median(abs_errors):,.2f}")
        print(f"  Mean Percentage Error: {pct_errors.mean():.2f}%")
        print(f"  RMSE: ${np.sqrt(np.mean(errors**2)):,.2f}")
        print(f"  R² Score: {1 - (np.sum(errors**2) / np.sum((actual - actual.mean())**2)):.4f}")
        
        # Show best and worst predictions
        print("\n5 Best Predictions (lowest % error):")
        best = results.nsmallest(5, 'pct_error')[
            ['city', 'state', 'actual_price', 'predicted_price', 'pct_error']
        ]
        print(best.to_string(index=False))

        print("\n5 Worst Predictions (highest % error):")
        worst = results.nlargest(5, 'pct_error')[
            ['city', 'state', 'actual_price', 'predicted_price', 'pct_error']
        ]
        print(worst.to_string(index=False))
        
        # Save to database
        if save_to_db:
            self.save_predictions(results)
        
        return results
    
    def save_predictions(self, results):
        """Save predictions to database"""
        print("\nSaving predictions to database...")
        
        predictions_df = pd.DataFrame({
            'property_id': results['property_id'],
            'model_name': self.model_name,
            'model_version': 'v1',
            'predicted_price': results['predicted_price'],
            'actual_price': results['actual_price'],
            'prediction_error': results['error'],
            'absolute_error': results['abs_error'],
            'percentage_error': results['pct_error']
        })
        
        try:
            predictions_df.to_sql(
                'predictions',
                self.db.engine,
                schema='models',
                if_exists='append',
                index=False,
                chunksize=1000
            )
            print(f"✓ Saved {len(predictions_df)} predictions")
        except Exception as e:
            print(f"✗ Failed to save predictions: {e}")
    
    def analyze_prediction_errors(self):
        """Analyze prediction errors by different segments"""
        print(f"\n{'='*60}")
        print("Prediction Error Analysis")
        print(f"{'='*60}\n")
        
        query = """
        SELECT 
            p.state,
            COUNT(*) as num_predictions,
            AVG(pred.percentage_error) as avg_pct_error,
            AVG(pred.absolute_error) as avg_abs_error
        FROM models.predictions pred
        JOIN raw.properties p ON pred.property_id = p.property_id
        WHERE pred.model_name = %s
        GROUP BY p.state
        HAVING COUNT(*) >= 3
        ORDER BY avg_pct_error
        LIMIT 10
        """
        
        results = self.db.read_sql(query, params=(self.model_name,))
        print("Error by State (Top 10 best):")
        print(results.to_string(index=False))

if __name__ == "__main__":
    # Test predictions with different models
    models_to_test = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']
    
    for model_name in models_to_test:
        try:
            predictor = PricePredictor(model_name=model_name)
            results = predictor.make_predictions(save_to_db=True)
            predictor.analyze_prediction_errors()
        except FileNotFoundError as e:
            print(f"\nSkipping {model_name}: {e}\n")





