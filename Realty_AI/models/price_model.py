import numpy as np

class PriceModel:
    def __init__(self):
        pass
        
    def predict(self, input_data):
        """
        Simulates price prediction based on dummy input features.
        input_data: dict or single row DataFrame
        """
        # Base price (e.g. 50 Lakhs)
        price = 5000000
        
        # Add random variance based on features or just random
        # Here we just mock it
        if isinstance(input_data, dict):
            sqft = input_data.get('SquareFootage', 1000)
            bedrooms = input_data.get('Bedrooms', 3)
        else:
            # Assume dataframe
            sqft = input_data['SquareFootage'].iloc[0] if 'SquareFootage' in input_data else 1000
            bedrooms = input_data['Bedrooms'].iloc[0] if 'Bedrooms' in input_data else 3
            
        price += sqft * 5000  # ₹5000 per sqft
        price += bedrooms * 500000 # ₹5L per bedroom
        
        # Add some noise
        noise = np.random.normal(0, 500000)
        return max(2000000, price + noise)
