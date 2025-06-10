import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import os

class RecommendationModel:
    def __init__(self):
        try:
            print("Loading recommendation model...")
            # Get the absolute path to the project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.dataset_dir = os.path.join(project_root, "models_and_csvs")
            
            if not os.path.exists(self.dataset_dir):
                os.makedirs(self.dataset_dir)
                raise FileNotFoundError(
                    f"Directory not found. Please place SocialMediaUsersDataset.csv in {self.dataset_dir}"
                )
                
            self.original_data, self.similarity_matrix = self._load_and_process_data()
            print("Successfully loaded recommendation model")
        except Exception as e:
            print(f"Error loading recommendation model: {str(e)}")
            raise

    def _load_and_process_data(self):
        dataset_path = os.path.join(self.dataset_dir, "SocialMediaUsersDataset.csv")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found. Please place SocialMediaUsersDataset.csv in {self.dataset_dir}"
            )
            
        dataset = pd.read_csv(dataset_path)
        dataset = dataset.head(10000)  # Limit for performance

        # Process features
        interests = dataset['Interests'].str.get_dummies(', ')
        gender_encoded = pd.get_dummies(dataset[['Gender']], dtype=int)
        
        # Age calculation
        dob = pd.to_datetime(dataset['DOB'], errors='coerce')
        current_date = datetime.now()
        dataset['Age'] = (current_date - dob).dt.days / 365.25

        # Location encoding
        location_encoded = pd.get_dummies(dataset[['City', 'Country']], dtype=int)

        # Combine features
        features = pd.concat([
            interests, 
            gender_encoded, 
            dataset[['Age']], 
            location_encoded
        ], axis=1).fillna(0)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(features)
        return dataset, similarity_matrix

    def get_recommendations(self, user_id):
        try:
            if not (1 <= user_id <= len(self.original_data)):
                return None

            similar_users_indices = self.similarity_matrix[user_id - 1].argsort()[::-1]
            top_similar_users = similar_users_indices[1:6]  # Exclude self

            recommendations = []
            for idx in top_similar_users:
                user = self.original_data.iloc[idx]
                recommendations.append({
                    'user_id': int(idx + 1),
                    'name': str(user['Name']),
                    'similarity_score': float(self.similarity_matrix[user_id - 1][idx])
                })

            return recommendations

        except Exception as e:
            print(f"Error in recommendations: {str(e)}")
            return None
