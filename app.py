from flask import Flask, request, render_template
from flask_restful import Api, Resource
from handlers.textModel_handler import ToxicityModel
from handlers.imageModel_handler import ImageModel
from handlers.recommendation_handler import RecommendationModel

app = Flask(__name__, template_folder='templates')
app.secret_key = 'FREE USE'
api = Api(app)

# Initialize the models
text_model = ToxicityModel()
image_model = ImageModel()
recommendation_model = RecommendationModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

class textAimodelResource(Resource):
    def get(self):
        return {'message': 'Hello, this is a GET request!'}

    def post(self):
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return {'error': 'No text provided'}, 400

            text = data['text']
            prediction = text_model.predict_toxicity(text)

            return {
                'text': text,
                'is_toxic': bool(prediction),
                'message': 'Toxic content detected' if prediction == 1 else 'Non-toxic content'
            }, 200

        except Exception as e:
            return {'error': str(e)}, 500

class ImageAiModelResource(Resource):
    def get(self):
        return {'message': 'Image classification endpoint is ready'}, 200

    def post(self):
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return {'error': 'No image provided'}, 400

            result = image_model.predict_image(data['image'])
            
            if result is None:
                return {'error': 'Error processing image'}, 500

            return {
                'prediction': {
                    'class_name': result['class_name'],
                    'confidence': result['confidence'],  # Make sure confidence is included
                    'is_harmful': bool(result['is_harmful'])
                },
                'message': 'Harmful content detected' if result['is_harmful'] else 'Non-harmful content'
            }, 200

        except Exception as e:
            return {'error': str(e)}, 500

class RecommendationResource(Resource):
    def get(self):
        return {'message': 'Recommendation endpoint is ready'}, 200

    def post(self):
        try:
            data = request.get_json()
            if not data or 'user_id' not in data:
                return {'error': 'No user_id provided'}, 400

            try:
                user_id = int(data['user_id'])
            except ValueError:
                return {'error': 'Invalid user_id format'}, 400

            recommendations = recommendation_model.get_recommendations(user_id)
            
            if recommendations is None:
                return {'error': 'Invalid user_id or error processing request'}, 400

            return {
                'user_id': user_id,
                'recommendations': recommendations
            }, 200

        except Exception as e:
            return {'error': str(e)}, 500

api.add_resource(textAimodelResource, '/checktext', endpoint='checktext')
api.add_resource(ImageAiModelResource, '/checkimage', endpoint='checkimage')
api.add_resource(RecommendationResource, '/recommend', endpoint='recommend')

def main():
    # For development, you can use a self-signed certificate
    # For production, replace with your actual certificate paths
    ssl_context = ('cert.pem', 'key.pem')
    app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context)

if __name__ == '__main__':
    main()
