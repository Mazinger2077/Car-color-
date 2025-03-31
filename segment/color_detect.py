import numpy as np
import cv2
import tensorflow.compat.v1 as tf

class ColorClassifier:
    def __init__(self, model_file, label_file, input_layer, output_layer, input_size):
        """
        Initialize the color classifier
        
        Args:
            model_file (str): Path to the TensorFlow model file
            label_file (str): Path to the labels file
            input_layer (str): Name of the input layer
            output_layer (str): Name of the output layer
            input_size (tuple): Expected input image size (width, height)
        """
        # Disable TensorFlow 2.x behavior
        tf.disable_v2_behavior()
        
        # Load the graph
        self.graph = self._load_graph(model_file)
        
        # Load labels
        self.labels = self._load_labels(label_file)
        
        # Prepare input and output operations
        input_name = f"import/{input_layer}"
        output_name = f"import/{output_layer}"
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        
        # Create session
        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()
        
        # Store configuration
        self.input_size = input_size

    def _load_graph(self, model_file):
        """
        Load TensorFlow graph from a .pb file
        
        Args:
            model_file (str): Path to the model file
        
        Returns:
            tf.Graph: Loaded TensorFlow graph
        """
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def _load_labels(self, label_file):
        """
        Load labels from a text file
        
        Args:
            label_file (str): Path to the labels file
        
        Returns:
            list: List of labels
        """
        with open(label_file, "r", encoding='cp1251') as f:
            return [line.strip() for line in f]

    def predict(self, image):
        """
        Predict the color of a vehicle
        
        Args:
            image (numpy.ndarray): Input image of the vehicle
        
        Returns:
            tuple: (predicted color, confidence)
        """
        # Ensure input is RGB
        image = image[:, :, ::-1]
        
        # Resize image to expected input size
        image = cv2.resize(image, self.input_size)
        
        # Prepare image for tensorflow
        image = np.expand_dims(image, axis=0)
        
        # Normalize image
        image = image.astype(np.float32)
        image /= 127.5
        image -= 1.
        
        # Run inference
        results = self.sess.run(
            self.output_operation.outputs[0], 
            {self.input_operation.outputs[0]: image}
        )
        results = np.squeeze(results)
        
        # Get top prediction
        top_index = results.argmax()
        
        return self.labels[top_index], float(results[top_index])

def create_color_classifier(
    model_file='segment\color_detection\model-weights-spectrico-car-colors-recognition-mobilenet_v3-224x224-180420.pb',
    label_file='segment\color_detection\labels.txt',
    input_layer='input_1',
    output_layer='Predictions/Softmax/Softmax',
    input_size=(224, 224)
):
    """
    Convenience function to create a ColorClassifier with default parameters
    
    Args:
        model_file (str, optional): Path to model file
        label_file (str, optional): Path to labels file
        input_layer (str, optional): Input layer name
        output_layer (str, optional): Output layer name
        input_size (tuple, optional): Input image size
    
    Returns:
        ColorClassifier: Initialized color classifier
    """
    return ColorClassifier(
        model_file, 
        label_file, 
        input_layer, 
        output_layer, 
        input_size
    )

# Example usage
if __name__ == "__main__":
    # Create classifier
    color_classifier = create_color_classifier()
    
    # Load an example image
    import cv2
    
    # Replace with your own test image path
    test_image = cv2.imread('test_car.jpg')
    
    if test_image is not None:
        # Predict color
        color, confidence = color_classifier.predict(test_image)
        print(f"Color: {color}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print("Could not load test image")