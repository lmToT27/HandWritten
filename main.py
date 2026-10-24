import argparse
# TODO: Import train_model from src.train
from src.train import train_model
# TODO: Import predict_image from src.predict
from src.predict import predict_image

def main():
    parser = argparse.ArgumentParser(description="EMNIST Handwritten Recognition System using NumPy")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help="Select 'train' to train the model or 'test' to predict an image")
    # TODO: Add '--image' argument (optional) to pass the image path when in test mode
    parser.add_argument('--image', type=str, default=None,
                        help="Path to the image file (required when --mode is 'test')")

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training process...")
        # TODO: Call train_model() here
        train_model()
    elif args.mode == 'test':
        print("Starting inference mode...")
        # TODO: Check if the user provided the image path via args.image
        if args.image is None:
            print("Error: Please provide an image path using --image <path>")
            print("Example: python main.py --mode test --image examples/A.png")
        else:
            # TODO: Call predict_image(args.image) here
            predict_image(args.image)

if __name__ == "__main__":
    main()

