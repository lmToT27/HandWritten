import argparse
# TODO: Import train_model from src.train
# TODO: Import predict_image from src.predict

def main():
    parser = argparse.ArgumentParser(description="EMNIST Handwritten Recognition System using NumPy")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, 
                        help="Select 'train' to train the model or 'test' to predict an image")
    # TODO: Add '--image' argument (optional) to pass the image path when in test mode
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training process...")
        # TODO: Call train_model() here
        pass
    elif args.mode == 'test':
        print("Starting inference mode...")
        # TODO: Check if the user provided the image path via args.image
        # TODO: Call predict_image(args.image) here
        pass

if __name__ == "__main__":
    main()

