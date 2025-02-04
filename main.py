import argparse
from train import train
from test import test
from data_loader import load_data
from LGDAAN_Net import LGDAAN_Net

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Test LGDAAN_Net model.")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_args()

    source_time_data, source_spectral_data, target_time_data, target_spectral_data, source_labels, target_labels = load_data(
        './preprocess/DEAP_temp_sour', './preprocess/DEAP_spec_sour', './preprocess/DEAP_temp_targ', './preprocess/DEAP_spec_targ')

    model = LGDAAN_Net()

    if args.mode == 'train':
        train(model, source_time_data, source_spectral_data, target_time_data, target_spectral_data, source_labels, args.epochs, args.batch_size, args.learning_rate)

    elif args.mode == 'test':
        model.load_weights("LGDAAN_Net_weights")
        test(model, target_time_data, target_spectral_data)

if __name__ == '__main__':
    main()
