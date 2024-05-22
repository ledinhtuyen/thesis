import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split encoder and decoder')
    parser.add_argument('--ckpt', type=str, required=True, help='path to the model')
    parser.add_argument('--encoder', type=str, required=True, help='path to the encoder')
    parser.add_argument('--decoder', type=str, required=True, help='path to the decoder')
    return parser.parse_args()
  
def split_encoder_decoder(ckpt, encoder, decoder):
    checkpoint = torch.load(ckpt)
    encoder_state_dict, decoder_state_dict = {}, {}
    
    for key, value in checkpoint['model'].items():
        if key.startswith('encoder'):
            encoder_state_dict[key] = value
        elif key.startswith('decoder'):
            decoder_state_dict[key] = value
        elif key.startswith('cls_token'):
            encoder_state_dict[key] = value
        else:
            decoder_state_dict[key] = value
      
    torch.save(encoder_state_dict, encoder)
    torch.save(decoder_state_dict, decoder)
    
if __name__ == '__main__':
    args = parse_args()
    split_encoder_decoder(args.ckpt, args.encoder, args.decoder)
