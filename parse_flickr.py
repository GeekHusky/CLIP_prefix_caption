import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from random import randint

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

string_sim_threshold = 0.05

nltk.download('punkt') # if necessary...

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def string_dist(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/flickr/{clip_model_name}_train.pkl"
    annotation_path = "./data/flickr/captions.txt"
    images_dir = "./data/flickr/Images"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    dataset = []

    caps = []

    with open(annotation_path, 'r') as f:
        # header
        f.readline()
        line = f.readline()

        index = 1
        while line:

            if index % 5 == 1:

                if len(caps) != 0:
                    dataset.append({"embedding": prefix, 
                                    "captions": caps})
                    caps = []
                
                [img_path, cap] = line.split(sep=",", maxsplit=1) 

                img_path = os.path.join(images_dir, img_path)
                
                image = io.imread(img_path)
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    prefix = clip_model.encode_image(image).cpu()
                
            else:
                [_, cap] = line.split(sep=",", maxsplit=1) 

            cap = cap.strip()
            cap = cap.strip('\"')
            caps.append({"caption": cap, "label": 1})

            # all_captions.append(cap)
            # all_embeddings.append(prefix)

            index += 1

            line = f.readline()

    # generate the hard negative sample
    for i, data in enumerate(dataset):
        
        i_captions = data["captions"]

        contrastive_captions = []
        index_tracker = set()

        while len(contrastive_captions) < 6:
            j = randint(0, len(dataset)-1)

            if j == i:
                continue

            j_captions = dataset[j]["captions"]
            j_cap_idx = randint(0, 4)
            j_cap = j_captions[j_cap_idx]

            for i_cap in i_captions:
                if string_dist(i_cap, j_cap) > string_sim_threshold:
                    continue

                if (j, j_cap_idx) in index_tracker:
                    continue

            contrastive_captions.append({"caption": j_cap, "label": 0})
            index_tracker.add(j, j_cap_idx)

        data["captions"] += contrastive_captions



    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)

    print('Done')
    print("%0d embeddings saved " % len(dataset))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
